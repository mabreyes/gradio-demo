"""Hugging Face model adapter (Infrastructure Layer - implements domain interface)."""
from typing import Iterator, List, Optional, Tuple

import logging
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from config.settings import Settings, settings as default_settings
from domain.chat.entities import ChatMessage
from domain.chat.interfaces import IModelProvider


MAX_INPUT_LENGTH = 1024
MAX_NEW_TOKENS = 256
SPECIAL_TOKENS_TO_STRIP = ("<|endoftext|>", "<|im_end|>", "<|im_start|>")
DEFAULT_EMPTY_RESPONSE = "I'm sorry, I couldn't generate a response. Please try again."
ENGLISH_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Always respond in English, even if the user writes in another language."
)


class HuggingFaceModelAdapter(IModelProvider):
    """Hugging Face model adapter following SRP and DDD principles."""

    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        settings: Optional[Settings] = None,
    ):
        """Initialize the adapter with model configuration.

        Args:
            model_name: Hugging Face model name/path
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self._settings = settings or default_settings
        self.model_name = model_name or self._settings.MODEL_NAME

        # Determine device
        if device:
            self.device = device
        elif self._settings.USE_GPU and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.tokenizer = None
        self.model = None
        self._initialized = False
        self.use_chat_template = self._settings.USE_CHAT_TEMPLATE
    
    def initialize(self) -> None:
        """Initialize the model and tokenizer (Infrastructure concern)."""
        if self._initialized:
            return

        self._logger.info("Loading model: %s", self.model_name)
        self._logger.info("Using device: %s", self.device)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self._settings.HF_TOKEN,
            trust_remote_code=True  # Allow custom tokenizers for some models
        )
        
        # Check if tokenizer has chat template (for instruction-tuned models)
        has_chat_template = hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None
        if has_chat_template:
            self._logger.info("Chat template available: %s", self.use_chat_template)
            # Use chat template if available and enabled
            if not self.use_chat_template:
                self._logger.warning(
                    "Model has chat template but USE_CHAT_TEMPLATE is disabled",
                )
        else:
            self._logger.info("No chat template found, using legacy format")
            self.use_chat_template = False
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
                self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
            else:
                # Add a new pad token
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        # Use a different pad token if pad_token_id equals eos_token_id to avoid attention mask issues
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id and self.tokenizer.unk_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
            self.tokenizer.pad_token = self.tokenizer.unk_token
        
        # Load model with appropriate settings for edge devices
        # Use dtype instead of torch_dtype (torch_dtype is deprecated)
        model_kwargs = {
            "token": self._settings.HF_TOKEN,
            "trust_remote_code": True,  # Allow custom models
            "dtype": torch.float32 if self.device == "cpu" else torch.float16,
            "low_cpu_mem_usage": True,  # Optimize for low memory
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Resize token embeddings if we added a new pad token
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Move to device
        self.model.to(self.device)
        self.model.eval()
        
        # Enable optimizations for CPU inference
        if self.device == "cpu":
            # Use torch.compile for faster inference (PyTorch 2.0+)
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self._logger.info("Model compiled for faster CPU inference")
            except Exception as e:
                self._logger.exception("Could not compile model: %s", e)
        self._initialized = True
        vocab_size = (
            self.model.config.vocab_size
            if hasattr(self.model.config, "vocab_size")
            else "unknown"
        )
        self._logger.info("Model loaded successfully")
        self._logger.info("Model size: %s vocab size", vocab_size)
        self._logger.info("Using chat template: %s", self.use_chat_template)

    def _get_messages_with_current_input(
        self,
        user_input: str,
        conversation_history: List[ChatMessage],
    ) -> List[ChatMessage]:
        """Return history including the current user input exactly once.

        Args:
            user_input: Current raw user input text.
            conversation_history: Existing list of chat messages that
                represent the conversation so far.

        Returns:
            List[ChatMessage]: Conversation history that always contains the
            current user input as the most recent user message.
        """
        history_includes_current = (
            bool(conversation_history)
            and conversation_history[-1].role == "user"
            and conversation_history[-1].content == user_input
        )

        if history_includes_current:
            return list(conversation_history)

        return list(conversation_history) + [
            ChatMessage(content=user_input, role="user")
        ]

    @staticmethod
    def _strip_special_tokens(text: str) -> str:
        """Remove known special tokens without altering other content.

        Args:
            text: Raw model output text that may contain special tokens.

        Returns:
            str: Text with all known special tokens removed.
        """
        cleaned = text
        for token in SPECIAL_TOKENS_TO_STRIP:
            cleaned = cleaned.replace(token, "")
        return cleaned

    def _clean_response_text(self, text: str) -> str:
        """Normalize model output text into a user-facing response.

        Args:
            text: Raw model output text, potentially including special tokens
                and assistant role prefixes.

        Returns:
            str: Cleaned response string suitable for presentation to the
            end user.
        """
        cleaned = self._strip_special_tokens(text).strip()

        # If using chat template, some models add an assistant role prefix
        if cleaned.startswith("assistant\n") or cleaned.startswith("Assistant:"):
            cleaned = cleaned.split("\n", 1)[-1].split(":", 1)[-1].strip()

        return cleaned
    
    def _build_conversation_messages(self, user_input: str, conversation_history: List[ChatMessage]) -> List[dict]:
        """Build conversation messages in format expected by chat template.
        
        Args:
            user_input: Current user input
            conversation_history: Previous messages (may include the current user message)
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        messages: List[dict] = []

        # Always include an English-only system instruction
        messages.append(
            {
                "role": "system",
                "content": ENGLISH_SYSTEM_PROMPT,
            }
        )

        messages_to_process = self._get_messages_with_current_input(
            user_input, conversation_history
        )

        # Convert to format expected by chat template
        for msg in messages_to_process:
            role = "assistant" if msg.role == "assistant" else "user"
            messages.append({"role": role, "content": msg.content})

        return messages
    
    def _prepare_input_for_generation(
        self,
        user_input: str,
        conversation_history: List[ChatMessage],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Prepare input for model generation.

        Supports both chat template (for instruction-tuned models) and legacy format.

        Args:
            user_input: Current user input
            conversation_history: Previous messages

        Returns:
            Tuple of (input_ids, attention_mask, input_length)
        """
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            # Use chat template for modern instruction-tuned models
            messages = self._build_conversation_messages(
                user_input, conversation_history
            )

            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Tokenize the formatted prompt
            encoded = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=2048,
            )
            
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
            input_length = input_ids.shape[1]
        else:
            # Legacy format: build token sequence manually
            input_ids_list: List[int] = []

            # Prepend English-only system instruction
            system_ids = self.tokenizer.encode(
                ENGLISH_SYSTEM_PROMPT,
                add_special_tokens=False,
            )
            input_ids_list.extend(system_ids)
            input_ids_list.append(self.tokenizer.eos_token_id)

            messages_to_process = self._get_messages_with_current_input(
                user_input, conversation_history
            )

            # Build token sequence from all messages
            for msg in messages_to_process:
                msg_ids = self.tokenizer.encode(msg.content, add_special_tokens=False)
                input_ids_list.extend(msg_ids)
                input_ids_list.append(self.tokenizer.eos_token_id)

            # Convert to tensor
            input_ids = torch.tensor([input_ids_list], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            input_length = input_ids.shape[1]

        return input_ids, attention_mask, input_length

    def _resolve_generation_params(
        self,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> Tuple[float, int]:
        """Resolve generation parameters with defaults and safety limits."""
        gen_temperature = (
            temperature
            if temperature is not None
            else self._settings.TEMPERATURE
        )
        gen_max_tokens = (
            max_tokens if max_tokens is not None else self._settings.MAX_LENGTH
        )
        max_new_tokens = min(gen_max_tokens, MAX_NEW_TOKENS)
        return gen_temperature, max_new_tokens

    def generate_response(
        self,
        user_input: str,
        conversation_history: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a response using the Hugging Face model.

        Args:
            user_input: The user's message
            conversation_history: Previous messages in the conversation

        Returns:
            Generated response text
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        # Prepare input (supports both chat template and legacy format)
        input_ids, attention_mask, input_length = self._prepare_input_for_generation(
            user_input, conversation_history
        )
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Truncate input if too long (most models have max context of 2048-4096)
        if input_length > MAX_INPUT_LENGTH:
            # Keep only the most recent conversation
            input_ids = input_ids[:, -MAX_INPUT_LENGTH:]
            attention_mask = attention_mask[:, -MAX_INPUT_LENGTH:]
            input_length = input_ids.shape[1]

        gen_temperature, max_new_tokens = self._resolve_generation_params(
            temperature, max_tokens
        )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_length=1,
                temperature=gen_temperature,
                do_sample=gen_temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
                repetition_penalty=1.2,
                top_p=0.9,
                top_k=50,
                no_repeat_ngram_size=2,
            )

        # Extract only the newly generated tokens (response)
        # Remove the input tokens, keep only the generated response
        generated_ids = outputs[0][input_length:]

        # Decode and normalize only the generated response
        response = self._clean_response_text(
            self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        )

        # If response is empty or just whitespace, return a default message
        if not response:
            response = DEFAULT_EMPTY_RESPONSE

        return response

    def generate_response_stream(
        self,
        user_input: str,
        conversation_history: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Generate a streaming response using the Hugging Face model.
        
        Args:
            user_input: The user's message
            conversation_history: Previous messages in the conversation
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Yields:
            Generated response tokens/text as they're produced
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        # Prepare input (supports both chat template and legacy format)
        input_ids, attention_mask, input_length = self._prepare_input_for_generation(
            user_input, conversation_history
        )
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Truncate input if too long
        if input_length > MAX_INPUT_LENGTH:
            input_ids = input_ids[:, -MAX_INPUT_LENGTH:]
            attention_mask = attention_mask[:, -MAX_INPUT_LENGTH:]
            input_length = input_ids.shape[1]

        gen_temperature, max_new_tokens = self._resolve_generation_params(
            temperature, max_tokens
        )

        # Create streamer for streaming output
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60.0
        )

        # Generation parameters
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "min_length": 1,
            "temperature": gen_temperature,
            "do_sample": gen_temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "num_return_sequences": 1,
            "repetition_penalty": 1.2,
            "top_p": 0.9,
            "top_k": 50,
            "no_repeat_ngram_size": 2,
            "streamer": streamer
        }
        
        # Start generation in a separate thread
        generation_thread = Thread(
            target=self.model.generate, kwargs=generation_kwargs, daemon=True
        )
        generation_thread.start()
        
        # Yield tokens as they're generated
        accumulated_text = ""
        for new_text in streamer:
            if new_text:
                accumulated_text += new_text
                # Clean up special tokens
                clean_text = self._strip_special_tokens(new_text)
                if clean_text.strip():
                    yield clean_text
        
        # Final cleanup
        if accumulated_text:
            final_text = self._clean_response_text(accumulated_text)

            if not final_text:
                yield DEFAULT_EMPTY_RESPONSE
