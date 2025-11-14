# Gradio Chat Interface with Hugging Face

A Gradio-based chat interface for running Hugging Face text generation models. The focus of this project is on clean separation of layers and straightforward local or containerized setup.

## What the Application Does

- Starts a Gradio web UI for chatting with a Hugging Face model.
- Streams responses token-by-token so you see answers as they are generated.
- Lets you adjust generation settings such as temperature and maximum response length from the UI.
- Keeps a conversation history and allows you to clear it at any time.

## Prerequisites

- Python 3.11 or higher (for local development)
- Docker and Docker Compose (for containerized deployment)
- A Hugging Face account and token if you use private or gated models

## Local Setup (Virtual Environment)

1. **Create and activate a virtual environment**

   ```bash
   chmod +x setup_venv.sh
   ./setup_venv.sh
   source venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Create a `.env` file in the project root (optional but recommended):

   ```text
   MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
   MAX_LENGTH=512
   TEMPERATURE=0.7
   USE_CHAT_TEMPLATE=True
   GRADIO_SERVER_NAME=0.0.0.0
   GRADIO_SERVER_PORT=7860
   GRADIO_SHARE=False
   HF_TOKEN=your_huggingface_token_here
   USE_GPU=False
   ```

4. **Run the application**

   ```bash
   python main.py
   ```

5. **Open the UI**

   In your browser, go to `http://localhost:7860`.

## Setup with Docker Compose

1. **Create a `.env` file**

   In the project root, create a `.env` file (same variables as for local setup):

   ```text
   MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
   MAX_LENGTH=512
   TEMPERATURE=0.7
   USE_CHAT_TEMPLATE=True
   GRADIO_SHARE=False
   HF_TOKEN=your_huggingface_token_here
   USE_GPU=False
   ```

2. **Build and run**

   ```bash
   docker-compose up --build
   ```

3. **Open the UI**

   In your browser, go to `http://localhost:7860`.

## Using the Chat Interface

Once the UI is running:

- Type your message into the text box at the bottom of the chat panel.
- Click **Send** or press Enter to start generation.
- Watch the assistant response stream into the chat window.
- Adjust **Temperature** and **Max Tokens** in the right-hand panel:
  - Temperature: higher values produce more varied responses.
  - Max Tokens: caps the length of the generated answer (up to the configured limit).
- Use **Clear Conversation** to reset the history.
- Use the example prompt buttons to quickly populate the input field with sample questions.

## Configuration and Environment Variables

The application is configured through environment variables, typically set in a `.env` file.

| Variable              | Description                                       | Default                         |
|-----------------------|---------------------------------------------------|---------------------------------|
| `MODEL_NAME`          | Hugging Face model name or path                   | `Qwen/Qwen2.5-0.5B-Instruct`    |
| `MAX_LENGTH`          | Maximum response length (tokens)                  | `512`                           |
| `TEMPERATURE`         | Generation temperature                            | `0.7`                           |
| `USE_CHAT_TEMPLATE`   | Use model-specific chat template if available     | `True`                          |
| `GRADIO_SERVER_NAME`  | Server host name                                  | `0.0.0.0`                       |
| `GRADIO_SERVER_PORT`  | Port for the Gradio server                        | `7860`                          |
| `GRADIO_SHARE`        | Whether to create a Gradio public share link      | `False`                         |
| `HF_TOKEN`            | Hugging Face API token                            | `None`                          |
| `USE_GPU`             | Use GPU for inference if available                | `False`                         |

### Choosing a Model

The app can work with many Hugging Face causal language models. For local or CPU-only setups, smaller instruction-tuned models are typically the most practical, such as:

- `Qwen/Qwen2.5-0.5B-Instruct` (default, small and fast)
- `Qwen/Qwen2.5-1.5B-Instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

To change the model, set `MODEL_NAME` in your `.env` file or in the environment before running:

```bash
export MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
python main.py
```

## Architecture and Project Layout

The codebase is organized into clear layers:

- **Domain layer** (`domain/`): Core chat entities and interfaces.
- **Application layer** (`application/`): Chat service orchestrating the conversation flow.
- **Infrastructure layer** (`infrastructure/`): Hugging Face adapter and other external integrations.
- **Presentation layer** (`presentation/`): Gradio UI.
- **Configuration** (`config/`): Centralized settings.

Project structure:

```text
gradio-demo/
├── domain/
│   └── chat/
│       ├── entities.py
│       └── interfaces.py
├── application/
│   └── chat/
│       └── chat_service.py
├── infrastructure/
│   └── models/
│       └── huggingface_adapter.py
├── presentation/
│   └── gradio_interface.py
├── config/
│   └── settings.py
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── setup_venv.sh
└── README.md
```

## Troubleshooting

### Model Loading

- Check that `MODEL_NAME` is valid and available on Hugging Face.
- Ensure you have network access when the model is downloaded the first time.
- If the model is private or gated, verify that `HF_TOKEN` is set correctly.

### Memory Usage

- Use a smaller model if you see out-of-memory errors.
- Lower `MAX_LENGTH` to reduce response size.
- Set `USE_GPU=False` if GPU memory is limited or unstable.

### Docker Issues

- Ensure the Docker daemon is running.
- Check that you have enough disk space.
- Verify your Docker Compose version supports the provided configuration.

## Development

- Add tests and run them with `pytest` if you extend the project.
- Keep changes aligned with the existing layering (domain, application, infrastructure, presentation) and configuration patterns.

## License

This project is open source and available under the MIT License.
