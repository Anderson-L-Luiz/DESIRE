# Code Assistant with DeepSeek Coder

This project provides a backend server that uses a DeepSeek Coder Large Language Model (LLM) to provide code suggestions based on a provided knowledge base. It communicates with a frontend application via WebSockets.

## How to Run

Follow the steps below to set up and run the Code Assistant.

### Setup

1.  **Create the directory structure:** Ensure you have the following directory structure:

    ```
    code-assistant/
    ├── backend/
    │   ├── knowledge_base/
    │   ├── scripts/
    │   ├── app.py
    │   ├── config.py
    │   └── requirements.txt
    └── ... (frontend files, if any)
    ```

2.  **Place your knowledge base documents:** Put your text files containing code examples, documentation snippets, etc., into the `backend/knowledge_base/` directory.

3.  **Install Python dependencies:** Navigate to the `backend/` directory in your terminal and run:

    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional) Create a `.env` file:** In the `backend/` directory, you can create a `.env` file to override default configurations in `config.py`. For example:

    ```
    LLM_API_URL=http://your_llm_server:port/generate
    ```

5.  **Crucially: Set up and run your DeepSeek Coder LLM:** You need to have a running DeepSeek Coder LLM accessible via an inference server like [Text Generation Inference (TGI)](https://huggingface.co/docs/text-generation-inference/index) or [vLLM](https://vllm.ai/). **Ensure that the API endpoint of your LLM inference server matches the `LLM_API_URL` configured in `backend/config.py` or your `.env` file.**

### Load Knowledge Base

1.  Navigate to the project root directory (`code-assistant/`).

2.  Run the `load_kb.py` script to process your knowledge base documents and populate the vector database:

    ```bash
    python backend/scripts/load_kb.py
    ```

    This step needs to be performed only once initially or whenever you significantly update the content of your knowledge base.

### Run the Backend Server

1.  Navigate to the project root directory (`code-assistant/`).

2.  Use Uvicorn to run the ASGI application:

    ```bash
    uvicorn backend.app:app --host 0.0.0.0 --port 5001 --reload
    ```

    The `--reload` flag is useful during development as it automatically restarts the server upon code changes. You should remove it for production deployments.

### Connect Frontend

You will need a separate frontend application (e.g., built with HTML/JS and potentially using libraries like Monaco Editor for code editing and Socket.IO client for WebSocket communication) to interact with the backend server.

The frontend application should perform the following actions:

1.  **Establish a Socket.IO connection:** Connect to the WebSocket server running at `ws://localhost:5001` (or the address and port where you deployed the backend).

2.  **Send `context_update` events:** When the user types in the code editor or updates the scope input, the frontend should emit a `context_update` event to the backend with the following JSON payload (use debouncing to avoid excessive requests):

    ```json
    { "code": "...", "scope": "..." }
    ```

3.  **Listen for `new_suggestion` events:** The backend will send code suggestions via `new_suggestion` events. The frontend should listen for these events and display the `data.suggestion` content to the user.

4.  **Listen for `status` and `error` events:** The backend may send `status` updates or `error` messages. The frontend should listen for these events to provide appropriate feedback to the user.