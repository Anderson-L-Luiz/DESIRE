#!/bin/bash

# Bash script to create the project structure and populate files for the DESIRE (code-assistant) project.

# --- Configuration ---
PROJECT_DIR="DESIRE"

# --- Create Directories ---
echo "Creating directory structure..."
mkdir -p "$PROJECT_DIR/backend/knowledge_base"
mkdir -p "$PROJECT_DIR/vector_db"
mkdir -p "$PROJECT_DIR/scripts"
echo "Directory structure created."

# --- Create .gitignore ---
echo "Creating .gitignore file..."
cat << EOF > "$PROJECT_DIR/.gitignore"
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
# Usually these files are written by a python script from a template
# before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
# According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
# Pipfile.lock

# poetry
# poetry.lock

# pdm
# pdm.lock
# .pdm.toml

# PEP 582; used by PDM, PEP 582 proposals
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv/
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static analysis results
.pytype/

# Cython debug symbols
cython_debug/

# Vector DB persistent storage (adjust if needed)
vector_db/

# IDE specific files
.idea/
.vscode/
*.swp
*.swo
EOF
echo ".gitignore created."

# --- Create backend/config.py ---
echo "Creating backend/config.py..."
cat << EOF > "$PROJECT_DIR/backend/config.py"
# backend/config.py
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from a .env file if present

# --- RAG Configuration ---
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "../vector_db") # Persistent storage path for ChromaDB
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
# Alternative code-specific models: 'microsoft/codebert-base', 'Salesforce/codet5-base'
# Requires installing appropriate libraries and potentially different sentence-transformers usage.
# Consider models listed in report: Code Llama embeddings, etc.
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "code_context")
CONTEXT_CHUNK_SIZE = int(os.getenv("CONTEXT_CHUNK_SIZE", 500)) # Character size for chunking docs
CONTEXT_CHUNK_OVERLAP = int(os.getenv("CONTEXT_CHUNK_OVERLAP", 50))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 3)) # Number of context snippets to retrieve

# --- Dynamic Triggering Configuration ---
# Simple heuristics for PoC. Real system needs more sophisticated logic (AST diff, etc.)
TRIGGER_MIN_CODE_CHANGE_LEN = int(os.getenv("TRIGGER_MIN_CODE_CHANGE_LEN", 10)) # Chars added/removed
TRIGGER_MIN_SCOPE_CHANGE_LEN = int(os.getenv("TRIGGER_MIN_SCOPE_CHANGE_LEN", 5)) # Chars changed in scope

# --- LLM Configuration ---
# Assumes DeepSeek Coder is served via an API (like TGI or vLLM)
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8080/generate") # Example TGI endpoint
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 30)) # Seconds

# --- Prompt Template ---
# Clearly structure the input for the LLM
PROMPT_TEMPLATE = """
You are a helpful AI code assistant. Analyze the user's code snippet and their stated goal (scope).
Use the provided context documents, if relevant, to generate an accurate and helpful code suggestion, completion, or explanation.

**User's Current Code:**
\`\`\`python
{code}
\`\`\`

**User's Goal/Scope:**
{scope}

**Relevant Context from Knowledge Base:**
{retrieved_context}

**Your Suggestion:**
"""

# --- Backend Server ---
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5001))

# --- Other ---
# Placeholder for more advanced code analysis if needed
ENABLE_ADVANCED_CODE_ANALYSIS = os.getenv("ENABLE_ADVANCED_CODE_ANALYSIS", "False").lower() == "true"
EOF
echo "backend/config.py created."

# --- Create backend/llm_service.py ---
echo "Creating backend/llm_service.py..."
cat << EOF > "$PROJECT_DIR/backend/llm_service.py"
# backend/llm_service.py
import httpx
import logging
from backend.config import LLM_API_URL, LLM_TIMEOUT, PROMPT_TEMPLATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def format_prompt(code: str, scope: str, retrieved_context_str: str) -> str:
    """Formats the prompt using the template."""
    return PROMPT_TEMPLATE.format(
        code=code,
        scope=scope,
        retrieved_context=retrieved_context_str if retrieved_context_str else "No relevant context found."
    )

async def get_llm_suggestion(prompt: str) -> str | None:
    """
    Makes an asynchronous API call to the LLM inference server (e.g., DeepSeek Coder via TGI/vLLM).
    """
    logger.info(f"Sending request to LLM API: {LLM_API_URL}")
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,  # Adjust as needed
            "return_full_text": False, # Often we only want the generated part
            # Add other parameters like temperature, top_p as supported by the API
        }
    }
    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(LLM_API_URL, json=payload)
            response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)

            result = response.json()
            # Adjust based on the actual API response structure of your inference server
            # TGI often returns [{'generated_text': '...'}]
            if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                 suggestion = result[0]["generated_text"].strip()
                 logger.info("Suggestion received from LLM.")
                 return suggestion
            elif isinstance(result, dict) and "generated_text" in result: # Handle other possible formats
                 suggestion = result["generated_text"].strip()
                 logger.info("Suggestion received from LLM.")
                 return suggestion
            else:
                 logger.warning(f"Unexpected LLM API response format: {result}")
                 return None

    except httpx.RequestError as exc:
        logger.error(f"HTTP request failed: {exc}")
        return None
    except Exception as e:
        logger.error(f"An error occurred during LLM interaction: {e}")
        return None
EOF
echo "backend/llm_service.py created."

# --- Create backend/utils.py ---
echo "Creating backend/utils.py..."
cat << EOF > "$PROJECT_DIR/backend/utils.py"
# backend/utils.py
import logging

# Placeholder for future, more advanced code analysis (e.g., using AST)
# For now, we might just use string comparisons or regex in rag_module.py

logger = logging.getLogger(__name__)

def analyze_code_change(current_code: str, previous_code: str) -> int:
    """
    Simple heuristic: calculate the absolute difference in code length.
    Replace with more sophisticated analysis (e.g., AST diff) if needed.
    """
    return abs(len(current_code) - len(previous_code))

def analyze_scope_change(current_scope: str, previous_scope: str) -> int:
    """
    Simple heuristic: calculate the absolute difference in scope length.
    Could use fuzzy matching or semantic comparison later.
    """
    # A more robust check might involve NLP techniques or simple diff libraries
    # For now, just length difference or simple inequality check
    # return abs(len(current_scope) - len(previous_scope))
    if current_scope != previous_scope:
        return len(current_scope) # Return a value indicating change occurred
    return 0

# --- Knowledge Base Text Processing ---
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Splits text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if chunk_overlap >= chunk_size: # Prevent infinite loops
             logger.warning("Chunk overlap is >= chunk size, adjusting stride.")
             start = end # Move to the next non-overlapping block
    return chunks
EOF
echo "backend/utils.py created."

# --- Create backend/rag_module.py ---
echo "Creating backend/rag_module.py..."
cat << EOF > "$PROJECT_DIR/backend/rag_module.py"
# backend/rag_module.py
import chromadb
from chromadb.utils import embedding_functions
import logging
from backend.config import (
    VECTOR_DB_PATH, EMBEDDING_MODEL_NAME, COLLECTION_NAME,
    TRIGGER_MIN_CODE_CHANGE_LEN, TRIGGER_MIN_SCOPE_CHANGE_LEN, RETRIEVAL_K,
    ENABLE_ADVANCED_CODE_ANALYSIS
)
from backend.utils import analyze_code_change, analyze_scope_change

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
            # Use a sentence transformer model for embeddings
            # Ensure the model is suitable for code if possible/needed
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL_NAME
            )
            # Get or create the collection
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"} # Use cosine distance
            )
            logger.info(f"Connected to ChromaDB at '{VECTOR_DB_PATH}' and collection '{COLLECTION_NAME}'.")
            logger.info(f"Items in collection: {self.collection.count()}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client or collection: {e}", exc_info=True)
            raise

    def should_trigger_update(self, current_code: str, current_scope: str, previous_context: dict) -> bool:
        """
        Determines if the RAG retrieval and LLM generation should be triggered.
        Uses simple heuristics based on config thresholds.
        Requires previous context { 'code': str, 'scope': str }.
        """
        if not previous_context: # Always trigger on the first interaction
            logger.info("Triggering update: First interaction.")
            return True

        code_change = analyze_code_change(current_code, previous_context.get('code', ''))
        scope_change = analyze_scope_change(current_scope, previous_context.get('scope', ''))

        if code_change >= TRIGGER_MIN_CODE_CHANGE_LEN:
            logger.info(f"Triggering update: Code change detected (diff: {code_change}).")
            return True

        # Check scope change based on simple difference for now
        # if scope_change >= TRIGGER_MIN_SCOPE_CHANGE_LEN:
        if current_scope != previous_context.get('scope', ''): # Trigger if scope text is different
             logger.info(f"Triggering update: Scope change detected.")
             return True

        # --- Placeholder for more advanced triggers ---
        if ENABLE_ADVANCED_CODE_ANALYSIS:
             # Add logic here, e.g., check for cursor position changes,
             # significant AST node changes, new import statements etc.
             pass
        # --------------------------------------------

        logger.debug("No significant change detected, skipping update.")
        return False

    def formulate_query(self, code: str, scope: str) -> str:
        """
        Creates a query string for the vector database search.
        Simple strategy: combine scope and a snippet of code.
        Improve this with keyword extraction, code analysis (function names, libraries).
        """
        # Extract maybe the last N lines or a snippet around a potential cursor?
        # For now, just combine scope and the start of the code.
        code_snippet = code[:200] # Limit code length in query
        query = f"Goal: {scope}\nCode context: {code_snippet}"
        logger.debug(f"Formulated query: {query}")
        return query

    async def retrieve_context(self, query: str) -> str:
        """
        Queries the vector database to retrieve relevant context snippets.
        Runs potentially blocking DB call in asyncio's default executor.
        """
        if self.collection.count() == 0:
             logger.warning("Knowledge base collection is empty. Cannot retrieve context.")
             return ""
        try:
            logger.info(f"Querying vector DB with k={RETRIEVAL_K}...")
            # This is a synchronous call, ideally run in an executor for async context
            # For simplicity here, we call it directly. In a real async app, use asyncio.to_thread
            results = self.collection.query(
                query_texts=[query],
                n_results=RETRIEVAL_K,
                include=['documents', 'metadatas', 'distances'] # Include metadata and distances if needed for filtering/ranking
            )
            logger.info(f"Retrieved {len(results.get('documents', [[]])[0])} results.")

            # Simple combination of retrieved documents
            # Add filtering/re-ranking based on metadata or distances here if needed
            documents = results.get('documents', [[]])[0]
            if not documents:
                 return ""

            context_str = "\n---\n".join(documents)
            return context_str

        except Exception as e:
            logger.error(f"Error querying vector database: {e}", exc_info=True)
            return "" # Return empty string on error

# Global instance (consider dependency injection for larger apps)
try:
    rag_system = RAGSystem()
except Exception:
     rag_system = None # Handle initialization failure in app.py
EOF
echo "backend/rag_module.py created."

# --- Create backend/app.py ---
echo "Creating backend/app.py..."
cat << EOF > "$PROJECT_DIR/backend/app.py"
# backend/app.py
import asyncio
import logging
from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room
import os
import sys # Add sys import

# Set uvloop as the default event loop if available (optional performance boost)
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

# Ensure the backend directory is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import HOST, PORT # Use relative import for config
from rag_module import rag_system # Import the initialized instance
from llm_service import format_prompt, get_llm_suggestion

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24) # Replace with a strong secret in production
# Use async_mode='asgi' for compatibility with ASGI servers like Uvicorn/Hypercorn
# Ensure gevent-websocket is NOT installed if using ASGI
socketio = SocketIO(app, async_mode='asgi', cors_allowed_origins="*") # Allow all origins for dev

# Store user context (simple in-memory dictionary for this example)
# Key: session ID (sid), Value: {'code': str, 'scope': str}
user_contexts = {}

@socketio.on('connect')
def handle_connect():
    """Handles new client connections."""
    sid = request.sid
    join_room(sid) # Each client gets their own room for targeted emits
    user_contexts[sid] = {'code': '', 'scope': ''} # Initialize context
    logger.info(f"Client connected: {sid}")
    emit('status', {'message': 'Connected to Code Assistant Backend'}, room=sid)
    if rag_system is None:
         emit('error', {'message': 'Backend RAG system failed to initialize.'}, room=sid)


@socketio.on('disconnect')
def handle_disconnect():
    """Handles client disconnections."""
    sid = request.sid
    leave_room(sid)
    if sid in user_contexts:
        del user_contexts[sid]
    logger.info(f"Client disconnected: {sid}")

@socketio.on('context_update')
async def handle_context_update(data):
    """
    Handles incoming code and scope updates from the client.
    Triggers RAG and LLM processes if necessary.
    """
    sid = request.sid
    logger.info(f"Received context_update from {sid}")

    if rag_system is None:
         logger.error("RAG system not initialized. Cannot process update.")
         emit('error', {'message': 'Backend RAG system is unavailable.'}, room=sid)
         return

    if not isinstance(data, dict) or 'code' not in data or 'scope' not in data:
        logger.warning(f"Invalid data received from {sid}: {data}")
        emit('error', {'message': 'Invalid data format received.'}, room=sid)
        return

    current_code = data.get('code', '')
    current_scope = data.get('scope', '')
    previous_context = user_contexts.get(sid, {})

    # Check if an update should be triggered
    should_update = rag_system.should_trigger_update(current_code, current_scope, previous_context)

    # Update the stored context regardless of trigger, for the next check
    user_contexts[sid] = {'code': current_code, 'scope': current_scope}

    if should_update:
        logger.info(f"Processing update for {sid}...")
        emit('status', {'message': 'Processing...'}, room=sid)

        try:
            # 1. Formulate Query (potentially async if complex analysis needed)
            query = rag_system.formulate_query(current_code, current_scope)

            # 2. Retrieve Context (async)
            # Use asyncio.create_task if you want to run multiple async things concurrently
            # Use asyncio.to_thread for synchronous blocking calls like chromadb query
            retrieved_context_str = await asyncio.to_thread(rag_system.collection.query, query_texts=[query], n_results=RETRIEVAL_K, include=['documents'])
            # Process the result correctly
            documents = retrieved_context_str.get('documents', [[]])[0]
            context_str = "\n---\n".join(documents) if documents else ""


            # 3. Format Prompt (sync, but could be async if complex)
            prompt = await format_prompt(current_code, current_scope, context_str)

            # 4. Get LLM Suggestion (async)
            suggestion = await get_llm_suggestion(prompt)

            if suggestion:
                logger.info(f"Sending suggestion to {sid}")
                emit('new_suggestion', {'suggestion': suggestion}, room=sid)
            else:
                logger.warning(f"No suggestion generated for {sid}.")
                emit('status', {'message': 'Could not generate suggestion.'}, room=sid)

        except Exception as e:
            logger.error(f"Error processing update for {sid}: {e}", exc_info=True)
            emit('error', {'message': f'An error occurred: {e}'}, room=sid)
            emit('status', {'message': 'Error during processing.'}, room=sid) # Reset status
    else:
         logger.info(f"Skipping LLM update for {sid} - no significant change.")
         # Optionally send a status update indicating no change if needed
         # emit('status', {'message': 'No significant change detected'}, room=sid)


if __name__ == '__main__':
    # This is for development run only. Use Uvicorn directly for production.
    # Example: uvicorn backend.app:app --host 0.0.0.0 --port 5001 --reload
    logger.info(f"Starting Flask-SocketIO server on {HOST}:{PORT}")
    # Ensure the app object is passed correctly to socketio.run
    socketio.run(app, host=HOST, port=PORT, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
    # Note: Debug mode with reloader might cause issues with background tasks/state.
    # Use `use_reloader=False` or run directly with Uvicorn for stability.
    # Added allow_unsafe_werkzeug=True for newer Werkzeug versions with Flask dev server
EOF
echo "backend/app.py created."


# --- Create scripts/load_kb.py ---
echo "Creating scripts/load_kb.py..."
cat << EOF > "$PROJECT_DIR/scripts/load_kb.py"
# scripts/load_kb.py
import os
import sys
import logging
import chromadb
from chromadb.utils import embedding_functions

# Adjust path to import from backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from backend (relative imports might be tricky here)
try:
    from backend.config import (
        KNOWLEDGE_BASE_DIR, VECTOR_DB_PATH, COLLECTION_NAME,
        EMBEDDING_MODEL_NAME, CONTEXT_CHUNK_SIZE, CONTEXT_CHUNK_OVERLAP
    )
    from backend.utils import chunk_text
except ImportError as e:
     print(f"Error importing backend modules: {e}")
     print("Ensure the script is run from the project root or adjust sys.path accordingly.")
     sys.exit(1)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_file(filepath: str) -> tuple[str, dict]:
    """Reads a file and returns its content and basic metadata."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        metadata = {'source': os.path.basename(filepath)}
        # Add more metadata extraction here (e.g., language detection, library detection)
        return content, metadata
    except Exception as e:
        logger.error(f"Error processing file {filepath}: {e}")
        return None, None

def load_knowledge_base():
    """
    Processes files in the knowledge base directory, chunks them,
    generates embeddings, and loads them into the ChromaDB collection.
    """
    # Construct absolute paths based on script location or assume run from root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # Assumes scripts/ is one level down
    kb_dir_abs = os.path.join(project_root, KNOWLEDGE_BASE_DIR)
    vector_db_path_abs = os.path.join(project_root, VECTOR_DB_PATH)


    if not os.path.exists(kb_dir_abs):
        logger.error(f"Knowledge base directory not found: {kb_dir_abs}")
        return

    logger.info("Initializing ChromaDB client...")
    try:
        client = chromadb.PersistentClient(path=vector_db_path_abs)
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        logger.info(f"Getting or creating collection '{COLLECTION_NAME}'...")
        # Delete collection if it exists to start fresh? Or update?
        try:
            logger.info(f"Attempting to delete existing collection '{COLLECTION_NAME}' for fresh load...")
            client.delete_collection(name=COLLECTION_NAME)
            logger.info(f"Collection '{COLLECTION_NAME}' deleted.")
        except Exception as delete_exc:
             # Catch potential exception if collection doesn't exist
             logger.warning(f"Could not delete collection (might not exist): {delete_exc}")


        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
         logger.error(f"Failed to initialize ChromaDB: {e}")
         return

    all_chunks = []
    all_metadatas = []
    all_ids = []
    doc_count = 0

    logger.info(f"Processing documents in {kb_dir_abs}...")
    for filename in os.listdir(kb_dir_abs):
        filepath = os.path.join(kb_dir_abs, filename)
        if os.path.isfile(filepath):
            logger.info(f"Processing {filename}...")
            content, metadata = process_file(filepath)
            if content and metadata:
                doc_count += 1
                chunks = chunk_text(content, CONTEXT_CHUNK_SIZE, CONTEXT_CHUNK_OVERLAP)
                logger.info(f"  Generated {len(chunks)} chunks.")
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{metadata['source']}_{i}"
                    all_chunks.append(chunk)
                    all_metadatas.append(metadata) # Add chunk-specific metadata if needed
                    all_ids.append(chunk_id)

    if not all_chunks:
        logger.warning("No chunks generated. Is the knowledge base directory empty or files unreadable?")
        # Create a dummy entry if needed for testing RAG module initialization
        # logger.info("Adding a dummy entry to initialize the collection.")
        # collection.add(ids=["dummy_id"], documents=["dummy document"], metadatas=[{"source":"dummy"}])
        return # Exit if no real data

    logger.info(f"Adding {len(all_chunks)} chunks from {doc_count} documents to collection '{COLLECTION_NAME}'...")
    try:
        # Add in batches if dataset is very large
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
             batch_chunks = all_chunks[i:i+batch_size]
             batch_metadatas = all_metadatas[i:i+batch_size]
             batch_ids = all_ids[i:i+batch_size]
             logger.info(f"  Adding batch {i//batch_size + 1} ({len(batch_ids)} items)...")
             collection.add(
                 documents=batch_chunks,
                 metadatas=batch_metadatas,
                 ids=batch_ids
             )
        logger.info("Successfully added chunks to the collection.")
        logger.info(f"Total items in collection: {collection.count()}")
    except Exception as e:
        logger.error(f"Error adding chunks to ChromaDB: {e}", exc_info=True)

if __name__ == "__main__":
    # Add a dummy file to knowledge_base if it's empty, so ChromaDB initializes correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    kb_dir_abs = os.path.join(project_root, KNOWLEDGE_BASE_DIR)
    if not os.listdir(kb_dir_abs):
        dummy_file_path = os.path.join(kb_dir_abs, "placeholder.txt")
        with open(dummy_file_path, "w") as f:
            f.write("This is a placeholder document to ensure the knowledge base is not empty.")
        logger.info(f"Created placeholder file: {dummy_file_path}")

    load_knowledge_base()
EOF
echo "scripts/load_kb.py created."

# --- Create backend/requirements.txt ---
echo "Creating backend/requirements.txt..."
cat << EOF > "$PROJECT_DIR/backend/requirements.txt"
# Backend Framework & Server
flask
flask-socketio
uvicorn
python-dotenv

# Real-time Communication & Async HTTP
python-socketio >= 5.0 # Check compatibility with Flask-SocketIO
httpx

# RAG - Vector DB & Embeddings
chromadb >= 0.4.0,<0.5.0 # Pin version range for stability
sentence-transformers # For embedding model

# Optional performance boost for asyncio
# uvloop # Uncomment if needed and installable on your system

# Optional for more advanced code analysis (add if used)
# astroid
# radon

# Added for compatibility with newer Werkzeug in Flask dev server
Werkzeug >= 2.0
EOF
echo "backend/requirements.txt created."

# --- Create placeholder knowledge base file ---
echo "Creating placeholder knowledge base file..."
touch "$PROJECT_DIR/backend/knowledge_base/example_doc.txt"
echo "Placeholder file created."

echo "--- Project setup complete in directory '$PROJECT_DIR' ---"
echo "Next steps:"
echo "1. cd $PROJECT_DIR"
echo "2. (Optional) Create and activate a Python virtual environment: python -m venv venv && source venv/bin/activate"
echo "3. Install dependencies: pip install -r backend/requirements.txt"
echo "4. Add actual documents to backend/knowledge_base/"
echo "5. Run the knowledge base loading script: python scripts/load_kb.py"
echo "6. Set up your LLM API endpoint (e.g., using TGI/vLLM) and configure LLM_API_URL (e.g., in backend/.env)"
echo "7. Run the backend server: uvicorn backend.app:app --host 0.0.0.0 --port 5001 --reload"
echo "8. Develop the frontend application to connect to ws://localhost:5001"


