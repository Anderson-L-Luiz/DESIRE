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
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8345/generate") # Example TGI endpoint
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 30)) # Seconds

# --- Prompt Template ---
# Clearly structure the input for the LLM
PROMPT_TEMPLATE = """
You are a helpful AI code assistant. Analyze the user's code snippet and their stated goal (scope).
Use the provided context documents, if relevant, to generate an accurate and helpful code suggestion, completion, or explanation.

**User's Current Code:**
```python
{code}
```

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
