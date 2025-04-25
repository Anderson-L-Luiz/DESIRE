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
