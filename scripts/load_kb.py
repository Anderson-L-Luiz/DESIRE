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
