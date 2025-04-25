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
