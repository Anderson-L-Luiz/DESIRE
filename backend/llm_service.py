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
