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
    # Use  or run directly with Uvicorn for stability.
    # Added allow_unsafe_werkzeug=True for newer Werkzeug versions with Flask dev server
