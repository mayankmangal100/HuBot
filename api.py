"""Flask API for RAG chatbot."""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
from src.utils import Config
from src.rag_system import RAGSystem
from src.chitchat_handler import ChitChatHandler

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize RAG system
config = Config.load_config("config/config.json")
rag_system = RAGSystem(config=config)

# @app.route('/')
# def index():
#     """Serve the chat interface."""
#     return render_template('index.html')

# Try to load existing data
if rag_system.load_existing_data():
    print("Using existing vector store")
else:
    print("No existing data found")

@socketio.on('ask_question')
def handle_question(data):
    """Handle chat requests through WebSocket."""
    try:
        if not data or 'question' not in data:
            emit('error', {'error': 'No question provided'})
            return
        
        question = data['question']

        chitchat = ChitChatHandler()
        if chitchat.is_chitchat(question):
            response = chitchat.handle(question)
            emit('token', {'token': response})
        else:
            # Stream the response
            for token in rag_system.answer_question(question, stream=True):
                emit('token', {'token': token})
        
        # Get the sources after completion
        sources = [
            {
                'source': doc['metadata']['source'],
                'chunk_id': doc['metadata']['chunk_id'],
                'score': float(score)
            }
            for doc, score in rag_system.last_context_docs
        ]
        
        # Send sources
        emit('sources', {'sources': sources})
        
        # Update conversation history
        last_response = rag_system.llm.get_last_streamed_response()
        # rag_system.query_rewriter.add_exchange(
        #     query=question,
        #     rewritten_query=rag_system.last_retrieval_query,
        #     answer=last_response,
        #     query_embedding=rag_system.last_query_embedding,
        #     context_docs=rag_system.last_context_docs
        # )
        
        # Signal completion
        emit('complete')
        
    except Exception as e:
        emit('error', {'error': str(e)})

@app.route('/process-document', methods=['POST'])
def process_document():
    """Process/reprocess the document."""
    try:
        document_path = "Documents/EveriseHandbook.pdf"
        if not os.path.exists(document_path):
            return jsonify({'error': 'Document not found'}), 404
        
        rag_system.ingest(document_path)
        return jsonify({'message': 'Document processed successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/Documents/<path:filename>', methods=['GET'])
def download_file(filename):
    documents_dir = os.path.join(os.getcwd(), 'Documents')
    return send_from_directory(directory=documents_dir, path=filename, as_attachment=True)

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
        socketio.run(app, 
                    host='0.0.0.0',
                    port=port,
                    debug=False,
                    allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"Error starting server: {e}")
