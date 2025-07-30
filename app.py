from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import torch  # Fix: import torch at the top!
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model loader
def load_model():
    try:
        logger.info("Loading summarization model...")
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1,
            truncation=True  # Allow summarizer to truncate long input
        )
        logger.info("Model loaded successfully")
        return summarizer
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

# Initialize model at startup
summarizer = load_model()

@app.route('/summarize', methods=['POST'])
def summarize():
    """API endpoint for text summarization"""
    try:
        # Validate input
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text or len(text) < 50:
            return jsonify({"error": "Text must be at least 50 characters"}), 400
        
        # Model max input tokens ~1024; restrict input length accordingly
        # For BART, roughly 1 token ≈ 1 word, but to be sure, cut by characters (not perfect)
        if len(text) > 3000:
            text = text[:3000]
        
        # Generate summary
        summary = summarizer(
            text,
            max_length=min(150, len(text) // 4),
            min_length=30,
            do_sample=False,
            truncation=True
        )
        
        return jsonify({
            "summary": summary[0]['summary_text'],
            "original_length": len(text),
            "summary_length": len(summary[0]['summary_text'])
        })
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
