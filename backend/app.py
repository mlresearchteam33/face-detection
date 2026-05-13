"""
Enhanced Face Detection Flask App
Improved version with better error handling, logging, and production features
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
import traceback
import os
import logging
from datetime import datetime
import io
from werkzeug.utils import secure_filename
import time
import math

# =========================
# CONFIGURATION
# =========================

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
IMAGE_SIZE = (128, 128)
MODEL_PATH = "face_model.h5"
EMBEDDING_PATH = "face_model.pkl"

# =========================
# LOGGING SETUP
# =========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('face_detection.log')
    ]
)
logger = logging.getLogger(__name__)

# =========================
# FLASK APP INITIALIZATION
# =========================

app = Flask(
    __name__,
    template_folder="../"
)

# Security configuration
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['JSON_SORT_KEYS'] = False

# =========================
# CORS CONFIGURATION
# =========================

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True,
            "max_age": 3600
        }
    }
)

# =========================
# GLOBAL VARIABLES
# =========================

model = None
mean_emb = None
threshold = None
model_loaded = False
load_error = None

# =========================
# UTILITY FUNCTIONS
# =========================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_file(file):
    """Validate uploaded file"""
    if not file or file.filename == '':
        return False, "No file selected"
    
    if not allowed_file(file.filename):
        return False, f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large. Max size: {MAX_FILE_SIZE / (1024*1024):.1f}MB"
    
    if file_size == 0:
        return False, "File is empty"
    
    return True, "Valid"

def load_image_from_file(file):
    """Load and process image from uploaded file"""
    try:
        image = Image.open(file).convert("RGB")
        return True, image
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        return False, None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Resize image
        image_resized = image.resize(IMAGE_SIZE)
        
        # Convert to numpy array
        img_array = np.array(image_resized).astype("float32") / 255.0
        
        # Expand dimensions for batch
        img_batch = np.expand_dims(img_array, axis=0)
        
        return True, img_batch
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return False, None

def get_embedding(img_batch):
    """Get embedding from model"""
    try:
        embedding = model.predict(img_batch, verbose=0)[0]
        return True, np.array(embedding)
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        return False, None

def classify_face(embedding, distance):

    # =========================
    # HUMAN FACE
    # =========================
    if distance < threshold:

        # Convert distance into strong confidence
        ratio = distance / threshold

        # Better UI confidence scaling
        confidence = 100 - (ratio * 35)

        # boost clear faces
        if ratio < 0.80:
            confidence += 15

        confidence = min(99.9, max(60.0, confidence))

        return {
            "classification": "Human Face ✅",
            "is_face": True,
            "confidence": round(confidence, 2),
            "human_percent": round(confidence, 2),
            "nonhuman_percent": round(100 - confidence, 2)
        }

    # =========================
    # NON HUMAN
    # =========================
    else:

        ratio = distance / threshold

        nonhuman = 55 + ((ratio - 1) * 120)

        nonhuman = min(99.9, max(55.0, nonhuman))

        return {
            "classification": "Non Human Face ❌",
            "is_face": False,
            "confidence": round(nonhuman, 2),
            "human_percent": round(100 - nonhuman, 2),
            "nonhuman_percent": round(nonhuman, 2)
        }
def load_models():
    """Load ML models on startup"""
    global model, mean_emb, threshold, model_loaded, load_error
    
    try:
        logger.info("🔄 Loading models...")
        
        # Check if model files exist
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        if not os.path.exists(EMBEDDING_PATH):
            raise FileNotFoundError(f"Embedding file not found: {EMBEDDING_PATH}")
        
        # Load model
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("✅ Face model loaded successfully")
        
        # Load embedding data
        with open(EMBEDDING_PATH, "rb") as f:
            model_data = pickle.load(f)
        
        mean_emb = np.array(model_data["mean_emb"])
        threshold = float(model_data["threshold"])
        
        logger.info(f"✅ Embedding data loaded (threshold: {threshold:.4f})")
        logger.info("✅ All models loaded successfully")
        
        model_loaded = True
        load_error = None
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        logger.error(traceback.format_exc())
        model_loaded = False
        load_error = str(e)

# =========================
# ROUTES
# =========================

@app.route("/", methods=["GET"])
def home():
    """Serve home page"""
    try:
        return render_template("index.html")
    except Exception as e:
        logger.error(f"Error serving home page: {str(e)}")
        return jsonify({
            "error": "Error loading home page",
            "details": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": model_loaded,
        "timestamp": datetime.now().isoformat(),
        "service": "Face Detection API",
        "version": "2.0"
    }), 200

@app.route("/status", methods=["GET"])
def status():
    """Get detailed status"""
    status_data = {
        "service": "Face Detection API",
        "version": "2.0",
        "status": "running",
        "models_loaded": model_loaded,
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "GET /": "Home page",
            "GET /health": "Health check",
            "GET /status": "Detailed status",
            "POST /predict": "Predict face (with image upload)"
        }
    }
    
    if not model_loaded:
        status_data["error"] = load_error
        return jsonify(status_data), 503
    
    return jsonify(status_data), 200

@app.route("/predict", methods=["POST", "OPTIONS"])
@cross_origin()
def predict():
    """
    Main prediction endpoint
    
    Request:
    --------
    POST /predict
    Content-Type: multipart/form-data
    Body:
      - file: Image file (PNG, JPG, JPEG, WebP, GIF, BMP)
    
    Response:
    ---------
    {
        "success": true,
        "result": "Human Face ✅",
        "is_face": true,
        "distance": 0.1234,
        "threshold": 0.5,
        "confidence": 75.23,
        "processing_time_ms": 245.67,
        "image_info": {
            "original_size": [800, 600],
            "processed_size": [128, 128],
            "format": "RGB"
        }
    }
    """
    
    start_time = time.time()
    
    try:
        # ====== CHECK MODEL STATUS ======
        if not model_loaded:
            logger.error("❌ Models not loaded")
            return jsonify({
                "success": False,
                "result": "Models not loaded. Please try again later.",
                "error": load_error
            }), 503
        
        logger.info("📥 Request received for prediction")
        
        # ====== VALIDATE REQUEST ======
        if "file" not in request.files:
            logger.warning("⚠️ No file in request")
            return jsonify({
                "success": False,
                "result": "No file uploaded"
            }), 400
        
        file = request.files["file"]
        logger.info(f"📸 File: {file.filename}")
        
        # ====== VALIDATE FILE ======
        is_valid, message = validate_image_file(file)
        if not is_valid:
            logger.warning(f"⚠️ Invalid file: {message}")
            return jsonify({
                "success": False,
                "result": message
            }), 400
        
        # ====== LOAD IMAGE ======
        logger.info("🖼️ Loading image...")
        success, image = load_image_from_file(file)
        if not success:
            logger.error("❌ Failed to load image")
            return jsonify({
                "success": False,
                "result": "Failed to load image"
            }), 400
        
        original_size = image.size
        logger.info(f"✅ Image loaded: {original_size}")
        
        # ====== PREPROCESS IMAGE ======
        logger.info("🔄 Preprocessing image...")
        success, img_batch = preprocess_image(image)
        if not success:
            logger.error("❌ Failed to preprocess image")
            return jsonify({
                "success": False,
                "result": "Failed to preprocess image"
            }), 400
        
        logger.info("✅ Image preprocessed")
        
        # ====== GET EMBEDDING ======
        logger.info("🤖 Generating embedding...")
        success, embedding = get_embedding(img_batch)
        if not success:
            logger.error("❌ Failed to generate embedding")
            return jsonify({
                "success": False,
                "result": "Failed to generate embedding"
            }), 500
        
        logger.info("✅ Embedding generated")
        
        # ====== CALCULATE DISTANCE ======
        logger.info("📏 Calculating distance...")
        distance = float(np.linalg.norm(embedding - mean_emb))
        logger.info(f"📏 Distance: {distance:.4f} (threshold: {threshold:.4f})")
        
        # ====== CLASSIFY FACE ======
        logger.info("🔍 Classifying...")
        classification = classify_face(embedding, distance)
        logger.info(f"✅ Result: {classification['classification']}")
        
        # ====== CALCULATE PROCESSING TIME ======
        processing_time = (time.time() - start_time) * 1000
        
        # ====== RETURN RESPONSE ======
        response = {
            "success": True,
            "result": classification['classification'],
            "is_face": classification['is_face'],
            "distance": round(distance, 4),
            "threshold": round(threshold, 4),
            "confidence": classification['confidence'],
            "human_percent": classification['human_percent'],
            "nonhuman_percent": classification['nonhuman_percent'],
            "processing_time_ms": round(processing_time, 2),
            "image_info": {
                "original_size": list(original_size),
                "processed_size": list(IMAGE_SIZE),
                "format": "RGB"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Response sent (time: {processing_time:.2f}ms)")
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        
        processing_time = (time.time() - start_time) * 1000
        
        return jsonify({
            "success": False,
            "result": "Internal server error",
            "error": str(e),
            "processing_time_ms": round(processing_time, 2)
        }), 500

# =========================
# ERROR HANDLERS
# =========================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    logger.warning(f"⚠️ File too large: {error}")
    return jsonify({
        "success": False,
        "result": f"File too large. Max size: {MAX_FILE_SIZE / (1024*1024):.1f}MB"
    }), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404 error"""
    logger.warning(f"⚠️ Not found: {error}")
    return jsonify({
        "success": False,
        "result": "Endpoint not found"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 error"""
    logger.warning(f"⚠️ Method not allowed: {error}")
    return jsonify({
        "success": False,
        "result": "Method not allowed"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 error"""
    logger.error(f"❌ Internal server error: {error}")
    return jsonify({
        "success": False,
        "result": "Internal server error"
    }), 500

# =========================
# BEFORE REQUEST HOOKS
# =========================

@app.before_request
def log_request():
    """Log incoming requests"""
    if request.method != 'OPTIONS':
        logger.info(f"📡 {request.method} {request.path} from {request.remote_addr}")

@app.after_request
def after_request(response):
    """Add security headers"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# =========================
# STARTUP & SHUTDOWN
# =========================

def startup():
    """Initialize on startup"""
    logger.info("=" * 50)
    logger.info("🚀 Face Detection API Starting Up...")
    logger.info("=" * 50)
    load_models()
    logger.info("=" * 50)
    logger.info("✅ Initialization Complete")
    logger.info("=" * 50)

def shutdown(exception=None):
    """Cleanup on shutdown"""
    logger.info("=" * 50)
    logger.info("🛑 Shutting Down...")
    logger.info("=" * 50)

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    # Startup
    startup()
    
    # Register shutdown
    import atexit
    atexit.register(shutdown)
    
    # Get port from environment
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV", "production") == "development"
    
    logger.info(f"🌐 Starting server on 0.0.0.0:{port}")
    logger.info(f"🔧 Debug mode: {debug_mode}")
    
    # Run app
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug_mode,
        use_reloader=False  # Disable reloader in production
    )
