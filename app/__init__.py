from flask import Flask, render_template
from app.controllers.upload_controller import upload_bp
from app.controllers.result_controller import result_bp
from app.services.face_detector import FaceDetector
from app.services.explicit_detector import ExplicitDetector
from app.services.ai_inference_engine import AIInferenceEngine
import os


def create_app():
    

    app = Flask(__name__)
    app.secret_key = "super_secret_trueimage_key"     

    app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "temp_uploads")
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    
    app.face_detector = FaceDetector()
    app.explicit_detector = ExplicitDetector(threshold=0.50)

    model_path = os.path.join(app.root_path, "models", "trueimage_model.keras")

    try:
        app.ai_inference_engine = AIInferenceEngine(model_path=model_path)
        app.logger.info("AI inference engine loaded successfully.")
    except FileNotFoundError as error:
        app.ai_inference_engine = None
        app.logger.warning(f"AI inference engine not loaded: {error}")

    # Register routes
    app.register_blueprint(upload_bp)
    app.register_blueprint(result_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app