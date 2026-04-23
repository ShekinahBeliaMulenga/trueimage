from flask import Flask, render_template
from app.controllers.upload_controller import upload_bp
from app.controllers.result_controller import result_bp
from app.services.face_detector import FaceDetector
from app.services.explicit_detector import ExplicitDetector
import os


def create_app():

    app = Flask(__name__)

    app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "temp_uploads")
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    
    app.face_detector = FaceDetector()
    app.explicit_detector = ExplicitDetector(threshold=0.50)

    # Register routes
    app.register_blueprint(upload_bp)
    app.register_blueprint(result_bp)

    @app.route("/")
    def home():
        return render_template("index.html")

    return app