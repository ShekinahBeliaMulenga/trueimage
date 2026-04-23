from flask import Blueprint, request, render_template, current_app, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import os
import random

upload_bp = Blueprint("upload", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@upload_bp.route("/temp_uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)

@upload_bp.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", error="No file provided")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", error="Empty filename")

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Unsupported file format. Please upload a JPG or PNG image."
        )

    filename = secure_filename(file.filename)
    upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(upload_path)

        # ---------------------------------------------------------
        # Step 1: Explicit-content screening (Fail-Fast)
        # ---------------------------------------------------------
        moderation_result = current_app.explicit_detector.predict(upload_path)

        if moderation_result.verdict in ["EXPLICIT", "SUGGESTIVE"]:
            # Delete the file immediately to uphold privacy/safety
            os.remove(upload_path)
            return render_template("index.html", error=moderation_result.message)

        # Output paths
        output_filename = f"processed_{filename}"
        output_path = os.path.join(current_app.config["UPLOAD_FOLDER"], output_filename)

        # ---------------------------------------------------------
        # Step 2: Face detection & Annotation via Service
        # ---------------------------------------------------------
        detection = current_app.face_detector.process_image(upload_path, output_path)

        if not detection.success:
            # Rejects with standard UI format if no face or bad read
            return render_template("index.html", error=detection.error_message)

        # ---------------------------------------------------------
        # Step 3: Model inference (Currently Mocked)
        # ---------------------------------------------------------
        # In the future, this is where your EfficientNetV2-S model logic goes
        raw_probability = random.random()

        # ---------------------------------------------------------
        # Step 4: Redirect to Result Controller
        # ---------------------------------------------------------
        return redirect(url_for("result.show_result", 
                                filename=output_filename, 
                                probability=raw_probability, 
                                faces=detection.face_count))

    except Exception as e:
        return render_template("index.html", error=str(e))

    finally:
        # Cleanup original un-annotated upload
        if os.path.exists(upload_path):
            try:
                os.remove(upload_path)
            except OSError:
                pass