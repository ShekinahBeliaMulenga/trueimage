from flask import Blueprint, request, current_app, send_from_directory, url_for, jsonify
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
    try:
        # ---------------------------------------------------------
        # Initial File Validation (Returning JSON 400 Bad Request)
        # ---------------------------------------------------------
        if "image" not in request.files:
            return jsonify({"status": "error", "message": "System Error: No file stream detected in payload."}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"status": "error", "message": "Upload rejected: Empty filename detected."}), 400

        if not allowed_file(file.filename):
            return jsonify({"status": "error", "message": "Unsupported file format. Please upload a standard JPG or PNG image."}), 400

        # ---------------------------------------------------------
        # File Saving
        # ---------------------------------------------------------
        filename = secure_filename(file.filename)
        upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(upload_path)

        # ---------------------------------------------------------
        # Step 1: Explicit-content screening (Fail-Fast)
        # ---------------------------------------------------------
        moderation_result = current_app.explicit_detector.predict(upload_path)
        
        if moderation_result.verdict in ["EXPLICIT", "SUGGESTIVE"]:
            os.remove(upload_path)
            return jsonify({"status": "error", "message": moderation_result.message}), 400

        # Output paths
        output_filename = f"processed_{filename}"
        output_path = os.path.join(current_app.config["UPLOAD_FOLDER"], output_filename)

        # ---------------------------------------------------------
        # Step 2: Face detection & Annotation via Service
        # ---------------------------------------------------------
        detection = current_app.face_detector.process_image(upload_path, output_path)

        if not detection.success:
            return jsonify({"status": "error", "message": detection.error_message}), 400

        # ---------------------------------------------------------
        # Step 3: Model inference (Currently Mocked)
        # ---------------------------------------------------------
        raw_probability = random.random()

        # ---------------------------------------------------------
        # Step 4: Return Success JSON with the Redirect URL
        # ---------------------------------------------------------
        redirect_url = url_for("result.show_result", 
                               filename=output_filename, 
                               probability=raw_probability, 
                               faces=detection.face_count)
        
        return jsonify({"status": "success", "redirect_url": redirect_url}), 200

    except Exception as e:
        # Catch unexpected server errors
        return jsonify({"status": "error", "message": f"System Exception: {str(e)}"}), 500

    finally:
        # Cleanup original un-annotated upload
        if 'upload_path' in locals() and os.path.exists(upload_path):
            try:
                os.remove(upload_path)
            except OSError:
                pass