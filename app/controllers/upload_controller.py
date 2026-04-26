from flask import Blueprint, request, current_app, send_from_directory, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import random
import cv2 
import time  # NEW: Added to track processing duration

upload_bp = Blueprint("upload", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@upload_bp.route("/temp_uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)

@upload_bp.route("/predict", methods=["POST"])
def predict():
    # START TIMER: Capture the exact moment the server begins work
    start_time = time.time()
    
    try:
        # ---------------------------------------------------------
        # Initial File Validation
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
        # Forensic Image Integrity Check
        # ---------------------------------------------------------
        test_img = cv2.imread(upload_path)
        if test_img is None:
            os.remove(upload_path) 
            return jsonify({
                "status": "error", 
                "message": "Image corruption detected. The file format may be invalid or incorrectly named."
            }), 400

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
        # FINALIZE TIMER: Calculate total elapsed time
        # ---------------------------------------------------------
        end_time = time.time()
        # Round to 2 decimal places (e.g., 1.42)
        processing_duration = round(end_time - start_time, 2)

        # ---------------------------------------------------------
        # Step 4: Return Success JSON with the Redirect URL
        # ---------------------------------------------------------
        # We now pass 'scan_time' into the arguments
        redirect_url = url_for("result.show_result", 
                               filename=output_filename, 
                               probability=raw_probability, 
                               faces=detection.face_count,
                               scan_time=processing_duration)
        
        return jsonify({"status": "success", "redirect_url": redirect_url}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": f"System Exception: {str(e)}"}), 500

    finally:
        if 'upload_path' in locals() and os.path.exists(upload_path):
            try:
                os.remove(upload_path)
            except OSError:
                pass