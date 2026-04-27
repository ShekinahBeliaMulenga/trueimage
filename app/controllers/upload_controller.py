from flask import Blueprint, request, current_app, send_from_directory, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import time
import cv2
import random
from datetime import datetime

upload_bp = Blueprint("upload", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

class TempFileManager:
    """Ensures temporary files are deleted even if the code crashes midway."""
    def __init__(self):
        self.files = []

    def track(self, filepath):
        if filepath and filepath not in self.files:
            self.files.append(filepath)

    def cleanup(self):
        for path in self.files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as e:
                current_app.logger.warning(f"Failed to cleanup temp file {path}: {e}")

@upload_bp.route("/temp_uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)

@upload_bp.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    temp_manager = TempFileManager()
    
    try:
        # ---------------------------------------------------------
        # 1. Initial File Validation & Size Limits
        # ---------------------------------------------------------
        if "image" not in request.files:
            return jsonify({"status": "error", "message": "No file stream detected in payload."}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"status": "error", "message": "Upload rejected: Empty filename detected."}), 400

        if not allowed_file(file.filename):
            return jsonify({"status": "error", "message": "Unsupported format. Please upload JPG or PNG."}), 400

        # Check file size in memory before writing to disk
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset cursor for saving later
        if file_size > MAX_FILE_SIZE_BYTES:
            return jsonify({"status": "error", "message": "File exceeds the 10MB size limit."}), 413

        # ---------------------------------------------------------
        # 2. Secure, Race-Condition-Proof File Saving
        # ---------------------------------------------------------
        # Use UUID + Timestamp to ensure two identical filenames don't overwrite each other
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}.{ext}"
        
        upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(upload_path)
        
        # Track the raw upload so we guarantee it gets deleted when the request finishes
        temp_manager.track(upload_path)

        # ---------------------------------------------------------
        # 3. Forensic Image Integrity Check & Memory-Bomb Protection
        # ---------------------------------------------------------
        test_img = cv2.imread(upload_path)
        if test_img is None:
            return jsonify({"status": "error", "message": "Image corruption detected or invalid format."}), 400

        # Prevent hackers from uploading tiny 1x1 images or massive 20,000px images to crash the server
        h, w = test_img.shape[:2]
        if h < 100 or w < 100 or h > 6000 or w > 6000:
            return jsonify({"status": "error", "message": "Image dimensions are out of acceptable bounds."}), 400
            
        del test_img  # Free RAM immediately

        # ---------------------------------------------------------
        # 4. Explicit-content screening (Fail-Fast)
        # ---------------------------------------------------------
        moderation_result = current_app.explicit_detector.predict(upload_path)
        if moderation_result.verdict in ["EXPLICIT", "SUGGESTIVE"]:
            return jsonify({"status": "error", "message": moderation_result.message}), 400

        # ---------------------------------------------------------
        # 5. Face Detection & Annotation
        # ---------------------------------------------------------
        output_filename = f"processed_{unique_filename}"
        output_path = os.path.join(current_app.config["UPLOAD_FOLDER"], output_filename)

        detection = current_app.face_detector.process_image(upload_path, output_path)

        if not detection.success:
            return jsonify({"status": "error", "message": detection.error_message}), 400

        # ---------------------------------------------------------
        # 6. Model inference (TODO: Replace with actual AI Model)
        # ---------------------------------------------------------
        raw_probability = random.random()

        # ---------------------------------------------------------
        # 7. Finalize Metrics & Respond
        # ---------------------------------------------------------
        processing_duration = round(time.time() - start_time, 2)

        redirect_url = url_for("result.show_result", 
                               filename=output_filename, 
                               probability=raw_probability, 
                               faces=detection.face_count,
                               scan_time=processing_duration)
        
        return jsonify({"status": "success", "redirect_url": redirect_url}), 200

    except (IOError, OSError, cv2.error) as e:
        # Expected environmental errors (Disk full, bad permissions, OpenCV freakout)
        current_app.logger.error(f"Processing error on {upload_path}: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Image processing failed. Please try a different image."}), 500

    except Exception as e:
        # The ultimate safety net: Logs the real error for you, hides it from the user.
        current_app.logger.critical(f"Unexpected critical failure: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "An unexpected system error occurred."}), 500

    finally:
        # GUARANTEED CLEANUP: 
        # This deletes the raw uploaded file, freeing up disk space instantly.
        temp_manager.cleanup()
        
        # Note on `output_path`: We deliberately DO NOT track/delete the `processed_xxx.jpg` 
        # here because the frontend UI still needs to load it to show the user the result page!