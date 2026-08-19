from flask import Blueprint, request, current_app, send_from_directory, url_for, jsonify
import os
import uuid
import time
import cv2
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
    upload_path = None
    
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
        # 4. Explicit-content screening (Fail-Fast, with bounded retry)
        # ---------------------------------------------------------
        # SAFE: falls straight through, identical cost to before.
        # EXPLICIT: rejected immediately, before face detection even runs.
        #   Not eligible for crop-and-retry - a high-confidence EXPLICIT
        #   verdict isn't the false-positive case this recovers, and letting
        #   it retry would defeat the point of the check (Chapter 5,
        #   Section 5.2.3, item c).
        # SUGGESTIVE: the only branch that does extra work. Face detection
        #   runs once here - it would have run anyway had the image been
        #   SAFE - and one extra ExplicitDetector.predict() call checks the
        #   crop. If the crop clears, detection runs a second time on the
        #   crop alone, so every downstream artifact (annotated image,
        #   inference input) is built from the crop, never the original.
        moderation_result = current_app.explicit_detector.predict(upload_path)

        # `detection` is populated either here (SUGGESTIVE-but-cleared) or
        # in step 5 below (SAFE). Step 5 only runs its own detection if this
        # is still None - that's what prevents a duplicate detection call.
        detection = None
        inference_input_path = upload_path  # overridden below if a crop is used

        if moderation_result.verdict == "EXPLICIT":
            return jsonify({"status": "error", "message": moderation_result.message}), 400

        elif moderation_result.verdict == "SUGGESTIVE":
            probe_output_path = os.path.join(current_app.config["UPLOAD_FOLDER"], f"probe_{unique_filename}")
            probe_crop_path = os.path.join(current_app.config["UPLOAD_FOLDER"], f"probecrop_{unique_filename}")

            # process_image() now always displays via the crop internally
            # (see face_detector.py), so this single pass on the original
            # upload already produces a cropped, annotated result - a
            # second detection pass on the crop is no longer needed here,
            # unlike earlier versions of this branch.
            probe_detection = current_app.face_detector.process_image(
                input_path=upload_path,
                output_path=probe_output_path,
                crop_output_path=probe_crop_path,
            )
            # Tracked unconditionally so a rejection below cleans them up.
            # On success these get renamed to their permanent filenames
            # further down; cleanup() silently skips paths that no longer
            # exist, so leaving these two entries tracked afterward is
            # harmless.
            temp_manager.track(probe_output_path)
            temp_manager.track(probe_crop_path)

            if not probe_detection.success or not probe_detection.crop_path:
                # No usable face to crop to - fall back to the original
                # rejection.
                return jsonify({"status": "error", "message": moderation_result.message}), 400

            recheck = current_app.explicit_detector.predict(probe_detection.crop_path)

            if recheck.verdict != "SAFE":
                # Confirmed, not a false positive - reject using the
                # recheck's own message, since it reflects the crop itself.
                return jsonify({"status": "error", "message": recheck.message}), 400

            # Crop is clear. Promote the probe's own outputs to their
            # permanent filenames rather than detecting a second time -
            # process_image() already built them entirely from the crop.
            output_filename = f"processed_{unique_filename}"
            output_path = os.path.join(current_app.config["UPLOAD_FOLDER"], output_filename)
            crop_filename = f"crop_{unique_filename}"
            crop_path = os.path.join(current_app.config["UPLOAD_FOLDER"], crop_filename)

            os.rename(probe_output_path, output_path)
            os.rename(probe_crop_path, crop_path)

            detection = probe_detection
            detection.crop_path = crop_path
            inference_input_path = crop_path
            temp_manager.track(crop_path)  # temp - only needed for inference below

        # ---------------------------------------------------------
        # 5. Face Detection, Annotation & Clean Crop
        #    (SAFE verdict only - the SUGGESTIVE-cleared branch above
        #    already produced `detection`, so this is skipped for it)
        # ---------------------------------------------------------
        if detection is None:
            output_filename = f"processed_{unique_filename}"
            output_path = os.path.join(current_app.config["UPLOAD_FOLDER"], output_filename)

            crop_filename = f"crop_{unique_filename}"
            crop_path = os.path.join(current_app.config["UPLOAD_FOLDER"], crop_filename)

            detection = current_app.face_detector.process_image(
                input_path=upload_path,
                output_path=output_path,
                crop_output_path=crop_path
            )

            if not detection.success:
                return jsonify({"status": "error", "message": detection.error_message}), 400

            if detection.crop_path:
                temp_manager.track(detection.crop_path)

            # Inference now runs on the padded face crop for every verdict,
            # not just the SUGGESTIVE-cleared path - see face_detector.py's
            # inference_crop_padding_ratio for the padding used. Falls back
            # to the full upload only in the unlikely case no crop was saved.
            inference_input_path = detection.crop_path or upload_path

        # ---------------------------------------------------------
        # 6. AI Model Inference
        # ---------------------------------------------------------
        if current_app.ai_inference_engine is None:
            return jsonify({
                "status": "error",
                "message": "AI model is not available."
            }), 503
        
        # Reads from inference_input_path rather than upload_path directly -
        # every verdict now runs inference on the padded face crop rather
        # than the full upload, matching the framing set out in
        # face_detector.py's inference_crop_padding_ratio.
        inference_result = current_app.ai_inference_engine.predict_probability(inference_input_path)

        if not inference_result.success:
            return jsonify({
                "status": "error",
                "message": inference_result.message
            }), 500

        raw_probability = inference_result.ai_probability

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
        # here because the frontend UI still needs to load it to show the user the result page.