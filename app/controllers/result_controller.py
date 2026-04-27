import os
from flask import Blueprint, request, render_template, current_app
from app.services.result_interpreter import ResultInterpreter

result_bp = Blueprint("result", __name__)

@result_bp.route("/analysis_report")
def show_result():
    # 1. Retrieve the data passed from the upload controller
    filename = request.args.get("filename")
    probability = request.args.get("probability", type=float)
    face_count = request.args.get("faces", type=int, default=0)
    scan_time = request.args.get("scan_time", type=float, default=0.0)

    # Fallback if accessed directly without data
    if not filename or probability is None:
        return render_template(
            "index.html", 
            error="No analysis data found. Please run a new scan."
        )

    # ---------------------------------------------------------
    # 2. THE BACKEND FAILSAFE (Zero Retention Enforcement)
    # ---------------------------------------------------------
    file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    
    if not os.path.exists(file_path):
        # The background worker deleted it! Intercept the request.
        return render_template(
            "index.html", 
            error="Zero Retention Policy enforced. The 45-second viewing window expired and your data has been securely purged. Please initialize a new scan."
        )

    # 3. Reconstruct the image URL
    image_url = f"/temp_uploads/{filename}"

    # 4. Call the logic service to interpret the raw math
    interpretation = ResultInterpreter.analyze(probability)

    # 5. Render the dashboard with all metadata
    return render_template(
        "result.html",
        result_label=interpretation.label,
        confidence=interpretation.confidence_score,
        ai_probability=interpretation.ai_probability,
        user_message=interpretation.user_message,
        image_path=image_url,
        face_count=face_count,
        scan_time=scan_time
    )