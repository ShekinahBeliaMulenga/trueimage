from flask import Blueprint, request, render_template
from app.services.result_interpreter import ResultInterpreter

result_bp = Blueprint("result", __name__)

@result_bp.route("/analysis_report")
def show_result():
    # 1. Retrieve the data passed from the upload controller
    filename = request.args.get("filename")
    probability = request.args.get("probability", type=float)
    face_count = request.args.get("faces", type=int, default=0)
    # NEW: Capture the processing duration sent from the upload step
    scan_time = request.args.get("scan_time", type=float, default=0.0)

    # Fallback if accessed directly without data
    if not filename or probability is None:
        return render_template(
            "index.html", 
            error="No analysis data found. Please run a new scan."
        )

    # 2. Reconstruct the image URL
    image_url = f"/temp_uploads/{filename}"

    # 3. Call the logic service to interpret the raw math
    interpretation = ResultInterpreter.analyze(probability)

    # 4. Render the dashboard with all metadata, including the new scan_time
    return render_template(
        "result.html",
        result_label=interpretation.label,
        confidence=interpretation.confidence_score,
        ai_probability=interpretation.ai_probability,
        user_message=interpretation.user_message,
        image_path=image_url,
        face_count=face_count,
        scan_time=scan_time  # Pass to HTML
    )