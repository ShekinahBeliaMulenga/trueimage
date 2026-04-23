from flask import Blueprint, request, render_template

result_bp = Blueprint("result", __name__)

@result_bp.route("/analysis_report")
def show_result():
    # 1. Retrieve the data passed from the upload controller
    filename = request.args.get("filename")
    probability = request.args.get("probability", type=float)
    face_count = request.args.get("faces", type=int, default=0)

    # Fallback if accessed directly without data
    if not filename or probability is None:
        return render_template("index.html", error="No analysis data found. Please run a new scan.")

    # 2. Reconstruct the image URL
    image_url = f"/temp_uploads/{filename}"

    # 3. Interpretation Logic (Translating raw math into Forensic UI labels)
    ai_probability = round(probability * 100, 1)
    confidence_score = ai_probability

    if ai_probability <= 40:
        label = "REAL"
        user_message = "Neural scan complete. The image appears to be a real photograph."
        # For 'REAL', confidence is the inverse of the AI probability
        confidence_score = round((1 - probability) * 100, 1)
    elif ai_probability < 75:
        label = "UNCERTAIN"
        user_message = "The analysis is inconclusive. Manual review is recommended."
    else:
        label = "AI-GENERATED"
        user_message = "Neural scan complete. Synthetic characteristics were detected in the image."

    # 4. Render the purple dashboard
    return render_template(
        "result.html",
        result_label=label,
        confidence=confidence_score,
        ai_probability=ai_probability,
        user_message=user_message,
        image_path=image_url,
        face_count=face_count
    )