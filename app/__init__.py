from flask import Flask, render_template

def create_app():

    app = Flask(__name__)

    app.config['UPLOAD_FOLDER'] = 'temp_uploads'
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

    # register blueprints
    from app.controllers.upload_controller import upload_bp
    app.register_blueprint(upload_bp)

    # home route (UI)
    @app.route("/")
    def home():
        return render_template("index.html")

    return app