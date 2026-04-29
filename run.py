from app import create_app
from scheduler import init_enterprise_scheduler

app = create_app()

# We pass 'app' so the scheduler knows your specific configuration
init_enterprise_scheduler(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True, use_reloader=False)