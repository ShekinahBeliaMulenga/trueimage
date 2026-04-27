import os
import time
import logging
from flask_apscheduler import APScheduler

logger = logging.getLogger("zero_retention_worker")
logger.setLevel(logging.INFO)

scheduler = APScheduler()

def scrub_expired_images(app):
    with app.app_context():
        upload_dir = app.config.get("UPLOAD_FOLDER")
        if not upload_dir or not os.path.exists(upload_dir):
            return

        current_time = time.time()
        
        # 1. THE EXPIRATION AGE - 45 SECONDS
        max_age_seconds = 45  

        deleted_count = 0
        
        for filename in os.listdir(upload_dir):
            if filename.startswith("processed_"):
                filepath = os.path.join(upload_dir, filename)
                
                try:
                    file_age = current_time - os.path.getmtime(filepath)
                    
                    if file_age > max_age_seconds:
                        os.remove(filepath)
                        deleted_count += 1
                except OSError as e:
                    pass # Silently ignore locked files, try again in 15 seconds
        
        if deleted_count > 0:
            logger.info(f"Aggressive Sweep: Purged {deleted_count} forensic artifacts older than {max_age_seconds}s.")

def init_enterprise_scheduler(app):
    app.config["SCHEDULER_API_ENABLED"] = False 
    scheduler.init_app(app)
    
    # 2. THE SWEEP INTERVAL
    scheduler.add_job(
        id='zero_retention_sweep_job',
        func=scrub_expired_images,
        args=[app],
        trigger='interval',
        seconds=15, 
        replace_existing=True
    )
    
    scheduler.start()
    logger.info("Hyper-Aggressive Zero Retention Online: Sweeping every 15 seconds.")