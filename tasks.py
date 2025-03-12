from celery import Celery
import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from manga_cleaner import MangaCleaner
from manga_processor import MangaPageProcessor

celery = Celery(
    "tasks",
    broker="sqla+sqlite:///celery.sqlite",
    # backend="memory://"
    backend="db+sqlite:///results.sqlite"
)

OUTPUT_FOLDER = "output"

processor = None


def get_server_url():
    """
    Get the public URL for this RunPod instance dynamically.
    """

    pod_id = os.getenv("RUNPOD_POD_ID")
    internal_port = os.getenv("INTERNAL_PORT")
    if pod_id:
        return f"https://{pod_id}-{internal_port}.proxy.runpod.net"
    return "http://localhost:8000"


@celery.task(bind=True)
def process_images_task(self, job_id, image_paths, webhook_url=None):
    """
    Celery task to process images asynchronously.
    """

    global processor
    if processor is None:
        print("🚀 Loading MangaProcessor models into memory...")
        processor = MangaPageProcessor("models/comic-speech-bubble-detector.pt", "models/manga-sfx-detector.pt")

    output_folder = os.path.join(OUTPUT_FOLDER, job_id)
    os.makedirs(output_folder, exist_ok=True)

    server_url = get_server_url() 

    processed_files = []

    for image_path in image_paths:
        try:
            image_basename = os.path.splitext(os.path.basename(image_path))[0]
            # output_path = os.path.join(output_folder, output_filename)
            processor.process_page(image_path, output_folder)

            processed_files.append({
                "image_url": f"{server_url}/outputs/{job_id}/{image_basename}.png",
                "kra_url": f"{server_url}/outputs/{job_id}/{image_basename}.kra"
            })

        except Exception as e:
            return {"error": str(e)}
        
    print(f"Processed files: {processed_files}")
        
    if webhook_url:
        print(f"Sending webhook to {webhook_url}")
        try:
            requests.post(webhook_url, json={"job_id": job_id, "status": "completed", "files": processed_files})
            print(f"Webhook sent to {webhook_url}")
        except requests.exceptions.RequestException as e:
            print(f"Webhook failed: {e}")

    return {"status": "completed", "job_id": job_id}
