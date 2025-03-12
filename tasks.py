from celery import Celery
import os
import sys
import requests
from dotenv import load_dotenv
import json

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from manga_cleaner import MangaCleaner
from manga_processor import MangaPageProcessor
from translator import MangaTranslator

celery = Celery(
    "tasks",
    broker="sqla+sqlite:///celery.sqlite",
    # backend="memory://"
    backend="db+sqlite:///results.sqlite"
)

OUTPUT_FOLDER = "output"

processor = None
translator = None


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

    global processor, translator
    if processor is None:
        print("🚀 Loading MangaProcessor models into memory...")
        processor = MangaPageProcessor("models/comic-speech-bubble-detector.pt", "models/manga-sfx-detector.pt")

    if translator is None:
        print("📝 Loading MangaTranslator model into memory...")
        translator = MangaTranslator("models/comic-speech-bubble-detector.pt")

    output_folder = os.path.join(OUTPUT_FOLDER, job_id)
    os.makedirs(output_folder, exist_ok=True)

    server_url = get_server_url() 

    processed_files = []
    target_languages = ["en", "zh", "es"]

    for image_path in image_paths:
        try:
            image_basename = os.path.splitext(os.path.basename(image_path))[0]
            output_kra_path = os.path.join(output_folder, f"{image_basename}.kra")
            translation_json_path = os.path.join(output_folder, f"{image_basename}_translation.json")

            # Check if we already have translations
            if os.path.exists(translation_json_path):
                print(f"📖 Loading existing translation for {image_basename}...")
                with open(translation_json_path, 'r', encoding='utf-8') as f:
                    translation_data = json.load(f)
            else:
                # Detect & translate text
                print(f"🔍 Detecting and translating text for {image_basename}...")
                _, translation_data = translator.process_image_with_custom_output(
                    image_path, target_languages, translation_json_path
                )

            print(f"🖌 Processing {image_basename} into Krita format...")
            processor.process_with_translations(image_path, translation_data, output_kra_path)

            processed_files.append({
                "image_url": f"{server_url}/outputs/{job_id}/{image_basename}.png",
                "kra_url": f"{server_url}/outputs/{job_id}/{image_basename}.kra",
                "translation_json_url": f"{server_url}/outputs/{job_id}/{image_basename}_translation.json"
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
