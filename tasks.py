from celery import Celery
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from manga_cleaner import MangaCleaner

celery = Celery(
    "tasks",
    broker="sqla+sqlite:///celery.sqlite",
    # backend="memory://"
    backend="db+sqlite:///results.sqlite"
)

OUTPUT_FOLDER = "output"

cleaner = None


@celery.task(bind=True)
def process_images_task(self, job_id, image_paths):
    """
    Celery task to process images asynchronously.
    """

    global cleaner
    if cleaner is None:
        print("🚀 Loading MangaCleaner models into memory...")
        cleaner = MangaCleaner("models/comic-speech-bubble-detector.pt", "models/manga-sfx-detector.pt")

    

    output_folder = os.path.join(OUTPUT_FOLDER, job_id)
    os.makedirs(output_folder, exist_ok=True)

    for image_path in image_paths:
        try:
            output_filename = os.path.splitext(os.path.basename(image_path))[0] + ".png"
            output_path = os.path.join(output_folder, output_filename)
            cleaner.clean_page(image_path, output_path, debug=False)
        except Exception as e:
            return {"error": str(e)}

    return {"status": "completed", "job_id": job_id}
