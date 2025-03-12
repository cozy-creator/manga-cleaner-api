from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from celery.result import AsyncResult
from typing import List, Optional
import shutil
import os
import uuid
from tasks import process_images_task


app = FastAPI()

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Serve the output folder as static files
app.mount("/outputs", StaticFiles(directory="output"), name="outputs")

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...), webhook_url: Optional[str] = None):
    """
    Upload multiple images and queue them for processing.
    """
    job_id = str(uuid.uuid4())  # Unique job ID

    job_folder = os.path.join(UPLOAD_FOLDER, job_id)
    os.makedirs(job_folder, exist_ok=True)

    image_paths = []

    print(f"Webhook URL: {webhook_url}")

    for file in files:
        file_path = os.path.join(job_folder, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image_paths.append(file_path)

    # Queue the task in Celery
    process_images_task.delay(job_id, image_paths, webhook_url)

    # Return the URL where processed images will be available
    return {
        "job_id": job_id,
        "status": "processing",
    }


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Check job status.
    """
    task_result = AsyncResult(job_id)
    
    if task_result.state == "PENDING":
        return {"job_id": job_id, "status": "pending"}
    elif task_result.state == "SUCCESS":
        return {"job_id": job_id, "status": "completed"}
    elif task_result.state == "FAILURE":
        return {"job_id": job_id, "status": "failed", "error": str(task_result.result)}
    else:
        return {"job_id": job_id, "status": "processing"}


@app.get("/result/{job_id}")
async def get_results(job_id: str):
    """
    Download cleaned images when processing is complete.
    """
    output_folder = os.path.join(OUTPUT_FOLDER, job_id)

    if not os.path.exists(output_folder):
        return JSONResponse({"error": "Results not found."}, status_code=404)

    images = os.listdir(output_folder)
    return JSONResponse({"job_id": job_id, "files": images})
