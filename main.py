from fastapi import FastAPI, UploadFile, File
from whisper import load_model
import torch

app = FastAPI()

# Load the Whisper model (use "tiny" or "small" to reduce memory usage)
model = load_model("tiny")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save the uploaded file
    with open("temp_audio.mp3", "wb") as buffer:
        buffer.write(await file.read())

    # Transcribe the audio
    result = model.transcribe("temp_audio.mp3")
    return {"transcription": result["text"]}

@app.get("/")
def read_root():
    return {"message": "Healthcare Translation Web App"}
