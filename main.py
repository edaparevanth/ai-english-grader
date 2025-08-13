from fastapi import FastAPI, File, UploadFile
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
import tempfile

# Point pytesseract to installed location
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess the uploaded image for better OCR results."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Resize for better accuracy
    scale_percent = 150
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)

    # Remove noise
    gray = cv2.medianBlur(gray, 3)

    # Apply threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
        cv2.imwrite(temp.name, processed_image)
        temp_path = temp.name

    extracted_text = pytesseract.image_to_string(Image.open(temp_path))
    return {"extracted_text": extracted_text.strip()}

@app.get("/")
def root():
    return {"message": "Upload an image to /extract-text (POST request) to get OCR text"}
