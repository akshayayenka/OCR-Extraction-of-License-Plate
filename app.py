from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
import cv2
import pytesseract
import os
from pathlib import Path

# Initialize FastAPI app
app = FastAPI()

# Configure static files and templates
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)  # Ensure the folder exists
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
templates = Jinja2Templates(directory="templates")

# Set Tesseract executable path (update based on your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to check allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# License plate detection and OCR function
def detect_license_plate(image_path: Path) -> str:
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply filtering
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plate_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:  # License plates generally have 4 corners
            plate_contour = approx
            break

    if plate_contour is None:
        return "License plate not detected"

    x, y, w, h = cv2.boundingRect(plate_contour)
    plate = gray[y:y + h, x:x + w]

    # OCR using Tesseract
    text = pytesseract.image_to_string(plate, config='--psm 8')
    return text.strip()

# Route for home page with file upload
@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>License Plate Recognition</title>
    </head>
    <body>
        <h1>Upload an Image</h1>
        <form action="/upload/" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Upload</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Route to handle file upload and OCR
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Validate file extension
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")

    # Save the uploaded file
    file_path = UPLOAD_FOLDER / file.filename
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Perform license plate detection and recognition
    detected_text = detect_license_plate(file_path)

    # Render the result page
    return templates.TemplateResponse("result.html", {
        "request": {}, 
        "image_url": f"/uploads/{file.filename}",
        "text": detected_text
    })
