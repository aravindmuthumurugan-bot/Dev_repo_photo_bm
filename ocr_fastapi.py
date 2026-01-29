from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import easyocr
import io
from PIL import Image
import numpy as np

app = FastAPI()

# Initialize EasyOCR reader (GPU enabled by default)
reader = easyocr.Reader(['en'])


@app.post("/ocr")
async def perform_ocr(image: UploadFile = File(...)):
    """
    Upload an image and get OCR text results.
    """
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        img_array = np.array(img)

        result = reader.readtext(img_array)

        extracted_text = [{"text": item[1], "confidence": float(item[2])} for item in result]
        full_text = " ".join([item[1] for item in result])

        return JSONResponse(content={
            "success": True,
            "full_text": full_text,
            "details": extracted_text
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
