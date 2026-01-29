from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import easyocr
import torch
import io
from PIL import Image
import numpy as np

app = FastAPI()

# Check GPU availability
gpu_available = torch.cuda.is_available()
if gpu_available:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Detected: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    print("EasyOCR will use GPU acceleration")
else:
    print("No GPU detected. EasyOCR will use CPU.")

# Initialize EasyOCR reader (GPU enabled by default if available)
reader = easyocr.Reader(['en'], gpu=gpu_available)


@app.get("/status")
async def get_status():
    """Check GPU status and EasyOCR configuration."""
    status = {
        "gpu_available": gpu_available,
        "gpu_name": torch.cuda.get_device_name(0) if gpu_available else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if gpu_available else None,
        "cuda_version": torch.version.cuda if gpu_available else None,
        "pytorch_version": torch.__version__,
        "device": "GPU" if gpu_available else "CPU"
    }
    return JSONResponse(content=status)


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
