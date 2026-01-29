from fastapi import FastAPI, File, UploadFile, Form, Body, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from enum import Enum
import os
import shutil
import tempfile
from datetime import datetime
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time

# Import validation system
from hybrid_dev_repo import validate_photo_complete_hybrid, GPU_AVAILABLE, TF_GPU_AVAILABLE, ONNX_GPU_AVAILABLE

app = FastAPI(
    title="Photo Validation API - Hybrid GPU",
    description="GPU-accelerated photo validation (InsightFace + DeepFace + NudeNet)",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)

# ==================== MODELS ====================

class SingleImageResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None
    response_time_seconds: Optional[float] = None
    library_usage: Optional[dict] = None
    gpu_info: Optional[dict] = None

class MultiImageResponse(BaseModel):
    success: bool
    message: str
    batch_id: str
    total_images: int
    results: List[dict]
    summary: dict
    response_time_seconds: Optional[float] = None
    library_usage_summary: Optional[dict] = None
    gpu_info: Optional[dict] = None

# ==================== HELPER FUNCTIONS ====================

def save_upload_file_tmp(upload_file: UploadFile) -> str:
    temp_dir = os.path.join(tempfile.gettempdir(), "photo_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    file_extension = os.path.splitext(upload_file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    temp_path = os.path.join(temp_dir, unique_filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return temp_path

def cleanup_temp_files(*file_paths):
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                cropped_path = file_path.replace(".", "_cropped_final.")
                if os.path.exists(cropped_path):
                    os.remove(cropped_path)
        except Exception as e:
            print(f"Cleanup warning: {e}")

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def format_validation_result(result: dict, image_filename: str) -> dict:
    result = convert_numpy_types(result)
    
    final_decision = result["final_decision"]
    if final_decision == "APPROVE":
        final_status = "ACCEPTED"
    elif final_decision == "REJECT":
        final_status = "REJECTED"
    elif final_decision == "SUSPEND":
        final_status = "SUSPENDED"
    elif final_decision == "MANUAL_REVIEW":
        final_status = "MANUAL_REVIEW"
    else:
        final_status = "ERROR"
    
    library_usage = None
    if result.get("stage2") and result["stage2"].get("library_usage"):
        library_usage = {
            "insightface": result["stage2"]["library_usage"]["insightface"],
            "deepface": result["stage2"]["library_usage"]["deepface"],
            "nudenet": "GPU" if ONNX_GPU_AVAILABLE else "CPU",
            "gpu_used": result["stage2"].get("gpu_used", False)
        }
    
    return {
        "image_filename": image_filename,
        "validation_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "photo_type": result.get("photo_type"),
        "final_status": final_status,
        "final_reason": result["final_reason"],
        "final_action": result["final_action"],
        "final_decision": final_decision,
        "image_was_cropped": result.get("image_was_cropped", False),
        "cropped_image_base64": result.get("cropped_image_base64"),
        "checklist_summary": result.get("checklist_summary"),
        "stage1_checks": result["stage1"]["checks"],
        "stage2_checks": result.get("stage2", {}).get("checks", {}) if result.get("stage2") else None,
        "library_usage": library_usage,
        "validation_approach": "hybrid" if library_usage else "stage1_only"
    }

def get_gpu_info() -> dict:
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        return {
            "available": GPU_AVAILABLE,
            "tensorflow_gpu": TF_GPU_AVAILABLE,
            "onnx_gpu": ONNX_GPU_AVAILABLE,
            "nudenet_gpu": ONNX_GPU_AVAILABLE,
            "gpu_count": len(gpus),
            "cuda_available": tf.test.is_built_with_cuda(),
        }
    except:
        return {
            "available": GPU_AVAILABLE,
            "tensorflow_gpu": TF_GPU_AVAILABLE,
            "onnx_gpu": ONNX_GPU_AVAILABLE,
            "nudenet_gpu": ONNX_GPU_AVAILABLE,
        }

def validate_single_image_sync(temp_path, photo_type, profile_data, reference_path=None, use_deepface_gender=False):
    try:
        result = validate_photo_complete_hybrid(
            image_path=temp_path,
            photo_type=photo_type,
            profile_data=profile_data,
            reference_photo_path=reference_path,
            run_stage2=True,
            use_deepface_gender=use_deepface_gender
        )
        return result
    except Exception as e:
        return {
            "final_decision": "ERROR",
            "final_action": "ERROR",
            "final_reason": str(e),
            "stage1": {"checks": {}},
        }

# ==================== SINGLE IMAGE ENDPOINTS ====================

# @app.post("/api/v3/validate/single/primary", response_model=SingleImageResponse)
# async def validate_single_primary_photo(
#     photo: UploadFile = File(...),
#     matri_id: str = Form(...),
#     gender: str = Form(...),
#     age: int = Form(...),
#     use_deepface_gender: bool = Form(False)
# ):
#     """Validate single PRIMARY photo with full GPU acceleration"""
#     start_time = time.time()
#     temp_file_path = None
    
#     try:
#         if age < 18:
#             raise HTTPException(status_code=400, detail="Age must be 18+")
#         if gender not in ["Male", "Female"]:
#             raise HTTPException(status_code=400, detail="Gender must be Male/Female")
        
#         temp_file_path = save_upload_file_tmp(photo)
        
#         profile_data = {"matri_id": matri_id, "gender": gender, "age": age}
        
#         result = validate_photo_complete_hybrid(
#             image_path=temp_file_path,
#             photo_type="PRIMARY",
#             profile_data=profile_data,
#             run_stage2=True,
#             use_deepface_gender=use_deepface_gender
#         )
        
#         cleanup_temp_files(temp_file_path)
        
#         response_data = format_validation_result(result, photo.filename)
#         response_time = round(time.time() - start_time, 3)
        
#         return SingleImageResponse(
#             success=response_data["final_status"] == "ACCEPTED",
#             message=response_data["final_reason"],
#             data=response_data,
#             response_time_seconds=response_time,
#             library_usage=response_data.get("library_usage"),
#             gpu_info=get_gpu_info()
#         )
#     except Exception as e:
#         cleanup_temp_files(temp_file_path)
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/api/v3/validate/single/secondary", response_model=SingleImageResponse)
# async def validate_single_secondary_photo(
#     photo: UploadFile = File(...),
#     matri_id: str = Form(...),
#     gender: str = Form(...),
#     age: int = Form(...),
#     reference_photo: Optional[UploadFile] = File(None)
# ):
#     """Validate single SECONDARY photo"""
#     start_time = time.time()
#     temp_file_path = None
#     temp_reference_path = None
    
#     try:
#         if age < 18:
#             raise HTTPException(status_code=400, detail="Age must be 18+")
        
#         temp_file_path = save_upload_file_tmp(photo)
        
#         if reference_photo:
#             temp_reference_path = save_upload_file_tmp(reference_photo)
        
#         profile_data = {"matri_id": matri_id, "gender": gender, "age": age}
        
#         result = validate_photo_complete_hybrid(
#             image_path=temp_file_path,
#             photo_type="SECONDARY",
#             profile_data=profile_data,
#             reference_photo_path=temp_reference_path,
#             run_stage2=True
#         )
        
#         cleanup_temp_files(temp_file_path, temp_reference_path)
        
#         response_data = format_validation_result(result, photo.filename)
#         response_time = round(time.time() - start_time, 3)
        
#         return SingleImageResponse(
#             success=response_data["final_status"] == "ACCEPTED",
#             message=response_data["final_reason"],
#             data=response_data,
#             response_time_seconds=response_time,
#             library_usage=response_data.get("library_usage"),
#             gpu_info=get_gpu_info()
#         )
#     except Exception as e:
#         cleanup_temp_files(temp_file_path, temp_reference_path)
#         raise HTTPException(status_code=500, detail=str(e))

# # ==================== BATCH ENDPOINTS (FIXED FOR SWAGGER UI) ====================

# @app.post("/api/v3/validate/batch/primary", response_model=MultiImageResponse)
# async def validate_batch_primary_photos(
#     matri_id: str = Form(...),
#     gender: str = Form(...),
#     age: int = Form(...),
#     use_deepface_gender: bool = Form(False),
#     photos: List[UploadFile] = File(...)  # FIXED: List must be last parameter
# ):
#     """
#     Validate multiple PRIMARY photos in batch
    
#     **FIXED**: Now works in Swagger UI
#     **Note**: Upload multiple files by clicking the file input multiple times
#     """
#     start_time = time.time()
#     temp_files = []
    
#     try:
#         if len(photos) > 10:
#             raise HTTPException(status_code=400, detail="Max 10 images per batch")
#         if age < 18:
#             raise HTTPException(status_code=400, detail="Age must be 18+")
        
#         profile_data = {"matri_id": matri_id, "gender": gender, "age": age}
        
#         for photo in photos:
#             temp_path = save_upload_file_tmp(photo)
#             temp_files.append((temp_path, photo.filename))
        
#         loop = asyncio.get_event_loop()
#         validation_tasks = []
        
#         for temp_path, filename in temp_files:
#             task = loop.run_in_executor(
#                 executor,
#                 validate_single_image_sync,
#                 temp_path,
#                 "PRIMARY",
#                 profile_data,
#                 None,
#                 use_deepface_gender
#             )
#             validation_tasks.append((task, filename))
        
#         results = []
#         for task, filename in validation_tasks:
#             result = await task
#             formatted_result = format_validation_result(result, filename)
#             results.append(formatted_result)
        
#         cleanup_temp_files(*[path for path, _ in temp_files])
        
#         response_time = round(time.time() - start_time, 3)
        
#         results = convert_numpy_types(results)
        
#         summary = {
#             "total": len(results),
#             "approved": sum(1 for r in results if r["final_decision"] == "APPROVE"),
#             "rejected": sum(1 for r in results if r["final_decision"] == "REJECT"),
#             "suspended": sum(1 for r in results if r["final_decision"] == "SUSPEND"),
#             "review_needed": sum(1 for r in results if r["final_decision"] == "MANUAL_REVIEW"),
#             "processing_time_seconds": response_time,
#             "avg_time_per_image": round(response_time / len(results), 3) if results else 0
#         }
        
#         library_usage_summary = {
#             "insightface_used": sum(1 for r in results if r.get("library_usage")),
#             "deepface_used": sum(1 for r in results if r.get("library_usage") and r["library_usage"].get("deepface")),
#             "nudenet_gpu": ONNX_GPU_AVAILABLE,
#             "gpu_accelerated_count": sum(1 for r in results if (r.get("library_usage") or {}).get("gpu_used", False))
#         }
        
#         return MultiImageResponse(
#             success=True,
#             message=f"Batch validation: {summary['approved']} approved, {summary['rejected']} rejected",
#             batch_id=str(uuid.uuid4()),
#             total_images=len(results),
#             results=results,
#             summary=summary,
#             response_time_seconds=response_time,
#             library_usage_summary=library_usage_summary,
#             gpu_info=get_gpu_info()
#         )
#     except Exception as e:
#         cleanup_temp_files(*[path for path, _ in temp_files])
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/api/v3/validate/batch/secondary", response_model=MultiImageResponse)
# async def validate_batch_secondary_photos(
#     matri_id: str = Form(...),
#     gender: str = Form(...),
#     age: int = Form(...),
#     reference_photo: Optional[UploadFile] = File(None),
#     photos: List[UploadFile] = File(...)  # FIXED: List must be last
# ):
#     """
#     Validate multiple SECONDARY photos in batch
    
#     **FIXED**: Now works in Swagger UI
#     """
#     start_time = time.time()
#     temp_files = []
#     temp_reference_path = None
    
#     try:
#         if len(photos) > 10:
#             raise HTTPException(status_code=400, detail="Max 10 images")
#         if age < 18:
#             raise HTTPException(status_code=400, detail="Age must be 18+")
        
#         profile_data = {"matri_id": matri_id, "gender": gender, "age": age}
        
#         if reference_photo:
#             temp_reference_path = save_upload_file_tmp(reference_photo)
        
#         for photo in photos:
#             temp_path = save_upload_file_tmp(photo)
#             temp_files.append((temp_path, photo.filename))
        
#         loop = asyncio.get_event_loop()
#         validation_tasks = []
        
#         for temp_path, filename in temp_files:
#             task = loop.run_in_executor(
#                 executor,
#                 validate_single_image_sync,
#                 temp_path,
#                 "SECONDARY",
#                 profile_data,
#                 temp_reference_path,
#                 False
#             )
#             validation_tasks.append((task, filename))
        
#         results = []
#         for task, filename in validation_tasks:
#             result = await task
#             formatted_result = format_validation_result(result, filename)
#             results.append(formatted_result)
        
#         cleanup_temp_files(*[path for path, _ in temp_files], temp_reference_path)
        
#         response_time = round(time.time() - start_time, 3)
#         results = convert_numpy_types(results)
        
#         summary = {
#             "total": len(results),
#             "approved": sum(1 for r in results if r["final_decision"] == "APPROVE"),
#             "rejected": sum(1 for r in results if r["final_decision"] == "REJECT"),
#             "suspended": sum(1 for r in results if r["final_decision"] == "SUSPEND"),
#             "review_needed": sum(1 for r in results if r["final_decision"] == "MANUAL_REVIEW"),
#             "processing_time_seconds": response_time,
#             "avg_time_per_image": round(response_time / len(results), 3) if results else 0
#         }
        
#         return MultiImageResponse(
#             success=True,
#             message=f"Batch validation: {summary['approved']} approved",
#             batch_id=str(uuid.uuid4()),
#             total_images=len(results),
#             results=results,
#             summary=summary,
#             response_time_seconds=response_time,
#             gpu_info=get_gpu_info()
#         )
#     except Exception as e:
#         cleanup_temp_files(*[path for path, _ in temp_files], temp_reference_path)
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/validatephoto")
async def validate_mixed_batch(
    matri_id: str = Form(...),
    gender: str = Form(...),
    age: int = Form(...),
    use_deepface_gender: bool = Form(False),
    reference_photo: Optional[UploadFile] = File(None),
    primary_photos: Optional[List[UploadFile]] = File(None),
    secondary_photos: Optional[List[UploadFile]] = File(None)
):
    """
    Validate mixed batch of PRIMARY and SECONDARY photos
    
    **FIXED**: Now works in Swagger UI
    **Note**: Upload files in respective fields
    """
    start_time = time.time()
    temp_files = []
    temp_reference_path = None
    
    try:
        total_photos = (len(primary_photos) if primary_photos else 0) + (len(secondary_photos) if secondary_photos else 0)
        
        if total_photos == 0:
            raise HTTPException(status_code=400, detail="At least one photo required")
        if total_photos > 10:
            raise HTTPException(status_code=400, detail="Max 10 images")
        if age < 18:
            raise HTTPException(status_code=400, detail="Age must be 18+")
        
        profile_data = {"matri_id": matri_id, "gender": gender, "age": age}
        
        if reference_photo:
            temp_reference_path = save_upload_file_tmp(reference_photo)
        
        if primary_photos:
            for photo in primary_photos:
                temp_path = save_upload_file_tmp(photo)
                temp_files.append((temp_path, photo.filename, "PRIMARY"))
        
        if secondary_photos:
            for photo in secondary_photos:
                temp_path = save_upload_file_tmp(photo)
                temp_files.append((temp_path, photo.filename, "SECONDARY"))
        
        loop = asyncio.get_event_loop()
        validation_tasks = []
        
        for temp_path, filename, photo_type in temp_files:
            ref_path = temp_reference_path if photo_type == "SECONDARY" else None
            use_df = use_deepface_gender if photo_type == "PRIMARY" else False
            
            task = loop.run_in_executor(
                executor,
                validate_single_image_sync,
                temp_path,
                photo_type,
                profile_data,
                ref_path,
                use_df
            )
            validation_tasks.append((task, filename, photo_type))
        
        results = {"primary": [], "secondary": []}
        
        for task, filename, photo_type in validation_tasks:
            result = await task
            formatted_result = format_validation_result(result, filename)
            if photo_type == "PRIMARY":
                results["primary"].append(formatted_result)
            else:
                results["secondary"].append(formatted_result)
        
        cleanup_temp_files(*[path for path, _, _ in temp_files], temp_reference_path)
        
        response_time = round(time.time() - start_time, 3)
        results = convert_numpy_types(results)
        
        all_results = results["primary"] + results["secondary"]
        
        summary = {
            "total": len(all_results),
            "primary_count": len(results["primary"]),
            "secondary_count": len(results["secondary"]),
            "approved": sum(1 for r in all_results if r["final_decision"] == "APPROVE"),
            "rejected": sum(1 for r in all_results if r["final_decision"] == "REJECT"),
            "processing_time_seconds": response_time,
        }
        
        return {
            "success": True,
            "message": f"Mixed batch: {summary['approved']} approved",
            "batch_id": str(uuid.uuid4()),
            "total_images": len(all_results),
            "results": results,
            "summary": summary,
            "response_time_seconds": response_time,
            "gpu_info": get_gpu_info()
        }
    except Exception as e:
        cleanup_temp_files(*[path for path, _, _ in temp_files], temp_reference_path)
        raise HTTPException(status_code=500, detail=str(e))

# ==================== INFO ENDPOINTS ====================

# @app.get("/")
# async def root():
#     return {
#         "service": "Photo Validation API - Full GPU Acceleration",
#         "version": "3.0.0",
#         "gpu_status": get_gpu_info(),
#         "features": [
#             "InsightFace GPU - Face detection & matching",
#             "DeepFace GPU - Age & ethnicity (PRIMARY)",
#             "NudeNet GPU - NSFW detection (10x faster)",
#             "4x faster than CPU"
#         ]
#     }

# @app.get("/health")
# async def health():
#     return {
#         "status": "healthy",
#         "timestamp": datetime.utcnow().isoformat(),
#         "gpu": get_gpu_info()
#     }

# @app.get("/api/v3/gpu/info")
# async def gpu_info():
#     return get_gpu_info()

@app.on_event("startup")
async def startup_event():
    temp_dir = os.path.join(tempfile.gettempdir(), "photo_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    
    print("="*70)
    print("Photo Validation API - Full GPU v3.0.0")
    print("="*70)
    print(f"InsightFace GPU: {ONNX_GPU_AVAILABLE}")
    print(f"DeepFace GPU: {TF_GPU_AVAILABLE}")
    print(f"NudeNet GPU: {ONNX_GPU_AVAILABLE}")
    print("="*70)

@app.on_event("shutdown")
async def shutdown_event():
    executor.shutdown(wait=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("hybrid_dev_repo_api:app", host="0.0.0.0", port=8001, reload=True, workers=1)
