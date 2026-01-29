import cv2
import os
import re
import numpy as np
import base64
import tempfile
import easyocr
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
import clip

# Register AVIF/HEIF support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORT = True
    print("AVIF/HEIF support enabled via pillow-heif")
except ImportError:
    HEIF_SUPPORT = False
    print("Warning: pillow-heif not installed. AVIF/HEIF support disabled. Install with: pip install pillow-heif")
from retinaface import RetinaFace
from nudenet import NudeDetector

# InsightFace imports (BACKBONE)
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# DeepFace imports (AGE & ETHNICITY ONLY)
from deepface import DeepFace

# TensorFlow GPU configuration (CRITICAL)
import tensorflow as tf

# ==================== GPU CONFIGURATION ====================

def configure_gpu():
    """Configure TensorFlow and ONNX Runtime to use GPU efficiently"""
    print("="*60)
    print("GPU CONFIGURATION")
    print("="*60)

    # Check GPU availability for TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs Available: {len(gpus)}")

    if gpus:
        try:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Set visible devices
            tf.config.set_visible_devices(gpus[0], 'GPU')

            # Get GPU details
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            print(f"GPU Device: {gpus[0]}")
            print(f"GPU Name: {gpu_details.get('device_name', 'Unknown')}")

            # Check CUDA and cuDNN
            print(f"CUDA Available: {tf.test.is_built_with_cuda()}")
            try:
                print(f"GPU is being used: {tf.test.is_gpu_available(cuda_only=True)}")
            except:
                print("GPU availability check completed")

            # Set mixed precision for better performance
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision enabled (float16)")

        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("WARNING: No GPU detected for TensorFlow. Running on CPU.")

    print("="*60 + "\n")
    return len(gpus) > 0

# Configure GPU at module load
TF_GPU_AVAILABLE = configure_gpu()

# Force GPU usage for DeepFace
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Try to import GPU-related libraries for ONNX Runtime (InsightFace)
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        ONNX_GPU_AVAILABLE = True
        print(f"ONNX Runtime GPU Available: CUDA")
    else:
        ONNX_GPU_AVAILABLE = False
        print("ONNX Runtime: Running on CPU")
except:
    ONNX_GPU_AVAILABLE = False
    print("ONNX Runtime not available or GPU not detected")

# Combined GPU availability
GPU_AVAILABLE = TF_GPU_AVAILABLE or ONNX_GPU_AVAILABLE
print(f"Overall GPU Available: {GPU_AVAILABLE}")


# ==================== STAGE 1 CONFIGURATION ====================

# Stage 1 Thresholds
MIN_RESOLUTION = 360
MIN_FACE_SIZE = 80
BLUR_REJECT = 21
MIN_FACE_COVERAGE_S1 = 0.05
MAX_YAW_ANGLE = 45
BLUR_AFTER_CROP_MIN = 40

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"
}

# NudeNet detector (loaded once)
nsfw_detector = NudeDetector()


# ==================== STAGE 2 CONFIGURATION ====================

# Gender validation
GENDER_CONFIDENCE_THRESHOLD = 0.70

# Ethnicity thresholds (DeepFace)
INDIAN_PROBABILITY_MIN = 0.20
DISALLOWED_ETHNICITIES = {
    "white": 0.60,
    "black": 0.60,
    "asian": 0.50,
    "middle eastern": 0.60,
    "latino hispanic": 0.60
}

# Age variance thresholds
AGE_VARIANCE_PASS = 8
AGE_VARIANCE_REVIEW = 15

# Face coverage for Stage 2
MIN_FACE_COVERAGE_S2 = 0.05

# Enhancement/filter detection
FILTER_SATURATION_THRESHOLD = 1.5

# Paper-of-photo indicators
PAPER_WHITE_THRESHOLD = 240

# Face similarity thresholds
DUPLICATE_THRESHOLD_STRICT = 0.40
DUPLICATE_THRESHOLD_REVIEW = 0.50
PRIMARY_PERSON_MATCH_THRESHOLD = 0.50


# ==================== INSIGHTFACE INITIALIZATION (BACKBONE) ====================

print("Initializing InsightFace (BACKBONE) with GPU acceleration...")
if ONNX_GPU_AVAILABLE:
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("InsightFace: Using CUDA GPU")
else:
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    print("InsightFace: Using CPU")

app.prepare(ctx_id=0 if ONNX_GPU_AVAILABLE else -1, det_size=(640, 640))

# Initialize recognition model for face comparison
print("Loading InsightFace recognition model...")
try:
    if ONNX_GPU_AVAILABLE:
        recognition_model = get_model('buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    else:
        recognition_model = get_model('buffalo_l', providers=['CPUExecutionProvider'])
    print("InsightFace recognition model loaded successfully")
except Exception as e:
    print(f"Note: Recognition model loading info: {e}")
    recognition_model = None

print(f"InsightFace initialized successfully (BACKBONE) - GPU: {ONNX_GPU_AVAILABLE}")


# ==================== DEEPFACE CONFIGURATION (AGE & ETHNICITY ONLY) ====================

print("DeepFace GPU Configuration:")
print(f"  - TensorFlow GPU Available: {TF_GPU_AVAILABLE}")
print(f"  - Mixed Precision: float16 enabled")
print("DeepFace will be used for:")
print("  - Age verification (PRIMARY photos only) - GPU accelerated")
print("  - Ethnicity validation (PRIMARY photos only) - GPU accelerated")
print("  - Gender validation (optional fallback) - GPU accelerated")


# ==================== CLIP MODEL INITIALIZATION ====================

print("Initializing CLIP model for style detection...")
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=CLIP_DEVICE)
clip_model.eval()
print(f"CLIP model loaded on {CLIP_DEVICE}")

# CLIP prompt groups for detection
CLIP_PROMPTS = {
    "ghibli_anime": [
        "a studio ghibli style illustration",
        "anime style artwork",
        "cartoon illustration",
        "hand drawn animation style",
        "digital painting illustration"
    ],
    "over_filtered": [
        "over filtered face photo",
        "heavily edited portrait",
        "beauty filter selfie",
        "airbrushed face photo",
        "instagram filter photo"
    ],
    "photo_of_photo": [
        "a photo of a printed photograph",
        "a photograph of a photograph",
        "photo of a photo frame",
        "picture of a printed image",
        "photo taken of a screen"
    ],
    "screenshot": [
        "a screenshot",
        "screen capture",
        "mobile screenshot",
        "computer screenshot",
        "image of a phone screen"
    ],
    "real_photo": [
        "a natural real photograph",
        "a realistic unedited photo",
        "a real photo taken by a camera",
        "a natural portrait photograph"
    ],
    "cartoon": [
        "a cartoon image",
        "cartoon character drawing",
        "comic book style illustration",
        "animated cartoon style",
        "digital cartoon artwork",
        "cartoon style rendering"
    ]
}

# Tokenize all prompts once at startup
_clip_all_prompts = sum(CLIP_PROMPTS.values(), [])
_clip_text_tokens = clip.tokenize(_clip_all_prompts).to(CLIP_DEVICE)

# Build prompt index map
_clip_prompt_idx = {}
_cursor = 0
for k, v in CLIP_PROMPTS.items():
    _clip_prompt_idx[k] = (_cursor, _cursor + len(v))
    _cursor += len(v)

print("CLIP prompts tokenized and ready")


# ==================== STAGE 1 UTILITY FUNCTIONS ====================

def load_image(image_path):
    """Load image using PIL (supports webp, avif, heif) and convert to OpenCV BGR format"""
    ext = os.path.splitext(image_path.lower())[1]

    # Use PIL for formats not well-supported by OpenCV
    if ext in {'.webp', '.avif', '.heif', '.heic'}:
        # Check if HEIF support is available for AVIF/HEIF files
        if ext in {'.avif', '.heif', '.heic'} and not HEIF_SUPPORT:
            print(f"ERROR: Cannot load {ext} file - pillow-heif not installed. Run: pip install pillow-heif")
            return None
        try:
            pil_img = Image.open(image_path)
            # Convert to RGB if necessary (handles RGBA, P mode, etc.)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            # Convert PIL to numpy array (RGB)
            img_rgb = np.array(pil_img)
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            return img_bgr
        except Exception as e:
            print(f"PIL failed to load {image_path}: {e}")
            return None
    else:
        # Use OpenCV for standard formats (jpg, png, gif, etc.)
        return cv2.imread(image_path)

def image_to_base64(image_array):
    """Convert numpy image array to base64 string"""
    try:
        _, buffer = cv2.imencode('.jpg', image_array)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return base64_str
    except Exception as e:
        print(f"Error converting image to base64: {str(e)}")
        return None

def reject(reason, checks, cropped_image=None):
    return {
        "stage": 1,
        "result": "REJECT",
        "reason": reason,
        "checks": checks,
        "cropped_image": cropped_image
    }

def pass_stage(checks, cropped_image=None):
    return {
        "stage": 1,
        "result": "PASS",
        "reason": None,
        "checks": checks,
        "cropped_image": cropped_image
    }

def is_supported_format(image_path):
    ext = os.path.splitext(image_path.lower())[1]
    return ext in SUPPORTED_EXTENSIONS

def is_resolution_ok(img):
    h, w = img.shape[:2]
    return min(h, w) >= MIN_RESOLUTION

def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_orientation_ok(landmarks):
    """Eyes must be above nose"""
    le_y = landmarks["left_eye"][1] 
    re_y = landmarks["right_eye"][1]
    nose_y = landmarks["nose"][1]
    
    if le_y > nose_y or re_y > nose_y:
        return False
    return True

def is_face_covered(landmarks):
    """If mouth landmarks missing → likely mask / full cover"""
    return (
        "mouth_left" not in landmarks or
        "mouth_right" not in landmarks
    )


def detect_mask_or_face_covering(img, face_area, landmarks):
    """
    Advanced mask/face covering detection using image analysis.

    Detects masks, bandanas, scarves, or other coverings over nose/mouth by:
    1. Comparing skin tone between upper face (forehead/cheeks) and lower face (nose/mouth)
    2. Analyzing texture differences between upper and lower face
    3. Detecting uniform color blocks in the mouth/nose region
    4. Checking for edge patterns typical of fabric/masks

    Returns: (is_covered: bool, reason: str, confidence: float)
    """
    try:
        img_h, img_w = img.shape[:2]

        # Handle face_area as list [x1, y1, x2, y2] or tuple
        if isinstance(face_area, (list, tuple)):
            x1, y1, x2, y2 = int(face_area[0]), int(face_area[1]), int(face_area[2]), int(face_area[3])
        else:
            return False, "Invalid face area format", 0.0

        face_width = x2 - x1
        face_height = y2 - y1

        # Validate face dimensions
        if face_width <= 0 or face_height <= 0:
            return False, "Invalid face dimensions", 0.0

        # Get landmark positions
        left_eye = landmarks["left_eye"]
        right_eye = landmarks["right_eye"]
        nose = landmarks["nose"]
        mouth_left = landmarks.get("mouth_left")
        mouth_right = landmarks.get("mouth_right")

        # If mouth landmarks are missing, definitely covered
        if mouth_left is None or mouth_right is None:
            return True, "Mouth landmarks not detected - face likely covered", 0.95

        # Calculate face regions using landmarks directly (more reliable)
        eye_y = int((left_eye[1] + right_eye[1]) / 2)
        nose_y = int(nose[1])
        mouth_y = int((mouth_left[1] + mouth_right[1]) / 2)

        # Use face center for x coordinates
        face_center_x = (x1 + x2) // 2

        # Define upper face region (area around eyes/forehead - visible skin)
        # Use a region around the eyes where we know skin should be visible
        upper_y1 = max(0, eye_y - int(face_height * 0.2))
        upper_y2 = max(upper_y1 + 10, min(img_h, eye_y + int(face_height * 0.05)))
        upper_x1 = max(0, face_center_x - int(face_width * 0.35))
        upper_x2 = min(img_w, face_center_x + int(face_width * 0.35))

        # Define lower face region (nose to below mouth - where mask would be)
        lower_y1 = max(0, nose_y)
        lower_y2 = max(lower_y1 + 10, min(img_h, mouth_y + int(face_height * 0.25)))
        lower_x1 = max(0, face_center_x - int(face_width * 0.35))
        lower_x2 = min(img_w, face_center_x + int(face_width * 0.35))

        # Ensure valid region dimensions
        if upper_y2 <= upper_y1:
            upper_y2 = upper_y1 + 20
        if upper_x2 <= upper_x1:
            upper_x2 = upper_x1 + 20
        if lower_y2 <= lower_y1:
            lower_y2 = lower_y1 + 20
        if lower_x2 <= lower_x1:
            lower_x2 = lower_x1 + 20

        # Clamp to image boundaries
        upper_y1, upper_y2 = max(0, upper_y1), min(img_h, upper_y2)
        upper_x1, upper_x2 = max(0, upper_x1), min(img_w, upper_x2)
        lower_y1, lower_y2 = max(0, lower_y1), min(img_h, lower_y2)
        lower_x1, lower_x2 = max(0, lower_x1), min(img_w, lower_x2)

        # Extract regions
        if (upper_y2 <= upper_y1 or upper_x2 <= upper_x1 or
            lower_y2 <= lower_y1 or lower_x2 <= lower_x1):
            return False, f"Could not extract face regions (upper: {upper_y1}-{upper_y2}, {upper_x1}-{upper_x2}, lower: {lower_y1}-{lower_y2}, {lower_x1}-{lower_x2})", 0.0

        upper_region = img[upper_y1:upper_y2, upper_x1:upper_x2]
        lower_region = img[lower_y1:lower_y2, lower_x1:lower_x2]

        if upper_region.size == 0 or lower_region.size == 0:
            return False, "Empty face regions", 0.0

        # Ensure minimum region size for analysis
        if upper_region.shape[0] < 5 or upper_region.shape[1] < 5:
            return False, "Upper face region too small", 0.0
        if lower_region.shape[0] < 5 or lower_region.shape[1] < 5:
            return False, "Lower face region too small", 0.0

        # Convert to different color spaces for analysis
        upper_hsv = cv2.cvtColor(upper_region, cv2.COLOR_BGR2HSV)
        lower_hsv = cv2.cvtColor(lower_region, cv2.COLOR_BGR2HSV)

        upper_lab = cv2.cvtColor(upper_region, cv2.COLOR_BGR2LAB)
        lower_lab = cv2.cvtColor(lower_region, cv2.COLOR_BGR2LAB)

        upper_gray = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)
        lower_gray = cv2.cvtColor(lower_region, cv2.COLOR_BGR2GRAY)

        # ========== BEARD DETECTION ==========
        # Beards have specific characteristics that distinguish them from masks:
        # 1. Dark color (low brightness/value in HSV)
        # 2. Brown/black hue (natural hair colors)
        # 3. High texture variance (hair strands)
        # 4. NOT uniform color (unlike masks)

        lower_brightness = np.mean(lower_hsv[:, :, 2])
        lower_hue_mean = np.mean(lower_hsv[:, :, 0])
        lower_sat_mean = np.mean(lower_hsv[:, :, 1])
        lower_hue_std = np.std(lower_hsv[:, :, 0])
        lower_sat_std = np.std(lower_hsv[:, :, 1])
        lower_brightness_std = np.std(lower_hsv[:, :, 2])

        # Calculate texture metrics
        lower_laplacian = cv2.Laplacian(lower_gray, cv2.CV_64F).var()
        upper_laplacian = cv2.Laplacian(upper_gray, cv2.CV_64F).var()

        # Edge detection
        lower_edges = cv2.Canny(lower_gray, 50, 150)
        edge_density = np.count_nonzero(lower_edges) / lower_edges.size

        # Beard characteristics:
        # - Dark (brightness < 100 typically)
        # - Natural hair hue (0-30 for brown/red, or very low saturation for black/gray)
        # - High texture (laplacian variance > 200)
        # - Non-uniform color (std > threshold)
        # - Hair-like edge patterns

        is_likely_beard = False
        beard_indicators = 0

        # Check 1: Dark lower face (beard is typically darker)
        if lower_brightness < 120:
            beard_indicators += 1

        # Check 2: Natural hair color range (brown/black/gray)
        # Black/gray beard: low saturation (<50)
        # Brown beard: hue in 0-30 range with moderate saturation
        if lower_sat_mean < 60 or (lower_hue_mean < 30 and lower_sat_mean < 100):
            beard_indicators += 1

        # Check 3: High texture variance (hair strands create texture)
        if lower_laplacian > 150:
            beard_indicators += 1

        # Check 4: Non-uniform color distribution (unlike solid-color masks)
        # Beards have varied brightness due to hair strands
        if lower_brightness_std > 20 or lower_hue_std > 10:
            beard_indicators += 1

        # Check 5: High edge density with natural pattern
        # Beards have fine edges from hair, masks have uniform or patterned edges
        if edge_density > 0.15 and lower_laplacian > 200:
            beard_indicators += 1

        # If 3+ beard indicators, likely a beard not a mask
        if beard_indicators >= 3:
            is_likely_beard = True

        issues = []
        confidence_scores = []

        # ========== CHECK 1: Skin tone difference ==========
        upper_hue_mean = np.mean(upper_hsv[:, :, 0])
        hue_diff = abs(upper_hue_mean - lower_hue_mean)

        # Only flag if NOT a beard pattern and significant hue difference
        # Masks typically have hues outside skin/hair range (blue=100-130, green=35-85, etc.)
        if hue_diff > 25 and not is_likely_beard:
            # Check if lower face hue is in unnatural range (not skin or hair color)
            if lower_hue_mean > 30 and lower_hue_mean < 170:  # Unnatural hue range
                issues.append(f"Significant hue difference with unnatural color ({hue_diff:.1f}, hue:{lower_hue_mean:.1f})")
                confidence_scores.append(0.75)

        # ========== CHECK 2: Saturation difference ==========
        upper_sat_mean = np.mean(upper_hsv[:, :, 1])
        sat_diff = abs(upper_sat_mean - lower_sat_mean)

        # High saturation in lower face suggests colored mask (not beard)
        if sat_diff > 40 and lower_sat_mean > 80 and not is_likely_beard:
            issues.append(f"Saturation mismatch in lower face ({sat_diff:.1f})")
            confidence_scores.append(0.6)

        # ========== CHECK 3: Color uniformity in lower face ==========
        # Masks tend to be MORE uniform than natural skin/beard
        # If lower face is VERY uniform (solid color mask)
        if lower_hue_std < 5 and lower_sat_std < 10 and not is_likely_beard:
            issues.append(f"Lower face has unusually uniform color (hue_std: {lower_hue_std:.1f}, sat_std: {lower_sat_std:.1f})")
            confidence_scores.append(0.7)

        # ========== CHECK 4: LAB color space analysis ==========
        upper_a_mean = np.mean(upper_lab[:, :, 1])  # a* channel (green-red)
        lower_a_mean = np.mean(lower_lab[:, :, 1])
        upper_b_mean = np.mean(upper_lab[:, :, 2])  # b* channel (blue-yellow)
        lower_b_mean = np.mean(lower_lab[:, :, 2])

        a_diff = abs(upper_a_mean - lower_a_mean)
        b_diff = abs(upper_b_mean - lower_b_mean)

        # Skin tone mismatch - but not for beards (which have different a*/b* than skin)
        # Only flag if it looks like a colored mask (high b* difference suggests blue/yellow mask)
        if (a_diff > 20 or b_diff > 25) and not is_likely_beard:
            # Check if lower face has unnatural colors
            if abs(lower_b_mean - 128) > 20:  # Far from neutral b* suggests colored mask
                issues.append(f"Unnatural color in lower face (a*:{a_diff:.1f}, b*:{b_diff:.1f})")
                confidence_scores.append(0.75)

        # ========== CHECK 5: Texture analysis ==========
        # Masks typically have LESS texture than skin/beard, OR very regular pattern
        texture_ratio = lower_laplacian / (upper_laplacian + 1e-6)

        # Only flag if lower face is TOO SMOOTH (mask) not too textured (beard)
        if texture_ratio < 0.3 and not is_likely_beard:
            issues.append(f"Lower face unusually smooth (ratio: {texture_ratio:.2f}) - possible mask")
            confidence_scores.append(0.7)

        # ========== CHECK 6: Blue/cloth color detection ==========
        lower_blue = np.mean(lower_region[:, :, 0])  # Blue channel
        lower_green = np.mean(lower_region[:, :, 1])
        lower_red = np.mean(lower_region[:, :, 2])

        # Check for blue-dominant region (common mask color) - NOT beard
        blue_dominant = lower_blue > lower_red + 25 and lower_blue > lower_green + 15
        if blue_dominant:
            issues.append("Blue-dominant color in lower face region (possible mask)")
            confidence_scores.append(0.85)

        # Check for cyan/teal (common bandana color)
        cyan_teal = lower_blue > lower_red + 20 and abs(lower_blue - lower_green) < 20 and lower_blue > 100
        if cyan_teal:
            issues.append("Cyan/teal color in lower face region (possible bandana)")
            confidence_scores.append(0.85)

        # Check for green-dominant (green mask/bandana)
        green_dominant = lower_green > lower_red + 25 and lower_green > lower_blue + 15
        if green_dominant:
            issues.append("Green-dominant color in lower face (possible mask)")
            confidence_scores.append(0.8)

        # ========== CHECK 7: Brightness difference ==========
        upper_brightness = np.mean(upper_hsv[:, :, 2])
        brightness_diff = abs(upper_brightness - lower_brightness)

        # Large brightness difference with bright lower face (white mask)
        if brightness_diff > 60 and lower_brightness > upper_brightness and not is_likely_beard:
            issues.append(f"Bright covering on lower face ({brightness_diff:.1f})")
            confidence_scores.append(0.65)

        # ========== CHECK 8: Sharp color boundary detection ==========
        # Masks have sharp edges where they meet skin, beards have gradual transition
        # Check the transition zone between upper and lower face
        mid_y = (upper_y2 + lower_y1) // 2
        transition_height = max(10, int(face_height * 0.1))
        trans_y1 = max(0, mid_y - transition_height // 2)
        trans_y2 = min(img_h, mid_y + transition_height // 2)

        if trans_y2 > trans_y1 and lower_x2 > lower_x1:
            transition_region = img[trans_y1:trans_y2, lower_x1:lower_x2]
            if transition_region.size > 0 and transition_region.shape[0] >= 3:
                trans_gray = cv2.cvtColor(transition_region, cv2.COLOR_BGR2GRAY)
                # Calculate vertical gradient (sharp edge = high gradient)
                gradient = np.abs(np.diff(trans_gray.astype(float), axis=0)).mean()

                # Very sharp boundary suggests mask edge
                if gradient > 30 and not is_likely_beard:
                    issues.append(f"Sharp color boundary detected (gradient: {gradient:.1f})")
                    confidence_scores.append(0.7)

        # ========== FINAL DECISION ==========
        # Require stronger evidence if beard-like features are present
        min_issues_required = 2 if is_likely_beard else 2

        if len(issues) >= 3:
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.7
            return True, f"Face covering detected: {'; '.join(issues[:3])}", min(avg_confidence + 0.1, 0.95)
        elif len(issues) >= min_issues_required:
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.6
            # For 2 issues, require higher confidence
            if avg_confidence >= 0.65:
                return True, f"Possible face covering: {'; '.join(issues)}", avg_confidence
        elif len(issues) == 1 and confidence_scores and confidence_scores[0] >= 0.8:
            # Single issue needs very high confidence
            return True, f"Likely face covering: {issues[0]}", confidence_scores[0]

        return False, "No face covering detected", 0.0

    except Exception as e:
        print(f"Mask detection error: {str(e)}")
        return False, f"Mask detection error: {str(e)}", 0.0

def detect_hand_occlusion_improved(img, face_area, landmarks):
    """Simplified and more reliable hand occlusion detection"""
    try:
        x1, y1, x2, y2 = face_area
        face_width = x2 - x1
        face_height = y2 - y1
        
        left_eye = landmarks["left_eye"]
        right_eye = landmarks["right_eye"]
        nose = landmarks["nose"]
        
        critical_regions = {
            "left_eye": {
                "center": left_eye,
                "radius_x": face_width * 0.08,
                "radius_y": face_height * 0.06
            },
            "right_eye": {
                "center": right_eye,
                "radius_x": face_width * 0.08,
                "radius_y": face_height * 0.06
            },
            "nose": {
                "center": nose,
                "radius_x": face_width * 0.10,
                "radius_y": face_height * 0.10
            }
        }
        
        padding = 30
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(img.shape[1], x2 + padding)
        y2_pad = min(img.shape[0], y2 + padding)
        
        face_roi = img[y1_pad:y2_pad, x1_pad:x2_pad]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray_face, 50, 150)
        
        for region_name, region_info in critical_regions.items():
            cx, cy = region_info["center"]
            rx, ry = region_info["radius_x"], region_info["radius_y"]
            
            cx_roi = int(cx - x1_pad)
            cy_roi = int(cy - y1_pad)
            rx_int = int(rx)
            ry_int = int(ry)
            
            region_x1 = max(0, cx_roi - rx_int)
            region_y1 = max(0, cy_roi - ry_int)
            region_x2 = min(face_roi.shape[1], cx_roi + rx_int)
            region_y2 = min(face_roi.shape[0], cy_roi + ry_int)
            
            if region_x2 <= region_x1 or region_y2 <= region_y1:
                continue
            
            feature_region = face_roi[region_y1:region_y2, region_x1:region_x2]
            
            if feature_region.size == 0:
                continue
            
            hsv_region = cv2.cvtColor(feature_region, cv2.COLOR_BGR2HSV)
            
            if "eye" in region_name:
                avg_brightness = np.mean(hsv_region[:, :, 2])
                if avg_brightness < 25:
                    return True, f"Hand covering {region_name} - feature not visible"
            
            feature_edges = edges[region_y1:region_y2, region_x1:region_x2]
            edge_density = np.count_nonzero(feature_edges) / feature_edges.size if feature_edges.size > 0 else 0
            
            if edge_density > 0.40:
                return True, f"Hand/object detected covering {region_name}"
        
        upper_face = face_roi[:int(face_roi.shape[0] * 0.6), :]
        upper_edges = cv2.Canny(upper_face, 50, 150)
        contours, _ = cv2.findContours(upper_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < face_width * face_height * 0.08:
                continue
            
            for region_name, region_info in critical_regions.items():
                if "eye" not in region_name:
                    continue
                
                cx, cy = region_info["center"]
                cx_roi = int(cx - x1_pad)
                cy_roi = int(cy - y1_pad)
                
                if cv2.pointPolygonTest(contour, (cx_roi, cy_roi), False) > 0:
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = area / hull_area
                        if solidity < 0.7:
                            return True, f"Hand/object detected near {region_name}"
        
        return False, "No facial feature occlusion detected"
    
    except Exception as e:
        print(f"Hand occlusion detection error: {str(e)}")
        return False, f"Occlusion detection error: {str(e)}"


def check_yaw_improved(landmarks, img_shape):
    """Enhanced yaw detection with multiple methods"""
    left_eye = np.array(landmarks["left_eye"])
    right_eye = np.array(landmarks["right_eye"])
    nose = np.array(landmarks["nose"])
    
    dl = np.linalg.norm(nose - left_eye)
    dr = np.linalg.norm(nose - right_eye)
    distance_ratio = abs(dl - dr) / max(dl, dr)
    yaw_angle = distance_ratio * 90
    
    eye_midpoint_x = (left_eye[0] + right_eye[0]) / 2
    nose_x = nose[0]
    horizontal_offset = abs(nose_x - eye_midpoint_x)
    eye_distance = np.linalg.norm(right_eye - left_eye)
    
    offset_ratio = horizontal_offset / eye_distance if eye_distance > 0 else 1.0
    
    MAX_YAW_ANGLE = 45
    MAX_OFFSET_RATIO = 0.35
    
    issues = []
    
    if yaw_angle > MAX_YAW_ANGLE:
        issues.append(f"Side angle detected ({yaw_angle:.1f}°)")
    
    if offset_ratio > MAX_OFFSET_RATIO:
        issues.append(f"Face not centered (nose offset: {offset_ratio:.2f})")
    
    if issues:
        return False, yaw_angle, "; ".join(issues)
    
    return True, yaw_angle, f"Frontal face verified ({yaw_angle:.1f}°)"


def check_yaw_relaxed(landmarks, img_shape):
    """Relaxed yaw detection for secondary single person photos - allows more side angle"""
    left_eye = np.array(landmarks["left_eye"])
    right_eye = np.array(landmarks["right_eye"])
    nose = np.array(landmarks["nose"])

    dl = np.linalg.norm(nose - left_eye)
    dr = np.linalg.norm(nose - right_eye)
    distance_ratio = abs(dl - dr) / max(dl, dr)
    yaw_angle = distance_ratio * 90

    eye_midpoint_x = (left_eye[0] + right_eye[0]) / 2
    nose_x = nose[0]
    horizontal_offset = abs(nose_x - eye_midpoint_x)
    eye_distance = np.linalg.norm(right_eye - left_eye)

    offset_ratio = horizontal_offset / eye_distance if eye_distance > 0 else 1.0

    # Relaxed thresholds for secondary single person photos
    MAX_YAW_ANGLE_RELAXED = 55  # was 30
    MAX_OFFSET_RATIO_RELAXED = 0.65  # was 0.24

    issues = []

    if yaw_angle > MAX_YAW_ANGLE_RELAXED:
        issues.append(f"Side angle detected ({yaw_angle:.1f}°)")

    if offset_ratio > MAX_OFFSET_RATIO_RELAXED:
        issues.append(f"Face not centered (nose offset: {offset_ratio:.2f})")

    if issues:
        return False, yaw_angle, "; ".join(issues)

    return True, yaw_angle, f"Face angle verified ({yaw_angle:.1f}°)"


def check_face_symmetry(img, face_area, landmarks):
    """Check if face is frontal by verifying both sides are equally visible"""
    try:
        x1, y1, x2, y2 = face_area
        face_roi = img[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return True, "Could not verify face symmetry"
        
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        h, w = gray_face.shape
        
        left_eye_x = landmarks["left_eye"][0] - x1
        right_eye_x = landmarks["right_eye"][0] - x1
        nose_x = landmarks["nose"][0] - x1
        
        face_center_x = w / 2
        nose_offset_ratio = abs(nose_x - face_center_x) / w

        # Relaxed threshold: 25% (was 15%) to be consistent with yaw check
        # This avoids rejecting faces that pass the yaw angle check
        if nose_offset_ratio > 0.35:
            return False, f"Face not frontal - nose offset {nose_offset_ratio*100:.1f}% from center"

        left_eye_from_left = left_eye_x
        right_eye_from_right = w - right_eye_x

        if left_eye_from_left > 0 and right_eye_from_right > 0:
            distance_ratio = max(left_eye_from_left, right_eye_from_right) / min(left_eye_from_left, right_eye_from_right)

            # Relaxed threshold: 1.6 (was 1.4)
            if distance_ratio > 2.0:
                return False, f"Face not frontal - one side significantly more visible than other"
        
        left_half = gray_face[:, :w//2]
        right_half = gray_face[:, w//2:]
        right_half_flipped = cv2.flip(right_half, 1)
        
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:, :min_width]
        right_half_flipped = right_half_flipped[:, :min_width]
        
        diff = cv2.absdiff(left_half, right_half_flipped)
        asymmetry = np.mean(diff)
        
        if asymmetry > 70:
            return False, f"Face not frontal - significant asymmetry detected (score: {asymmetry:.1f})"
        
        face_width = x2 - x1
        left_side_width = nose_x - left_eye_x
        right_side_width = right_eye_x - nose_x
        
        if left_side_width > 0 and right_side_width > 0:
            width_ratio = max(left_side_width, right_side_width) / min(left_side_width, right_side_width)
            
            if width_ratio > 1.5:
                return False, f"Face not frontal - uneven face width distribution"
        
        return True, f"Face is frontal (symmetry score: {asymmetry:.1f})"
    
    except Exception as e:
        print(f"Face symmetry check error: {str(e)}")
        return True, f"Face symmetry check error: {str(e)}"

def calculate_face_coverage(face_area, img_shape):
    """Calculate what percentage of the image the face covers"""
    fw = face_area[2] - face_area[0]
    fh = face_area[3] - face_area[1]
    face_area_pixels = fw * fh
    
    img_h, img_w = img_shape[:2]
    img_area_pixels = img_h * img_w
    
    coverage = face_area_pixels / img_area_pixels
    return coverage

def crop_image_for_face_coverage(img, face_area, target_coverage=MIN_FACE_COVERAGE_S1):
    """Crop the image so that the face covers at least target_coverage of the image"""
    img_h, img_w = img.shape[:2]
    
    fx1, fy1, fx2, fy2 = face_area
    fw = fx2 - fx1
    fh = fy2 - fy1
    face_area_pixels = fw * fh
    
    current_coverage = face_area_pixels / (img_h * img_w)
    
    if current_coverage >= target_coverage:
        return img, False
    
    required_img_area = face_area_pixels / target_coverage
    
    face_aspect = fw / fh
    
    crop_w = int(np.sqrt(required_img_area * face_aspect))
    crop_h = int(np.sqrt(required_img_area / face_aspect))
    
    face_center_x = (fx1 + fx2) // 2
    face_center_y = (fy1 + fy2) // 2
    
    crop_x1 = max(0, face_center_x - crop_w // 2)
    crop_y1 = max(0, face_center_y - crop_h // 2)
    crop_x2 = min(img_w, crop_x1 + crop_w)
    crop_y2 = min(img_h, crop_y1 + crop_h)
    
    if crop_x2 - crop_x1 < crop_w:
        crop_x1 = max(0, crop_x2 - crop_w)
    if crop_y2 - crop_y1 < crop_h:
        crop_y1 = max(0, crop_y2 - crop_h)
    
    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    return cropped_img, True


# ==================== STAGE 1 NSFW / BARE BODY CHECK ====================

def get_compatible_image_path(image_path, img=None):
    """
    Returns a path compatible with libraries that use cv2.imread internally (NudeNet, DeepFace).
    For AVIF/WEBP files, creates a temporary JPG copy.
    Returns (compatible_path, is_temp) tuple.
    """
    ext = os.path.splitext(image_path.lower())[1]
    if ext in {'.avif', '.webp', '.heif', '.heic'}:
        if img is None:
            img = load_image(image_path)
        if img is None:
            return None, False
        # Create temp jpg file
        temp_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
        cv2.imwrite(temp_path, img)
        return temp_path, True
    return image_path, False


def check_nsfw_stage1(image_path, img=None):
    """Stage-1 NSFW policy: ANY nudity / bare body → REJECT"""
    disallowed_classes = {
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "FEMALE_BREAST_COVERED",
        "MALE_BREAST_COVERED",
        "BELLY_EXPOSED",
        "BUTTOCKS_EXPOSED",
        "BUTTOCKS_COVERED",
        "UNDERWEAR",
        "SWIMWEAR"
    }

    # Get compatible path for NudeNet (doesn't support avif/webp)
    compatible_path, is_temp = get_compatible_image_path(image_path, img)
    if compatible_path is None:
        return False, "Could not read image for NSFW check"

    detections = nsfw_detector.detect(compatible_path)

    # Cleanup temp file
    if is_temp:
        os.remove(compatible_path)
    
    for d in detections:
        if d["class"] in disallowed_classes and d["score"] > 0.6:
            return False, f"Disallowed content detected ({d['class']})"
    
    return True, None


# ==================== GROUP PHOTO VALIDATION (INSIGHTFACE) ====================

def find_primary_person_in_group(
    group_photo_path: str,
    reference_photo_path: str
) -> Tuple[bool, Optional[int], Optional[float], str]:
    """Find the primary person in a group photo using InsightFace"""
    try:
        group_img = load_image(group_photo_path)
        ref_img = load_image(reference_photo_path)
        
        group_faces = app.get(group_img)
        
        if not group_faces or len(group_faces) == 0:
            return False, None, None, "No faces detected in group photo"
        
        if len(group_faces) == 1:
            return False, None, None, "Only one face detected - not a group photo"
        
        ref_faces = app.get(ref_img)
        if not ref_faces or len(ref_faces) == 0:
            return False, None, None, "No face detected in reference photo"
        
        ref_embedding = ref_faces[0].embedding
        
        best_match_index = None
        best_match_similarity = -1
        
        for idx, face in enumerate(group_faces):
            similarity = np.dot(ref_embedding, face.embedding) / (
                np.linalg.norm(ref_embedding) * np.linalg.norm(face.embedding)
            )
            
            if similarity > best_match_similarity:
                best_match_similarity = similarity
                best_match_index = idx
        
        if best_match_index is not None and best_match_similarity > PRIMARY_PERSON_MATCH_THRESHOLD:
            return True, best_match_index, best_match_similarity, f"Primary person found at face #{best_match_index + 1}"
        else:
            return False, best_match_index, best_match_similarity, f"Primary person not clearly identifiable in group photo (best match similarity: {best_match_similarity:.3f})"
            
    except Exception as e:
        return False, None, None, f"Error finding primary person: {str(e)}"


# ==================== STAGE 1 MAIN VALIDATOR ====================

def stage1_validate(
    image_path: str,
    photo_type: str = "PRIMARY",
    reference_photo_path: Optional[str] = None,
    profile_data: Optional[Dict] = None
):
    """Stage 1 validation - Basic quality and appropriateness checks"""

    checks = {}
    cropped_image = None

    # FORMAT
    if not is_supported_format(image_path):
        return reject("Unsupported image format", checks)
    checks["format"] = "PASS"

    # IMAGE READ
    img = load_image(image_path)
    if img is None:
        return reject("Invalid or unreadable image", checks)
    checks["image_read"] = "PASS"

    # FACE DETECTION (pass numpy array to avoid cv2.imread issues with avif/webp)
    faces = RetinaFace.detect_faces(img)
    if not faces:
        return reject("No face detected", checks)

    face_count = len(faces)
    checks["face_count"] = f"{face_count} face(s) detected"

    # PRIMARY PHOTO: Must have exactly 1 face
    if photo_type == "PRIMARY":
        if face_count > 1:
            return reject("Group photo not allowed as primary photo. Primary photo must contain only your face.", checks)
        checks["photo_type_validation"] = "PASS - Single face for PRIMARY photo"

    # SECONDARY PHOTO: Validation includes NSFW, face detection, quality, resolution, blur, orientation, and matching
    # For single-person secondary: also check face coverage and auto-crop if needed
    # For group photos (multiple faces): skip face coverage and auto-crop
    elif photo_type == "SECONDARY":
        # RESOLUTION CHECK for secondary photos
        if not is_resolution_ok(img):
            img_h, img_w = img.shape[:2]
            return reject(f"Low resolution image ({img_w}x{img_h}px, minimum: {MIN_RESOLUTION}px)", checks)
        checks["resolution"] = "PASS"

        # QUALITY (BLUR) CHECK for secondary photos
        blur = blur_score(img)
        checks["blur_score"] = f"{blur:.2f}"

        if blur < BLUR_REJECT:
            return reject("Image is too blurry", checks)
        checks["quality"] = "PASS"

        # NSFW CHECK for secondary photos
        nsfw_ok, nsfw_reason = check_nsfw_stage1(image_path, img)
        if not nsfw_ok:
            return reject(nsfw_reason, checks)
        checks["nsfw"] = "PASS"

        # FACE DETECTION for secondary photos
        checks["face_detection"] = f"PASS - {face_count} face(s) detected"

        # Get first face for orientation check (applicable for both single and group)
        first_face = list(faces.values())[0]
        first_landmarks = first_face["landmarks"]

        # ORIENTATION CHECK for secondary photos (eyes must be above nose)
        if not is_orientation_ok(first_landmarks):
            return reject("Improper image orientation", checks)
        checks["orientation"] = "PASS"

        # Handle group photos vs individual photos
        if face_count > 1:
            # YAW / FACE ANGLE CHECK for group photos (strict check)
            yaw_ok, yaw_angle, yaw_message = check_yaw_improved(first_landmarks, img.shape)
            checks["yaw_angle"] = yaw_message

            if not yaw_ok:
                return reject(yaw_message, checks)
            checks["face_pose"] = "PASS"
            # ========== GROUP PHOTO VALIDATION ==========
            # For group photos: Only check NSFW, face detection, quality, resolution, and face matching
            # Skip: age, gender, ethnicity, face size, face coverage, cropping

            checks["photo_type_validation"] = f"PASS - Group photo with {face_count} faces detected"

            if reference_photo_path is None:
                return reject("Group photo detected but no reference photo provided to identify primary person", checks)

            # Detect all faces in the group photo and check if primary person is present
            found, face_idx, similarity, message = find_primary_person_in_group(
                image_path,
                reference_photo_path
            )

            checks["group_photo_validation"] = message

            if not found:
                # REJECT: Primary person not found in the group photo
                return reject(
                    f"The person in the primary photo is not present in the group photo. {message}",
                    checks
                )

            # ACCEPT: Primary person found in the group photo
            checks["face_matching"] = f"PASS - Primary person found in group photo (face #{face_idx + 1}, similarity: {similarity:.3f})"
            checks["face_coverage_check"] = "SKIPPED - Not applicable for group photos"
            checks["cropping_applied"] = "NO - Group photos are not cropped"

            # For group photos, we're done after matching check
            return pass_stage(checks, cropped_image)

        else:
            # ========== INDIVIDUAL PHOTO VALIDATION ==========
            # For individual photos: Check NSFW, face detection, quality, resolution, face matching, face coverage, yaw (relaxed), and age

            checks["photo_type_validation"] = "PASS - Single face for SECONDARY photo"

            # YAW / FACE ANGLE CHECK for single person secondary photos (relaxed threshold)
            yaw_ok, yaw_angle, yaw_message = check_yaw_relaxed(first_landmarks, img.shape)
            checks["yaw_angle"] = yaw_message

            if not yaw_ok:
                return reject(yaw_message, checks)
            checks["face_pose"] = "PASS"

            # AGE CHECK for single person secondary photos
            if profile_data and profile_data.get("age"):
                age_result = validate_age_deepface(image_path, profile_data.get("age"), img)
                checks["age"] = age_result

                if age_result["status"] == "FAIL":
                    if age_result.get("action") == "SUSPEND":
                        return reject(f"Age verification failed: {age_result['reason']}", checks)
                    else:
                        return reject(f"Age mismatch: {age_result['reason']}", checks)
            else:
                checks["age"] = "SKIPPED - No profile age provided"

            # Get face information
            face = list(faces.values())[0]
            area = face["facial_area"]

            # FACE SIZE CHECK for individual secondary photos (basic validation)
            fw = area[2] - area[0]
            fh = area[3] - area[1]
            face_size_min = min(fw, fh)

            # Add face dimensions to checks for debugging
            checks["face_dimensions"] = f"{fw}x{fh}px"

            # Smart face size check: if coverage is good, allow smaller faces
            coverage = calculate_face_coverage(area, img.shape)
            effective_min_size = 40 if coverage >= MIN_FACE_COVERAGE_S1 else MIN_FACE_SIZE

            if face_size_min < effective_min_size:
                return reject(f"Face too small or unclear (size: {face_size_min:.0f}px, minimum: {effective_min_size}px)", checks)
            checks["face_size"] = "PASS"

            # FACE MATCHING: Check if face matches primary photo
            if reference_photo_path is None:
                return reject("No reference photo provided to verify face matching", checks)

            # Use InsightFace to verify the person matches
            try:
                ref_img = load_image(reference_photo_path)
                ref_faces = app.get(ref_img)
                if not ref_faces or len(ref_faces) == 0:
                    return reject("No face detected in reference photo", checks)

                curr_img = load_image(image_path)
                curr_faces = app.get(curr_img)
                if not curr_faces or len(curr_faces) == 0:
                    return reject("No face detected in secondary photo", checks)

                ref_embedding = ref_faces[0].embedding
                curr_embedding = curr_faces[0].embedding

                similarity = np.dot(ref_embedding, curr_embedding) / (
                    np.linalg.norm(ref_embedding) * np.linalg.norm(curr_embedding)
                )

                if similarity < PRIMARY_PERSON_MATCH_THRESHOLD:
                    return reject(
                        f"Face does not match primary photo (similarity: {similarity:.3f})",
                        checks
                    )

                checks["face_matching"] = f"PASS - Matches primary photo (similarity: {similarity:.3f})"
            except Exception as e:
                return reject(f"Error during face matching: {str(e)}", checks)

            # FACE COVERAGE CHECK: Ensure at least 5% face coverage
            coverage = calculate_face_coverage(area, img.shape)
            checks["face_coverage_original"] = f"{coverage * 100:.2f}%"

            if coverage < MIN_FACE_COVERAGE_S1:
                # Perform crop to achieve 5% coverage for individual secondary photos
                img_for_validation, was_cropped = crop_image_for_face_coverage(img, area)

                if not was_cropped:
                    return reject("Face coverage insufficient (less than 5%) and auto-crop failed", checks)

                checks["cropping_applied"] = "YES"

                # Re-detect face in cropped image (pass numpy array directly)
                faces_cropped = RetinaFace.detect_faces(img_for_validation)
                if not faces_cropped:
                    return reject("Face lost after cropping", checks)

                face = list(faces_cropped.values())[0]
                area = face["facial_area"]

                new_coverage = calculate_face_coverage(area, img_for_validation.shape)
                checks["face_coverage_after_crop"] = f"{new_coverage * 100:.2f}%"

                cropped_image = img_for_validation
            else:
                checks["cropping_applied"] = "NO"
                checks["face_coverage_check"] = f"PASS - Face coverage sufficient ({coverage * 100:.2f}%)"

            # For individual secondary photos, we're done
            return pass_stage(checks, cropped_image)
    
    face = list(faces.values())[0]
    area = face["facial_area"]
    landmarks = face["landmarks"]
    
    # FACE COVERAGE & CROPPING
    coverage = calculate_face_coverage(area, img.shape)
    checks["face_coverage_original"] = f"{coverage * 100:.2f}%"
    
    img_for_validation = img
    was_cropped = False
    
    if photo_type == "PRIMARY" and coverage < MIN_FACE_COVERAGE_S1:
        img_for_validation, was_cropped = crop_image_for_face_coverage(img, area)
        checks["cropping_applied"] = "YES"

        # Save cropped image as jpg (avif/webp may not be supported by cv2.imwrite)
        base_path = os.path.splitext(image_path)[0]
        cropped_temp_path = f"{base_path}_cropped.jpg"
        cv2.imwrite(cropped_temp_path, img_for_validation)

        # Pass numpy array directly to RetinaFace (avoids cv2.imread issues)
        faces_cropped = RetinaFace.detect_faces(img_for_validation)
        if not faces_cropped:
            os.remove(cropped_temp_path)
            return reject("Face lost after cropping", checks)
        
        face = list(faces_cropped.values())[0]
        area = face["facial_area"]
        landmarks = face["landmarks"]
        
        new_coverage = calculate_face_coverage(area, img_for_validation.shape)
        checks["face_coverage_after_crop"] = f"{new_coverage * 100:.2f}%"
        
        if photo_type == "PRIMARY":
            blur_after_crop = blur_score(img_for_validation)
            checks["blur_after_crop"] = f"{blur_after_crop:.2f}"
            
            if blur_after_crop < BLUR_AFTER_CROP_MIN:
                os.remove(cropped_temp_path)
                return reject(f"Image quality too low after cropping (blur score: {blur_after_crop:.2f})", checks)
            
            crop_h, crop_w = img_for_validation.shape[:2]
            if min(crop_h, crop_w) < MIN_RESOLUTION:
                os.remove(cropped_temp_path)
                return reject(f"Cropped image resolution too low ({crop_w}x{crop_h})", checks)
        
        cropped_image = img_for_validation
        validation_image_path = cropped_temp_path
    else:
        checks["cropping_applied"] = "NO"
        validation_image_path = image_path
    
    # FACE SIZE
    fw = area[2] - area[0]
    fh = area[3] - area[1]
    
    if min(fw, fh) < MIN_FACE_SIZE:
        if was_cropped:
            os.remove(cropped_temp_path)
        return reject("Face too small or unclear", checks, cropped_image)
    
    checks["face_size"] = "PASS"
    
    # RESOLUTION
    if not is_resolution_ok(img_for_validation):
        if was_cropped:
            os.remove(cropped_temp_path)
        return reject("Low resolution image", checks, cropped_image)
    checks["resolution"] = "PASS"
    
    # BLUR
    blur = blur_score(img_for_validation)
    checks["blur_score"] = f"{blur:.2f}"
    
    if blur < BLUR_REJECT:
        if was_cropped:
            os.remove(cropped_temp_path)
        return reject("Image is too blurry", checks, cropped_image)
    
    checks["blur"] = "PASS"
    
    # ORIENTATION
    if not is_orientation_ok(landmarks):
        if was_cropped:
            os.remove(cropped_temp_path)
        return reject("Improper image orientation", checks, cropped_image)
    
    checks["orientation"] = "PASS"
    
    # YAW / SIDE FACE CHECK
    yaw_ok, yaw_angle, yaw_message = check_yaw_improved(landmarks, img_for_validation.shape)
    checks["yaw_angle"] = yaw_message
    
    if not yaw_ok:
        if was_cropped:
            os.remove(cropped_temp_path)
        return reject(yaw_message, checks, cropped_image)
    
    checks["face_pose"] = "PASS"
    
    # FACE SYMMETRY CHECK (PRIMARY ONLY)
    if photo_type == "PRIMARY":
        is_frontal, symmetry_message = check_face_symmetry(img_for_validation, area, landmarks)
        checks["face_symmetry"] = symmetry_message
        
        if not is_frontal:
            if was_cropped:
                os.remove(cropped_temp_path)
            return reject(symmetry_message, checks, cropped_image)
        
        checks["face_frontal"] = "PASS - " + symmetry_message
    else:
        checks["face_symmetry"] = "SKIPPED - Not required for family/group photos"
    
    # MASK / FACE COVER - Enhanced detection
    # First check if landmarks are missing (basic check)
    if is_face_covered(landmarks):
        if was_cropped:
            os.remove(cropped_temp_path)
        return reject("Face is covered or wearing a mask (mouth landmarks missing)", checks, cropped_image)

    # Advanced mask/face covering detection using image analysis
    is_masked, mask_reason, mask_confidence = detect_mask_or_face_covering(img_for_validation, area, landmarks)
    if is_masked:
        checks["face_cover_details"] = {
            "detected": True,
            "reason": mask_reason,
            "confidence": mask_confidence
        }
        if was_cropped:
            os.remove(cropped_temp_path)
        return reject(f"Face is covered or wearing a mask ({mask_reason})", checks, cropped_image)

    checks["face_cover"] = "PASS"
    checks["face_cover_details"] = {
        "detected": False,
        "reason": mask_reason,
        "confidence": mask_confidence
    }
    
    # HAND OCCLUSION DETECTION (PRIMARY ONLY)
    if photo_type == "PRIMARY":
        is_occluded, occlusion_msg = detect_hand_occlusion_improved(img_for_validation, area, landmarks)
        checks["hand_occlusion"] = occlusion_msg
        
        if is_occluded:
            if was_cropped:
                os.remove(cropped_temp_path)
            return reject(occlusion_msg, checks, cropped_image)
        
        checks["hand_occlusion"] = "PASS - " + occlusion_msg
    else:
        checks["hand_occlusion"] = "SKIPPED - Not required for family/group photos"
    
    # NSFW / BARE BODY
    nsfw_ok, nsfw_reason = check_nsfw_stage1(validation_image_path, img_for_validation)
    if not nsfw_ok:
        if was_cropped:
            os.remove(cropped_temp_path)
        return reject(nsfw_reason, checks, cropped_image)
    
    checks["nsfw"] = "PASS"
    
    # CLEANUP & FINAL
    if was_cropped:
        os.remove(cropped_temp_path)
    
    return pass_stage(checks, cropped_image)


# ==================== INSIGHTFACE FACE ANALYSIS ====================

def analyze_face_insightface(img_path: str) -> Dict:
    """InsightFace analysis for face detection and embeddings"""
    try:
        img = load_image(img_path)
        if img is None:
            return {"error": "Could not read image", "data": None}
        
        faces = app.get(img)
        
        if not faces or len(faces) == 0:
            return {"error": "No face detected", "data": None}
        
        face = faces[0]
        
        face_data = {
            "bbox": face.bbox.tolist(),
            "kps": face.kps.tolist(),
            "det_score": float(face.det_score),
            "embedding": face.embedding,
            "gender": face.gender if hasattr(face, 'gender') else None,
            "age": int(face.age) if hasattr(face, 'age') else None,
        }
        
        return {"error": None, "data": face_data}
        
    except Exception as e:
        return {"error": str(e), "data": None}


# ==================== DEEPFACE: AGE VALIDATION ====================

def validate_age_deepface(img_path: str, profile_age: int, img: np.ndarray = None) -> Dict:
    """
    Age verification using DeepFace with GPU acceleration (PRIMARY photos only)
    DeepFace has better age accuracy than InsightFace
    """
    try:
        print(f"[DeepFace GPU] Running age detection for profile age: {profile_age}...")

        # Get compatible path for DeepFace (doesn't support avif/webp)
        compatible_path, is_temp = get_compatible_image_path(img_path, img)
        if compatible_path is None:
            return {"status": "ERROR", "reason": "Could not read image"}

        # DeepFace analyze
        result = DeepFace.analyze(
            img_path=compatible_path,
            actions=['age'],
            enforce_detection=True,
            detector_backend='retinaface',
            silent=True
        )

        # Cleanup temp file
        if is_temp:
            os.remove(compatible_path)
        
        # Handle list or dict result
        if isinstance(result, list):
            result = result[0]
        
        detected_age = int(result.get('age', 0))
        
        if detected_age == 0:
            return {
                "status": "REVIEW",
                "reason": "Could not detect age from photo",
                "detected_age": None,
                "profile_age": profile_age
            }
        
        variance = abs(detected_age - profile_age)
        
        # CRITICAL: Check for underage
        if detected_age < 18:
            return {
                "status": "FAIL",
                "reason": f"Underage detected: {detected_age} years",
                "detected_age": detected_age,
                "profile_age": profile_age,
                "variance": variance,
                "action": "SUSPEND"
            }
        
        # Extra scrutiny for young ages
        if detected_age < 23:
            return {
                "status": "REVIEW",
                "reason": f"Young age detected: {detected_age} years. Manual verification recommended.",
                "detected_age": detected_age,
                "profile_age": profile_age,
                "variance": variance
            }
        
        if variance < AGE_VARIANCE_PASS:
            return {
                "status": "PASS",
                "reason": f"Age verified: {detected_age} (profile: {profile_age}, variance: {variance} years)",
                "detected_age": detected_age,
                "profile_age": profile_age,
                "variance": variance
            }
        elif variance <= AGE_VARIANCE_REVIEW:
            return {
                "status": "REVIEW",
                "reason": f"Moderate age variance: profile {profile_age}, detected {detected_age} (variance: {variance} years)",
                "detected_age": detected_age,
                "profile_age": profile_age,
                "variance": variance
            }
        else:
            return {
                "status": "FAIL",
                "reason": f"Large age variance: profile {profile_age}, detected {detected_age} (variance: {variance} years)",
                "detected_age": detected_age,
                "profile_age": profile_age,
                "variance": variance
            }
        
    except Exception as e:
        print(f"[DeepFace GPU] Age detection error: {str(e)}")
        return {
            "status": "REVIEW",
            "reason": f"Age detection failed: {str(e)}",
            "detected_age": None,
            "profile_age": profile_age
        }


# ==================== DEEPFACE: ETHNICITY VALIDATION (GPU-ACCELERATED) ====================

def validate_ethnicity_deepface(img_path: str, img: np.ndarray = None) -> Dict:
    """
    Ethnicity validation using DeepFace with GPU acceleration (PRIMARY photos only)
    DeepFace has ethnicity detection, InsightFace doesn't

    Logic:
    1. DeepFace returns probabilities on 0-100 scale (e.g., 37.96 = 37.96%)
    2. Indian probability should be >= 30% (INDIAN_PROBABILITY_MIN = 0.20)
    3. Disallowed ethnicities should not exceed their thresholds
       - DISALLOWED_ETHNICITIES are in decimal format (0.60 = 60%)
    """
    try:
        print("[DeepFace GPU] Running ethnicity detection...")

        # Get compatible path for DeepFace (doesn't support avif/webp)
        compatible_path, is_temp = get_compatible_image_path(img_path, img)
        if compatible_path is None:
            return {"status": "ERROR", "reason": "Could not read image"}

        # DeepFace analyze
        result = DeepFace.analyze(
            img_path=compatible_path,
            actions=['race'],
            enforce_detection=True,
            detector_backend='retinaface',
            silent=True
        )

        # Cleanup temp file
        if is_temp:
            os.remove(compatible_path)
        
        # Handle list or dict result
        if isinstance(result, list):
            result = result[0]
        
        race_scores = result.get('race', {})
        
        if not race_scores:
            return {
                "status": "REVIEW",
                "reason": "Could not detect ethnicity",
                "indian_probability": None,
                "all_scores": None
            }
        
        # Get Indian probability (already on 0-100 scale from DeepFace)
        indian_prob = race_scores.get('indian', 0.0)
        
        print(f"[DeepFace GPU] Ethnicity scores: {race_scores}")
        print(f"[DeepFace GPU] Indian probability: {indian_prob:.2f}%")
        
        # Check disallowed ethnicities
        # DISALLOWED_ETHNICITIES thresholds are in decimal format (0.60 = 60%)
        # DeepFace scores are on 0-100 scale
        # So we need to multiply threshold by 100 for comparison
        for ethnicity, threshold_decimal in DISALLOWED_ETHNICITIES.items():
            # Convert threshold from decimal to percentage (0.60 -> 60.0)
            threshold_percentage = threshold_decimal * 100
            
            # Try to get the probability for this ethnicity
            prob = race_scores.get(ethnicity, 0.0)
            
            # Compare: if actual probability > threshold, reject
            if prob > threshold_percentage:
                return {
                    "status": "FAIL",
                    "reason": f"Ethnicity check failed: High {ethnicity} probability ({prob:.2f}% exceeds threshold {threshold_percentage:.0f}%)",
                    "indian_probability": indian_prob,
                    "all_scores": race_scores
                }
        
        # Check if Indian probability is sufficient
        # INDIAN_PROBABILITY_MIN = 0.20 (20%)
        # Convert to percentage for comparison: 0.30 * 100 = 30.0
        indian_threshold = INDIAN_PROBABILITY_MIN * 100
        
        if indian_prob < indian_threshold:
            return {
                "status": "REVIEW",
                "reason": f"Low Indian ethnicity probability ({indian_prob:.2f}%). Manual review recommended.",
                "indian_probability": indian_prob,
                "all_scores": race_scores
            }
        
        # PASS: Indian probability is sufficient and no disallowed ethnicity exceeds threshold
        return {
            "status": "PASS",
            "reason": f"Ethnicity verified: Indian ({indian_prob:.2f}%)",
            "indian_probability": indian_prob,
            "all_scores": race_scores
        }
        
    except Exception as e:
        print(f"[DeepFace GPU] Ethnicity detection error: {str(e)}")
        return {
            "status": "REVIEW",
            "reason": f"Ethnicity detection failed: {str(e)}",
            "indian_probability": None,
            "all_scores": None
        }


# ==================== DEEPFACE: GENDER VALIDATION (GPU-ACCELERATED, OPTIONAL FALLBACK) ====================

def validate_gender_deepface(img_path: str, profile_gender: str, img: np.ndarray = None) -> Dict:
    """
    Gender validation using DeepFace with GPU acceleration (optional fallback if InsightFace gender is unreliable)
    """
    try:
        print("[DeepFace GPU] Running gender detection...")

        # Get compatible path for DeepFace (doesn't support avif/webp)
        compatible_path, is_temp = get_compatible_image_path(img_path, img)
        if compatible_path is None:
            return {"status": "ERROR", "reason": "Could not read image"}

        result = DeepFace.analyze(
            img_path=compatible_path,
            actions=['gender'],
            enforce_detection=True,
            detector_backend='retinaface',
            silent=True
        )

        # Cleanup temp file
        if is_temp:
            os.remove(compatible_path)
        
        if isinstance(result, list):
            result = result[0]
        
        gender_scores = result.get('gender', {})
        
        if not gender_scores:
            return {
                "status": "REVIEW",
                "reason": "Could not detect gender",
                "detected": None,
                "expected": profile_gender
            }
        
        # Get dominant gender
        detected_gender = max(gender_scores, key=gender_scores.get)
        confidence = gender_scores[detected_gender] / 100.0
        
        if detected_gender.lower() != profile_gender.lower():
            return {
                "status": "FAIL",
                "reason": f"Gender mismatch: detected {detected_gender}, profile says {profile_gender}",
                "detected": detected_gender,
                "expected": profile_gender,
                "confidence": confidence
            }
        
        return {
            "status": "PASS",
            "reason": f"Gender verified as {detected_gender}",
            "detected": detected_gender,
            "expected": profile_gender,
            "confidence": confidence
        }
        
    except Exception as e:
        print(f"[DeepFace GPU] Gender detection error: {str(e)}")
        return {
            "status": "REVIEW",
            "reason": f"Gender detection failed: {str(e)}",
            "detected": None,
            "expected": profile_gender
        }


# ==================== INSIGHTFACE: GENDER VALIDATION (GPU-ACCELERATED) ====================

def validate_gender_insightface(img_path: str, profile_gender: str, face_data: Dict = None) -> Dict:
    """Gender validation using InsightFace (fast but may be less accurate)"""
    try:
        if face_data is None:
            analysis = analyze_face_insightface(img_path)
            if analysis["error"]:
                return {
                    "status": "REVIEW",
                    "reason": f"Face analysis failed: {analysis['error']}",
                    "detected": None,
                    "expected": profile_gender
                }
            face_data = analysis["data"]
        
        gender_value = face_data.get("gender")
        
        if gender_value is None:
            return {
                "status": "REVIEW",
                "reason": "Gender detection not available",
                "detected": None,
                "expected": profile_gender
            }
        
        detected_gender = "Male" if gender_value == 1 else "Female"
        confidence = 0.85
        
        if detected_gender.lower() != profile_gender.lower():
            return {
                "status": "FAIL",
                "reason": f"Gender mismatch: detected {detected_gender}, profile says {profile_gender}",
                "detected": detected_gender,
                "expected": profile_gender,
                "confidence": confidence
            }
        
        return {
            "status": "PASS",
            "reason": f"Gender verified as {detected_gender}",
            "detected": detected_gender,
            "expected": profile_gender,
            "confidence": confidence
        }
        
    except Exception as e:
        return {
            "status": "REVIEW",
            "reason": f"Gender detection failed: {str(e)}",
            "detected": None,
            "expected": profile_gender
        }


# ==================== DATABASE CHECKS REMOVED ====================
# Fraud database, celebrity database, and duplicate detection checks
# have been removed as they require database support not available in this phase


# ==================== INSIGHTFACE: FACE COVERAGE CHECK ====================

def check_face_coverage(img_path: str, face_data: Dict = None) -> Dict:
    """Face coverage check using InsightFace"""
    try:
        if face_data is None:
            analysis = analyze_face_insightface(img_path)
            if analysis["error"]:
                return {
                    "status": "REVIEW",
                    "reason": f"Face coverage check failed: {analysis['error']}"
                }
            face_data = analysis["data"]
        
        bbox = face_data.get("bbox")
        if bbox is None:
            return {
                "status": "REVIEW",
                "reason": "Could not get face bounding box"
            }
        
        face_x, face_y, face_x2, face_y2 = bbox
        face_w = face_x2 - face_x
        face_h = face_y2 - face_y
        
        img = load_image(img_path)
        img_h, img_w = img.shape[:2]
        
        face_area = face_w * face_h
        img_area = img_w * img_h
        coverage = face_area / img_area if img_area > 0 else 0
        
        if coverage < MIN_FACE_COVERAGE_S2:
            return {
                "status": "FAIL",
                "reason": f"Face too small in frame ({coverage:.2%} coverage)",
                "coverage": coverage
            }
        
        face_center_x = face_x + face_w / 2
        face_center_y = face_y + face_h / 2
        img_center_x = img_w / 2
        img_center_y = img_h / 2
        
        offset_x = abs(face_center_x - img_center_x) / img_w
        offset_y = abs(face_center_y - img_center_y) / img_h
        
        if offset_x > 0.3 or offset_y > 0.3:
            return {
                "status": "REVIEW",
                "reason": f"Face not centered. May indicate improper framing.",
                "coverage": coverage,
                "offset_x": offset_x,
                "offset_y": offset_y
            }
        
        return {
            "status": "PASS",
            "reason": f"Proper face framing ({coverage:.2%} coverage)",
            "coverage": coverage
        }
        
    except Exception as e:
        return {
            "status": "REVIEW",
            "reason": f"Face coverage check failed: {str(e)}"
        }




# ==================== CLIP-BASED STYLE DETECTION ====================

@torch.no_grad()
def clip_style_detect(image_path: str) -> Dict:
    """
    CLIP-based detection for:
    - Ghibli/Anime style
    - Over-filtered photos
    - Photo of photo
    - Screenshots
    - Cartoons

    Returns raw scores and boolean decisions.
    """
    try:
        # Handle AVIF/WEBP - use PIL directly for CLIP
        ext = os.path.splitext(image_path.lower())[1]
        if ext in {'.avif', '.webp', '.heif', '.heic'}:
            # Load with our helper and convert back to PIL
            img_cv = load_image(image_path)
            if img_cv is None:
                return {"error": "Could not load image"}
            # Convert BGR to RGB for PIL
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
        else:
            pil_image = Image.open(image_path).convert("RGB")

        image_input = clip_preprocess(pil_image).unsqueeze(0).to(CLIP_DEVICE)

        # Encode image and text
        img_feat = clip_model.encode_image(image_input)
        txt_feat = clip_model.encode_text(_clip_text_tokens)

        # Normalize
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        # Compute similarity
        probs = (100 * img_feat @ txt_feat.T).softmax(dim=-1)[0]

        # Extract scores for each category
        ghibli_score = probs[_clip_prompt_idx["ghibli_anime"][0]:_clip_prompt_idx["ghibli_anime"][1]].max().item()
        filtered_score = probs[_clip_prompt_idx["over_filtered"][0]:_clip_prompt_idx["over_filtered"][1]].max().item()
        photo_of_photo_score = probs[_clip_prompt_idx["photo_of_photo"][0]:_clip_prompt_idx["photo_of_photo"][1]].max().item()
        screenshot_score = probs[_clip_prompt_idx["screenshot"][0]:_clip_prompt_idx["screenshot"][1]].max().item()
        real_score = probs[_clip_prompt_idx["real_photo"][0]:_clip_prompt_idx["real_photo"][1]].max().item()
        cartoon_score = probs[_clip_prompt_idx["cartoon"][0]:_clip_prompt_idx["cartoon"][1]].max().item()

        return {
            # Raw scores
            "ghibli_score": ghibli_score,
            "over_filtered_score": filtered_score,
            "photo_of_photo_score": photo_of_photo_score,
            "screenshot_score": screenshot_score,
            "real_photo_score": real_score,
            "cartoon_score": cartoon_score,

            # Decisions (production-safe thresholds)
            "is_ghibli_anime": ghibli_score > 0.30 and ghibli_score > real_score,
            "is_over_filtered": filtered_score > 0.35 and filtered_score > real_score,
            "is_photo_of_photo": photo_of_photo_score > 0.25 and photo_of_photo_score > real_score,
            "is_screenshot": screenshot_score > 0.35 and screenshot_score > real_score,
            "is_cartoon": cartoon_score > 0.25 and cartoon_score > real_score,
        }

    except Exception as e:
        print(f"CLIP detection error: {e}")
        return {"error": str(e)}


def detect_all_image_issues(img_path: str) -> Dict:
    """
    Single unified function that calls CLIP once and returns all detection results.
    Returns results for: enhancement, photo_of_photo, ai_generated
    """
    # Call CLIP only ONCE
    clip_result = clip_style_detect(img_path)

    # Handle error case
    if "error" in clip_result:
        error_msg = clip_result["error"]
        return {
            "enhancement": {
                "saturation": {"status": "REVIEW", "reason": f"Detection error: {error_msg}"},
                "cartoon": {"status": "REVIEW", "reason": f"Detection error: {error_msg}"}
            },
            "photo_of_photo": {
                "status": "REVIEW",
                "reason": f"Detection error: {error_msg}"
            },
            "ai_generated": {
                "status": "REVIEW",
                "reason": f"Detection error: {error_msg}",
                "confidence": "LOW",
                "details": {"error": error_msg}
            },
            "clip_scores": clip_result
        }

    # ===== Enhancement Check =====
    enhancement = {}

    # Over-filtered check
    if clip_result["is_over_filtered"]:
        enhancement["saturation"] = {
            "status": "FAIL",
            "reason": f"Over-filtered/beauty filter detected (score: {clip_result['over_filtered_score']:.3f})"
        }
    else:
        enhancement["saturation"] = {
            "status": "PASS",
            "reason": f"Natural photo (filter score: {clip_result['over_filtered_score']:.3f})"
        }

    # Cartoon/anime check
    if clip_result["is_cartoon"] or clip_result["is_ghibli_anime"]:
        score = max(clip_result["cartoon_score"], clip_result["ghibli_score"])
        enhancement["cartoon"] = {
            "status": "FAIL",
            "reason": f"Cartoon/anime style detected (score: {score:.3f})"
        }
    else:
        enhancement["cartoon"] = {
            "status": "PASS",
            "reason": "Natural photograph"
        }

    # ===== Photo of Photo Check =====
    if clip_result["is_photo_of_photo"]:
        photo_of_photo = {
            "status": "FAIL",
            "reason": f"Photo of printed photo detected (score: {clip_result['photo_of_photo_score']:.3f})"
        }
    elif clip_result["is_screenshot"]:
        photo_of_photo = {
            "status": "FAIL",
            "reason": f"Screenshot detected (score: {clip_result['screenshot_score']:.3f})"
        }
    else:
        photo_of_photo = {
            "status": "PASS",
            "reason": f"Original digital photo (real score: {clip_result['real_photo_score']:.3f})"
        }

    # ===== AI Generated Check =====
    issues = []
    scores = {
        "ghibli_score": clip_result["ghibli_score"],
        "cartoon_score": clip_result["cartoon_score"],
        "over_filtered_score": clip_result["over_filtered_score"],
        "photo_of_photo_score": clip_result["photo_of_photo_score"],
        "screenshot_score": clip_result["screenshot_score"],
        "real_photo_score": clip_result["real_photo_score"]
    }

    if clip_result["is_ghibli_anime"]:
        issues.append(f"Ghibli/Anime style (score: {clip_result['ghibli_score']:.3f})")
    if clip_result["is_cartoon"]:
        issues.append(f"Cartoon style (score: {clip_result['cartoon_score']:.3f})")
    if clip_result["is_over_filtered"]:
        issues.append(f"Heavily filtered (score: {clip_result['over_filtered_score']:.3f})")
    if clip_result["is_photo_of_photo"]:
        issues.append(f"Photo of photo (score: {clip_result['photo_of_photo_score']:.3f})")
    if clip_result["is_screenshot"]:
        issues.append(f"Screenshot (score: {clip_result['screenshot_score']:.3f})")

    scores["total_issues"] = len(issues)

    if len(issues) >= 2:
        ai_generated = {
            "status": "FAIL",
            "reason": "Image appears to be AI-generated, cartoon, or heavily filtered",
            "confidence": "HIGH",
            "details": {"issues": issues, "scores": scores}
        }
    elif len(issues) == 1:
        ai_generated = {
            "status": "REVIEW",
            "reason": f"Potential issue detected: {issues[0]}",
            "confidence": "MEDIUM",
            "details": {"issues": issues, "scores": scores}
        }
    else:
        ai_generated = {
            "status": "PASS",
            "reason": f"Image appears to be authentic photograph (real score: {clip_result['real_photo_score']:.3f})",
            "confidence": "HIGH",
            "details": {"scores": scores}
        }

    return {
        "enhancement": enhancement,
        "photo_of_photo": photo_of_photo,
        "ai_generated": ai_generated,
        "clip_scores": clip_result
    }


# ==================== PII DETECTION VIA OCR ====================

# Initialize EasyOCR reader (GPU enabled by default if available)
print("Initializing EasyOCR for PII detection...")
EASYOCR_GPU = torch.cuda.is_available()
ocr_reader = easyocr.Reader(['en'], gpu=EASYOCR_GPU)
print(f"EasyOCR initialized - GPU: {EASYOCR_GPU}")

# PII Regex Patterns
PII_PATTERNS = {
    "phone_number": {
        "pattern": r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{10}|\d{5}[-.\s]?\d{5}",
        "description": "Phone number detected"
    },
    "email": {
        "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "description": "Email address detected"
    },
    "gmail": {
        "pattern": r"[a-zA-Z0-9._%+-]+@gmail\.com",
        "description": "Gmail address detected"
    },
    "instagram_id": {
        "pattern": r"(?:@|instagram\.com/|ig:|insta:)\s*[a-zA-Z0-9._]{1,30}|(?:instagram|insta|ig)\s*[:\-]?\s*[a-zA-Z0-9._]{1,30}",
        "description": "Instagram ID detected"
    },
    "facebook_id": {
        "pattern": r"(?:facebook\.com/|fb\.com/|fb:|facebook:)\s*[a-zA-Z0-9.]+|(?:facebook|fb)\s*[:\-]?\s*[a-zA-Z0-9.]+",
        "description": "Facebook ID detected"
    },
    "twitter_handle": {
        "pattern": r"(?:twitter\.com/|x\.com/|@)\s*[a-zA-Z0-9_]{1,15}|(?:twitter|tweet)\s*[:\-]?\s*@?[a-zA-Z0-9_]{1,15}",
        "description": "Twitter/X handle detected"
    },
    "whatsapp": {
        "pattern": r"(?:whatsapp|wa\.me/|wa:)\s*[:\-]?\s*\+?\d{10,15}",
        "description": "WhatsApp number detected"
    },
    "snapchat": {
        "pattern": r"(?:snapchat|snap|sc)\s*[:\-]?\s*[a-zA-Z0-9._-]{3,15}",
        "description": "Snapchat ID detected"
    },
    "telegram": {
        "pattern": r"(?:telegram|t\.me/|tg:)\s*[:\-]?\s*@?[a-zA-Z0-9_]{5,32}",
        "description": "Telegram ID detected"
    },
    "aadhaar": {
        "pattern": r"\d{4}[-\s]?\d{4}[-\s]?\d{4}",
        "description": "Aadhaar number pattern detected"
    },
    "pan_card": {
        "pattern": r"[A-Z]{5}\d{4}[A-Z]",
        "description": "PAN card number detected"
    },
    "website_url": {
        "pattern": r"(?:https?://)?(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?",
        "description": "Website URL detected"
    }
}


def check_pii_in_image(image_path: str) -> Dict:
    """
    Perform OCR on image and check for PII (Personally Identifiable Information).
    Uses EasyOCR directly to extract text from image.

    Returns:
        Dict with status (PASS/FAIL/REVIEW), detected PII types, and extracted text
    """
    result = {
        "status": "PASS",
        "pii_found": [],
        "pii_details": {},
        "extracted_text": "",
        "ocr_confidence": 0.0,
        "reason": "No PII detected in image",
        "gpu_used": EASYOCR_GPU
    }

    try:
        # Read image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return {
                "status": "REVIEW",
                "pii_found": [],
                "pii_details": {},
                "extracted_text": "",
                "ocr_confidence": 0.0,
                "reason": f"Failed to read image: {image_path}",
                "gpu_used": EASYOCR_GPU
            }

        # Perform OCR using EasyOCR
        ocr_results = ocr_reader.readtext(img)

        # Extract text and confidence scores
        texts = []
        confidences = []
        for detection in ocr_results:
            text = detection[1]
            confidence = detection[2]
            texts.append(text)
            confidences.append(confidence)

        extracted_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        result["extracted_text"] = extracted_text
        result["ocr_confidence"] = round(avg_confidence, 3)

        # If no text detected, pass
        if not extracted_text.strip():
            result["reason"] = "No text detected in image"
            return result

        # Check for PII patterns
        text_lower = extracted_text.lower()

        for pii_type, pii_config in PII_PATTERNS.items():
            pattern = pii_config["pattern"]
            flags = re.IGNORECASE if pii_type not in ["pan_card"] else 0

            matches = re.findall(pattern, extracted_text if pii_type == "pan_card" else text_lower, flags)

            if matches:
                # Filter out false positives (very short matches, common words)
                valid_matches = [m for m in matches if len(str(m)) > 3]

                if valid_matches:
                    result["pii_found"].append(pii_type)
                    result["pii_details"][pii_type] = {
                        "description": pii_config["description"],
                        "matches": valid_matches[:5],  # Limit to first 5 matches
                        "count": len(valid_matches)
                    }

        # Determine status based on PII found
        if result["pii_found"]:
            # Critical PII types that should fail immediately
            critical_pii = {"phone_number", "email", "gmail", "whatsapp", "aadhaar", "pan_card"}
            social_pii = {"instagram_id", "facebook_id", "twitter_handle", "snapchat", "telegram"}

            found_critical = set(result["pii_found"]) & critical_pii
            found_social = set(result["pii_found"]) & social_pii

            if found_critical:
                result["status"] = "FAIL"
                result["reason"] = f"Critical PII detected: {', '.join(found_critical)}"
            elif found_social:
                result["status"] = "FAIL"
                result["reason"] = f"Social media ID detected: {', '.join(found_social)}"
            else:
                result["status"] = "REVIEW"
                result["reason"] = f"Potential PII detected: {', '.join(result['pii_found'])}"
        else:
            result["status"] = "PASS"
            result["reason"] = "No PII detected in image text"

        return result

    except Exception as e:
        return {
            "status": "REVIEW",
            "pii_found": [],
            "pii_details": {},
            "extracted_text": "",
            "ocr_confidence": 0.0,
            "reason": f"PII check error: {str(e)}",
            "gpu_used": EASYOCR_GPU
        }


# ==================== STAGE 2 MAIN VALIDATOR (HYBRID) ====================

def stage2_validate_hybrid(
    image_path: str,
    profile_data: Dict,
    photo_type: str = "PRIMARY",
    use_deepface_gender: bool = False
) -> Dict:
    """
    Stage 2 validation with HYBRID approach:
    - InsightFace as backbone (detection, embeddings, face coverage)
    - DeepFace for age and ethnicity (PRIMARY only)
    - Optional: DeepFace for gender if InsightFace is unreliable

    Note: Fraud database, celebrity database, and duplicate detection checks
    have been removed as they require database support not available in this phase.
    """

    results = {
        "stage": 2,
        "photo_type": photo_type,
        "matri_id": profile_data.get("matri_id"),
        "gpu_used": GPU_AVAILABLE,
        "tf_gpu_used": TF_GPU_AVAILABLE,
        "onnx_gpu_used": ONNX_GPU_AVAILABLE,
        "library_usage": {
            "insightface": ["detection (GPU)" if ONNX_GPU_AVAILABLE else "detection (CPU)",
                          "embeddings (GPU)" if ONNX_GPU_AVAILABLE else "embeddings (CPU)",
                          "face_coverage"],
            "deepface": []
        },
        "checks": {},
        "checks_performed": [],
        "checks_skipped": [],
        "final_decision": None,
        "action": None,
        "reason": None,
        "early_exit": False
    }
    
    # ============= EARLY EXIT FOR SECONDARY PHOTOS =============
    # Secondary photos already completed all necessary checks in Stage 1:
    # - NSFW check
    # - Face detection and quality
    # - Face matching with primary photo
    # - Face coverage check (for individual photos only, group photos skip this)
    if photo_type == "SECONDARY":
        print("[Stage 2] SECONDARY photo detected - all checks completed in Stage 1")
        results["final_decision"] = "APPROVE"
        results["action"] = "PUBLISH"
        results["reason"] = "SECONDARY photo validation completed in Stage 1"
        results["early_exit"] = True
        results["checks_skipped"] = ["age", "gender", "ethnicity", "face_coverage",
                                     "enhancement", "photo_of_photo", "ai_generated", "pii"]
        return results

    # ============= INSIGHTFACE ANALYSIS (BACKBONE) =============
    face_data = None

    if photo_type == "PRIMARY":
        print("[InsightFace] Running face analysis (BACKBONE)...")
        analysis = analyze_face_insightface(image_path)
        face_data = analysis["data"] if not analysis["error"] else None

        if analysis["error"]:
            results["final_decision"] = "REVIEW"
            results["action"] = "SEND_TO_HUMAN"
            results["reason"] = f"Face detection failed: {analysis['error']}"
            results["early_exit"] = True
            return results
    
    # ============= PRIORITY 1: CRITICAL CHECKS =============
    
    if photo_type == "PRIMARY":
        # 1. AGE CHECK (DEEPFACE - PRIMARY ONLY, GPU-ACCELERATED)
        print("[P1 GPU] Checking age with DeepFace (GPU)...")
        results["checks"]["age"] = validate_age_deepface(image_path, profile_data.get("age", 25))
        results["checks_performed"].append("age")
        results["library_usage"]["deepface"].append("age (GPU)" if TF_GPU_AVAILABLE else "age (CPU)")
        
        if results["checks"]["age"]["status"] == "FAIL" and results["checks"]["age"].get("action") == "SUSPEND":
            results["final_decision"] = "SUSPEND"
            results["action"] = "SUSPEND_PROFILE"
            results["reason"] = "Underage detected - immediate suspension"
            results["early_exit"] = True
            results["checks_skipped"] = ["gender", "ethnicity", "face_coverage", "enhancement",
                                         "photo_of_photo", "ai_generated", "pii"]
            return results
    else:
        results["checks_skipped"].append("age")
        print("[P1 GPU] Skipping age check for SECONDARY photo")

    # ============= PRIORITY 2: HIGH IMPORTANCE CHECKS =============

    if photo_type == "PRIMARY":
        # 3. GENDER CHECK (INSIGHTFACE or DEEPFACE - GPU-ACCELERATED)
        if use_deepface_gender:
            print("[P2 GPU] Checking gender with DeepFace (GPU)...")
            results["checks"]["gender"] = validate_gender_deepface(image_path, profile_data.get("gender", "Unknown"))
            results["library_usage"]["deepface"].append("gender (GPU)" if TF_GPU_AVAILABLE else "gender (CPU)")
        else:
            print("[P2 GPU] Checking gender with InsightFace (GPU)...")
            results["checks"]["gender"] = validate_gender_insightface(image_path, profile_data.get("gender", "Unknown"), face_data)
        
        results["checks_performed"].append("gender")
        
        if results["checks"]["gender"]["status"] == "FAIL":
            results["final_decision"] = "REJECT"
            results["action"] = "SELFIE_VERIFICATION"
            results["reason"] = "Gender mismatch detected"
            results["early_exit"] = True
            results["checks_skipped"].extend(["ethnicity", "face_coverage", "enhancement",
                                         "photo_of_photo", "ai_generated", "pii"])
            return results

        # 4. ETHNICITY CHECK (DEEPFACE - PRIMARY ONLY, GPU-ACCELERATED)
        print("[P2 GPU] Checking ethnicity with DeepFace (GPU)...")
        results["checks"]["ethnicity"] = validate_ethnicity_deepface(image_path)
        results["checks_performed"].append("ethnicity")
        results["library_usage"]["deepface"].append("ethnicity (GPU)" if TF_GPU_AVAILABLE else "ethnicity (CPU)")

        if results["checks"]["ethnicity"]["status"] == "FAIL":
            results["final_decision"] = "REJECT"
            results["action"] = "SELFIE_VERIFICATION"
            results["reason"] = "Ethnicity check failed"
            results["early_exit"] = True
            results["checks_skipped"].extend(["face_coverage", "enhancement",
                                         "photo_of_photo", "ai_generated", "pii"])
            return results
    else:
        results["checks_skipped"].extend(["gender", "ethnicity"])
        print("[P2 GPU] Skipping gender/ethnicity checks for SECONDARY photo")

    # ============= PRIORITY 3: STANDARD CHECKS =============

    print("[P3 GPU] Running standard checks...")
    
    if photo_type == "PRIMARY":
        # 5. Face Coverage (INSIGHTFACE)
        results["checks"]["face_coverage"] = check_face_coverage(image_path, face_data)
        results["checks_performed"].append("face_coverage")
    else:
        results["checks_skipped"].append("face_coverage")
        print("[P3 GPU] Skipping face coverage check for SECONDARY photo")

    # 6-8. CLIP-based detection (single call for all checks)
    print("[P3 GPU] Running CLIP-based style detection...")
    clip_detection = detect_all_image_issues(image_path)

    results["checks"]["enhancement"] = clip_detection["enhancement"]
    results["checks_performed"].append("enhancement")

    results["checks"]["photo_of_photo"] = clip_detection["photo_of_photo"]
    results["checks_performed"].append("photo_of_photo")

    results["checks"]["ai_generated"] = clip_detection["ai_generated"]
    results["checks_performed"].append("ai_generated")

    # Store CLIP scores for debugging/analysis
    results["clip_scores"] = clip_detection.get("clip_scores", {})

    # 9. PII Check (OCR-based)
    print("[P3] Running PII detection via OCR...")
    results["checks"]["pii"] = check_pii_in_image(image_path)
    results["checks_performed"].append("pii")

    # ============= FINAL DECISION LOGIC =============
    
    fail_checks = []
    review_checks = []
    
    for check_name, check_result in results["checks"].items():
        if isinstance(check_result, dict) and "status" in check_result:
            if check_result["status"] == "FAIL":
                fail_checks.append(check_name)
            elif check_result["status"] == "REVIEW":
                review_checks.append(check_name)
        else:
            for sub_check, sub_result in check_result.items():
                if sub_result["status"] == "FAIL":
                    fail_checks.append(f"{check_name}.{sub_check}")
                elif sub_result["status"] == "REVIEW":
                    review_checks.append(f"{check_name}.{sub_check}")
    
    if fail_checks:
        results["final_decision"] = "REJECT"
        results["action"] = determine_rejection_action(fail_checks, results["checks"])
        results["reason"] = f"Failed checks: {', '.join(fail_checks)}"
    elif review_checks:
        results["final_decision"] = "MANUAL_REVIEW"
        results["action"] = "SEND_TO_HUMAN"
        results["reason"] = f"Requires manual review: {', '.join(review_checks)}"
    else:
        results["final_decision"] = "APPROVE"
        results["action"] = "PUBLISH"
        results["reason"] = "All checks passed"
    
    return results


def determine_rejection_action(fail_checks: List[str], all_checks: Dict) -> str:
    """Determine action based on failed checks"""

    if any(check in fail_checks for check in ["age"]):
        return "SUSPEND_PROFILE"

    if any(check in fail_checks for check in ["gender", "ethnicity"]):
        return "SELFIE_VERIFICATION"

    if "pii" in fail_checks:
        return "NUDGE_REMOVE_PII"

    if "enhancement" in fail_checks or any("enhancement" in check for check in fail_checks):
        return "NUDGE_UPLOAD_ORIGINAL"

    if "photo_of_photo" in fail_checks:
        return "NUDGE_UPLOAD_DIGITAL"

    return "NUDGE_REUPLOAD_PROPER"


def compile_checklist_summary(stage1_result: Dict, stage2_result: Optional[Dict], photo_type: str) -> Dict:
    """Compile a comprehensive 20-point checklist summary"""
    checklist = {
        "total_checks": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "review": 0,
        "checks": []
    }
    
    # Stage 1 checks
    stage1_checks_config = [
        {"id": 1, "name": "Format Validation", "stage": "S1", "check_key": "format"},
        {"id": 2, "name": "Image Readable", "stage": "S1", "check_key": "image_read"},
        {"id": 3, "name": "Face Detection", "stage": "S1", "check_key": "face_count"},
        {"id": 4, "name": "Auto-Cropping", "stage": "S1", "check_key": "cropping_applied"},
        {"id": 5, "name": "Face Size", "stage": "S1", "check_key": "face_size"},
        {"id": 6, "name": "Resolution", "stage": "S1", "check_key": "resolution"},
        {"id": 7, "name": "Blur Detection", "stage": "S1", "check_key": "blur"},
        {"id": 8, "name": "Orientation", "stage": "S1", "check_key": "orientation"},
        {"id": 9, "name": "Face Cover/Mask", "stage": "S1", "check_key": "face_cover"},
        {"id": 10, "name": "NSFW Content", "stage": "S1", "check_key": "nsfw"}
    ]
    
    for check_config in stage1_checks_config:
        check_id = check_config["id"]
        check_name = check_config["name"]
        check_key = check_config["check_key"]
        
        if check_key == "cropping_applied":
            # Check if this is a group photo (multiple faces)
            is_group_photo = "Group photo" in stage1_result["checks"].get("photo_type_validation", "")

            if photo_type == "SECONDARY" and is_group_photo:
                # Group photos are not auto-cropped
                checklist["checks"].append({
                    "id": check_id,
                    "name": check_name,
                    "stage": "S1",
                    "status": "SKIPPED",
                    "reason": "Group photos are not auto-cropped",
                    "details": None
                })
                checklist["skipped"] += 1
            else:
                # PRIMARY photos and SECONDARY single-person photos can be auto-cropped
                cropping_applied = stage1_result["checks"].get("cropping_applied", "NO")
                if cropping_applied == "YES":
                    checklist["checks"].append({
                        "id": check_id,
                        "name": check_name,
                        "stage": "S1",
                        "status": "APPLIED",
                        "reason": "Face coverage was low, image auto-cropped",
                        "details": {
                            "original_coverage": stage1_result["checks"].get("face_coverage_original"),
                            "after_crop": stage1_result["checks"].get("face_coverage_after_crop")
                        }
                    })
                    checklist["passed"] += 1
                else:
                    checklist["checks"].append({
                        "id": check_id,
                        "name": check_name,
                        "stage": "S1",
                        "status": "NOT_NEEDED",
                        "reason": "Face coverage already sufficient",
                        "details": {
                            "coverage": stage1_result["checks"].get("face_coverage_original")
                        }
                    })
                    checklist["passed"] += 1
        else:
            check_value = stage1_result["checks"].get(check_key, "UNKNOWN")
            
            if check_value == "PASS":
                status = "PASS"
                checklist["passed"] += 1
            elif "face(s) detected" in str(check_value):
                status = "PASS"
                checklist["passed"] += 1
            else:
                status = "INFO"
                checklist["passed"] += 1
            
            checklist["checks"].append({
                "id": check_id,
                "name": check_name,
                "stage": "S1",
                "status": status,
                "reason": None,
                "details": check_value
            })
    
    # Stage 2 checks
    if stage2_result:
        stage2_checks_config = [
            {"id": 11, "name": "Age Verification (DeepFace)", "check_key": "age"},
            {"id": 12, "name": "Gender Validation", "check_key": "gender"},
            {"id": 13, "name": "Ethnicity Validation (DeepFace)", "check_key": "ethnicity"},
            {"id": 14, "name": "Face Coverage (InsightFace)", "check_key": "face_coverage"},
            {"id": 15, "name": "Digital Enhancement", "check_key": "enhancement"},
            {"id": 16, "name": "Photo-of-Photo", "check_key": "photo_of_photo"},
            {"id": 17, "name": "AI-Generated", "check_key": "ai_generated"},
            {"id": 18, "name": "PII Detection (OCR)", "check_key": "pii"}
        ]

        performed = stage2_result.get("checks_performed", [])
        skipped = stage2_result.get("checks_skipped", [])

        for check_config in stage2_checks_config:
            check_id = check_config["id"]
            check_name = check_config["name"]
            check_key = check_config["check_key"]

            # Check if this was already performed in Stage 1 (e.g., age for secondary single-person photos)
            stage1_check_result = stage1_result["checks"].get(check_key)
            if stage1_check_result and isinstance(stage1_check_result, dict) and "status" in stage1_check_result:
                # Age was checked in Stage 1 for secondary single-person photos
                status = stage1_check_result.get("status", "UNKNOWN")
                if status == "PASS":
                    checklist["passed"] += 1
                elif status == "FAIL":
                    checklist["failed"] += 1
                elif status == "REVIEW":
                    checklist["review"] += 1

                checklist["checks"].append({
                    "id": check_id,
                    "name": check_name,
                    "stage": "S1",  # Mark as S1 since it was checked there
                    "status": status,
                    "reason": stage1_check_result.get("reason"),
                    "details": {
                        k: v for k, v in stage1_check_result.items()
                        if k not in ["status", "reason"]
                    } if stage1_check_result else None
                })
            elif check_key in skipped:
                # Determine if this is a group photo or single-person secondary
                is_group_photo = "Group photo" in stage1_result["checks"].get("photo_type_validation", "")

                skip_reasons_group = {
                    "age": "SECONDARY group photos skip age check (family members have different ages)",
                    "gender": "SECONDARY group photos skip gender check (family has both genders)",
                    "ethnicity": "SECONDARY group photos skip ethnicity check (family members may differ)",
                    "face_coverage": "SECONDARY group photos skip face coverage (group photos have smaller faces)"
                }
                skip_reasons_single = {
                    "age": "Age check not performed",
                    "gender": "Gender check skipped for SECONDARY photos",
                    "ethnicity": "Ethnicity check skipped for SECONDARY photos",
                    "face_coverage": "Face coverage check skipped for SECONDARY photos"
                }

                skip_reasons = skip_reasons_group if is_group_photo else skip_reasons_single

                checklist["checks"].append({
                    "id": check_id,
                    "name": check_name,
                    "stage": "S2",
                    "status": "SKIPPED",
                    "reason": skip_reasons.get(check_key, f"Skipped for {photo_type} photos"),
                    "details": None
                })
                checklist["skipped"] += 1

            elif check_key in performed:
                check_result = stage2_result["checks"].get(check_key, {})
                
                if isinstance(check_result, dict) and "status" not in check_result:
                    all_statuses = []
                    details = {}
                    
                    for sub_key, sub_result in check_result.items():
                        if isinstance(sub_result, dict) and "status" in sub_result:
                            all_statuses.append(sub_result["status"])
                            details[sub_key] = {
                                "status": sub_result["status"],
                                "reason": sub_result.get("reason")
                            }
                    
                    if "FAIL" in all_statuses:
                        status = "FAIL"
                        checklist["failed"] += 1
                    elif "REVIEW" in all_statuses:
                        status = "REVIEW"
                        checklist["review"] += 1
                    else:
                        status = "PASS"
                        checklist["passed"] += 1
                    
                    checklist["checks"].append({
                        "id": check_id,
                        "name": check_name,
                        "stage": "S2",
                        "status": status,
                        "reason": check_result.get("reason"),
                        "details": details
                    })
                else:
                    status = check_result.get("status", "UNKNOWN")
                    
                    if status == "PASS":
                        checklist["passed"] += 1
                    elif status == "FAIL":
                        checklist["failed"] += 1
                    elif status == "REVIEW":
                        checklist["review"] += 1
                    
                    checklist["checks"].append({
                        "id": check_id,
                        "name": check_name,
                        "stage": "S2",
                        "status": status,
                        "reason": check_result.get("reason"),
                        "details": {
                            k: v for k, v in check_result.items() 
                            if k not in ["status", "reason"]
                        } if check_result else None
                    })
    
    checklist["total_checks"] = len(checklist["checks"])
    
    return checklist


# ==================== COMBINED VALIDATION PIPELINE (HYBRID) ====================

def validate_photo_complete_hybrid(
    image_path: str,
    photo_type: str = "PRIMARY",
    profile_data: Dict = None,
    reference_photo_path: Optional[str] = None,
    run_stage2: bool = True,
    use_deepface_gender: bool = False
) -> Dict:
    """
    Complete photo validation pipeline with HYBRID approach:
    - InsightFace as backbone
    - DeepFace for age and ethnicity (PRIMARY only)

    Note: Fraud database, celebrity database, and duplicate detection checks
    have been removed as they require database support not available in this phase.
    """
    
    print("\n" + "="*70)
    print("STARTING HYBRID PHOTO VALIDATION PIPELINE (GPU-ACCELERATED)")
    print("InsightFace (Backbone) + DeepFace (Age/Ethnicity)")
    print(f"GPU Available: {GPU_AVAILABLE} (TF: {TF_GPU_AVAILABLE}, ONNX: {ONNX_GPU_AVAILABLE})")
    print("="*70)
    
    results = {
        "image_path": image_path,
        "photo_type": photo_type,
        "stage1": None,
        "stage2": None,
        "final_decision": None,
        "final_action": None,
        "final_reason": None
    }
    
    # ============= STAGE 1 VALIDATION =============
    print(f"\n[STAGE 1] Running basic quality checks for {photo_type} photo...")
    stage1_result = stage1_validate(image_path, photo_type, reference_photo_path, profile_data)
    results["stage1"] = stage1_result
    
    if stage1_result["result"] == "REJECT":
        print(f"[STAGE 1] ❌ REJECTED: {stage1_result['reason']}")
        results["final_decision"] = "REJECT"
        results["final_action"] = "REJECT_PHOTO"
        results["final_reason"] = f"Stage 1 failure: {stage1_result['reason']}"
        return results
    
    print("[STAGE 1] ✅ PASSED")
    
    # Handle cropped image
    validation_image_path = image_path
    cropped_image_array = None
    
    if stage1_result.get("cropped_image") is not None:
        cropped_image_array = stage1_result["cropped_image"]
        cropped_path = image_path.replace(".", "_cropped_final.")
        cv2.imwrite(cropped_path, cropped_image_array)
        print(f"[STAGE 1] Cropped image saved: {cropped_path}")
        results["cropped_image_path"] = cropped_path
        results["image_was_cropped"] = True
        validation_image_path = cropped_path
    else:
        results["image_was_cropped"] = False
    
    # ============= STAGE 2 VALIDATION (HYBRID) =============
    if run_stage2:
        if profile_data is None:
            print("[STAGE 2]   Skipping - No profile data provided")
            results["final_decision"] = "PASS_STAGE1_ONLY"
            results["final_action"] = "MANUAL_REVIEW"
            results["final_reason"] = "Stage 1 passed, Stage 2 skipped (no profile data)"
            return results
        
        print("\n[STAGE 2 GPU] Running HYBRID validation with GPU acceleration...")
        print(f"[STAGE 2 GPU] InsightFace: detection, embeddings, matching (GPU: {ONNX_GPU_AVAILABLE})")
        if photo_type == "PRIMARY":
            print(f"[STAGE 2 GPU] DeepFace: age, ethnicity (GPU: {TF_GPU_AVAILABLE})")
        print(f"[STAGE 2 GPU] Validating image: {validation_image_path}")
        
        stage2_result = stage2_validate_hybrid(
            image_path=validation_image_path,
            profile_data=profile_data,
            photo_type=photo_type,
            use_deepface_gender=use_deepface_gender
        )
        results["stage2"] = stage2_result
        
        results["final_decision"] = stage2_result["final_decision"]
        results["final_action"] = stage2_result["action"]
        results["final_reason"] = stage2_result["reason"]
        
        # Compile checklist
        results["checklist_summary"] = compile_checklist_summary(
            stage1_result, 
            stage2_result, 
            photo_type
        )
        
        # Print library usage summary
        print(f"\n[STAGE 2 GPU] Library Usage Summary:")
        print(f"  InsightFace: {', '.join(stage2_result['library_usage']['insightface'])}")
        if stage2_result['library_usage']['deepface']:
            print(f"  DeepFace: {', '.join(stage2_result['library_usage']['deepface'])}")
        print(f"  GPU Accelerated: {stage2_result['gpu_used']}")

        if stage2_result["final_decision"] == "SUSPEND":
            print(f"[STAGE 2 GPU] SUSPEND: {stage2_result['reason']}")
        elif stage2_result["final_decision"] == "REJECT":
            print(f"[STAGE 2 GPU] REJECT: {stage2_result['reason']}")
        elif stage2_result["final_decision"] == "MANUAL_REVIEW":
            print(f"[STAGE 2 GPU] MANUAL REVIEW: {stage2_result['reason']}")
        else:
            print(f"[STAGE 2 GPU] APPROVED: {stage2_result['reason']}")

            # Convert cropped image to base64 for both PRIMARY and SECONDARY photos
            # Only when validation is APPROVED
            if results.get("image_was_cropped") and cropped_image_array is not None:
                print(f"[STAGE 2 GPU] Converting cropped image to base64 for {photo_type} photo...")
                #cropped_base64 = image_to_base64(cropped_image_array)
                #if cropped_base64:
                #    results["cropped_image_base64"] = cropped_base64
                #    print(f"[STAGE 2 GPU] Cropped image base64 ready for {photo_type} photo")
    else:
        results["final_decision"] = "PASS_STAGE1_ONLY"
        results["final_action"] = "PUBLISH"
        results["final_reason"] = "Stage 1 passed, Stage 2 not requested"
        
        results["checklist_summary"] = compile_checklist_summary(
            stage1_result, 
            None,
            photo_type
        )
    
    return results


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("EXAMPLE: HYBRID PHOTO VALIDATION")
    print("="*70)
    
    profile_data = {
        "matri_id": "BM123456",
        "gender": "Male",
        "age": 28
    }
    
    result = validate_photo_complete_hybrid(
        image_path="test_image.jpg",
        photo_type="PRIMARY",
        profile_data=profile_data,
        run_stage2=True,
        use_deepface_gender=False  # Use InsightFace for gender (faster)
    )
    
    print(f"\nFinal Decision: {result['final_decision']}")
    print(f"Final Action: {result['final_action']}")
    print(f"Reason: {result['final_reason']}")
    
    if result.get('checklist_summary'):
        checklist = result['checklist_summary']
        print(f"\n{'='*70}")
        print("CHECKLIST SUMMARY")
        print(f"{'='*70}")
        print(f"Total: {checklist['total_checks']}, Passed: {checklist['passed']}, Failed: {checklist['failed']}")
        print(f"Skipped: {checklist['skipped']}, Review: {checklist['review']}")
    
    if result.get('stage2'):
        print(f"\n{'='*70}")
        print("LIBRARY USAGE")
        print(f"{'='*70}")
        print(f"InsightFace: {', '.join(result['stage2']['library_usage']['insightface'])}")
        if result['stage2']['library_usage']['deepface']:
            print(f"DeepFace: {', '.join(result['stage2']['library_usage']['deepface'])}")
