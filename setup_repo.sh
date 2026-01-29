#!/bin/bash

################################################################################
# FINAL GPU Setup Script for CUDA 12.4
# NVIDIA L40S | Driver 550.163.01 | Python 3.10.13
# Full GPU Acceleration: InsightFace + DeepFace + NudeNet
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║         FINAL GPU Setup for CUDA 12.4 + NudeNet GPU                 ║
║         NVIDIA L40S | Python 3.10.13                                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${GREEN}System Configuration:${NC}"
echo "  GPU: NVIDIA L40S (46GB VRAM)"
echo "  Driver: 550.163.01"
echo "  CUDA: 12.4"
echo "  Python: 3.10.13"
echo ""

echo -e "${YELLOW}This will install:${NC}"
echo "  ✓ TensorFlow 2.20.0 (GPU) - DeepFace"
echo "  ✓ ONNX Runtime GPU 1.20.1 - InsightFace & NudeNet"
echo "  ✓ InsightFace 0.7.3 (GPU)"
echo "  ✓ DeepFace 0.0.93 (GPU)"
echo "  ✓ NudeNet 3.4.2 (GPU) - NSFW Detection"
echo "  ✓ All dependencies"
echo ""
echo -e "${YELLOW}Press ENTER to continue or Ctrl+C to cancel${NC}"
read

# ==================== VERIFY SYSTEM ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 1: System Verification${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ nvidia-smi not found - GPU drivers not installed${NC}"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)
echo -e "${GREEN}✓${NC} GPU: $GPU_INFO"

# Check CUDA
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo -e "${GREEN}✓${NC} CUDA: $CUDA_VERSION"

if [[ ! "$CUDA_VERSION" =~ ^12\. ]]; then
    echo -e "${YELLOW}⚠${NC} Warning: Expected CUDA 12.x, found $CUDA_VERSION"
fi

# Check Python
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Python: $PYTHON_VERSION"

# ==================== BACKUP ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 2: Creating Backup${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
pip freeze > "$BACKUP_DIR/packages_before.txt" 2>/dev/null || true
echo -e "${GREEN}✓${NC} Backup: $BACKUP_DIR/"

# ==================== CLEAN ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 3: Removing Old Packages${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

pip uninstall -y tensorflow tensorflow-gpu tf-keras 2>/dev/null || true
pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
pip uninstall -y insightface deepface retina-face nudenet 2>/dev/null || true
pip cache purge 2>/dev/null || true
echo -e "${GREEN}✓${NC} Cleanup complete"

# ==================== INSTALL GPU LIBRARIES ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 4: Installing GPU Libraries (CUDA 12.4)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

echo -e "${YELLOW}Installing TensorFlow 2.20.0 (GPU)...${NC}"
pip install --break-system-packages --no-cache-dir tensorflow==2.20.0
echo -e "${GREEN}✓${NC} TensorFlow installed"

echo -e "${YELLOW}Installing ONNX Runtime GPU 1.20.1 (CUDA 12.x)...${NC}"
pip install --break-system-packages --no-cache-dir onnxruntime-gpu==1.20.1
echo -e "${GREEN}✓${NC} ONNX Runtime GPU installed"

# Quick verification
echo ""
echo "Verifying GPU libraries..."
python3 << 'PYEOF'
import sys
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print(f"  ✓ TensorFlow GPU: {len(gpus)} device(s)")
    if len(gpus) == 0:
        sys.exit(1)
except Exception as e:
    print(f"  ✗ TensorFlow: {e}")
    sys.exit(1)

try:
    import onnxruntime as ort
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print(f"  ✓ ONNX Runtime: CUDA available")
    else:
        print(f"  ✗ ONNX Runtime: CUDA not available")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ ONNX Runtime: {e}")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo -e "${RED}GPU verification failed. Check CUDA installation.${NC}"
    exit 1
fi

# ==================== INSTALL FACE LIBRARIES ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 5: Installing Face Recognition Libraries${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

echo -e "${YELLOW}Installing InsightFace...${NC}"
pip install --break-system-packages --no-cache-dir insightface==0.7.3
echo -e "${GREEN}✓${NC} InsightFace installed"

echo -e "${YELLOW}Installing DeepFace...${NC}"
pip install --break-system-packages --no-cache-dir deepface==0.0.93
echo -e "${GREEN}✓${NC} DeepFace installed"

echo -e "${YELLOW}Installing RetinaFace...${NC}"
pip install --break-system-packages --no-cache-dir retina-face==0.0.17
echo -e "${GREEN}✓${NC} RetinaFace installed"

# ==================== INSTALL NUDENET (GPU) ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 6: Installing NudeNet with GPU Support${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

echo -e "${YELLOW}Installing ONNX (for NudeNet GPU)...${NC}"
pip install --break-system-packages --no-cache-dir onnx==1.17.0
echo -e "${GREEN}✓${NC} ONNX installed"

echo -e "${YELLOW}Installing NudeNet (will use ONNX Runtime GPU)...${NC}"
pip install --break-system-packages --no-cache-dir nudenet==3.4.2
echo -e "${GREEN}✓${NC} NudeNet installed"

# ==================== INSTALL OTHER DEPENDENCIES ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 7: Installing Other Dependencies${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

pip install --break-system-packages --no-cache-dir \
    opencv-python==4.10.0.84 \
    opencv-contrib-python==4.10.0.84 \
    numpy==1.26.4 \
    scipy==1.14.1 \
    Pillow==11.0.0 \
    scikit-image==0.24.0 \
    fastapi==0.115.5 \
    uvicorn==0.32.1 \
    python-multipart==0.0.17 \
    pydantic==2.10.3 \
    requests==2.32.3 \
    tqdm==4.67.1

echo -e "${GREEN}✓${NC} All dependencies installed"

pip freeze > "$BACKUP_DIR/packages_after.txt"

# ==================== CONFIGURE ENVIRONMENT ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 8: Configuring GPU Environment${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Detect CUDA path
if [ -d "/usr/local/cuda-12.4" ]; then
    CUDA_PATH="/usr/local/cuda-12.4"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_PATH="/usr/local/cuda"
else
    CUDA_PATH="/usr/local/cuda-12.4"
fi

cat > gpu_env_config.sh << EOF
#!/bin/bash
# GPU Environment for CUDA 12.4

export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2
export TF_CPP_MIN_LOG_LEVEL=2

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:\$LD_LIBRARY_PATH
export PATH=${CUDA_PATH}/bin:\$PATH

export ORT_TENSORRT_FP16_ENABLE=1

echo "✓ GPU Environment configured (CUDA 12.4)"
EOF

chmod +x gpu_env_config.sh
echo -e "${GREEN}✓${NC} gpu_env_config.sh created"

cat > start_gpu_api.sh << 'EOF'
#!/bin/bash
source gpu_env_config.sh
echo "Starting API with full GPU acceleration..."
python hybrid_api.py
EOF

chmod +x start_gpu_api.sh
echo -e "${GREEN}✓${NC} start_gpu_api.sh created"

# ==================== RUN DIAGNOSTICS ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 9: Running Final Tests${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

source gpu_env_config.sh

python3 << 'PYEOF'
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*60)
print("FINAL GPU VERIFICATION")
print("="*60)

# TensorFlow
print("\n1. TensorFlow GPU:")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        details = tf.config.experimental.get_device_details(gpus[0])
        print(f"   ✓ Device: {details.get('device_name', 'Unknown')}")
        print(f"   ✓ Compute Capability: {details.get('compute_capability', 'Unknown')}")
    else:
        print("   ✗ No GPUs detected")
except Exception as e:
    print(f"   ✗ Error: {e}")

# ONNX Runtime
print("\n2. ONNX Runtime GPU:")
try:
    import onnxruntime as ort
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print(f"   ✓ CUDA provider available")
    else:
        print(f"   ✗ CUDA provider not available")
except Exception as e:
    print(f"   ✗ Error: {e}")

# InsightFace
print("\n3. InsightFace:")
try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print(f"   ✓ Initialized with GPU")
except Exception as e:
    print(f"   ✗ Error: {e}")

# DeepFace
print("\n4. DeepFace:")
try:
    from deepface import DeepFace
    if gpus:
        print(f"   ✓ Will use GPU")
    else:
        print(f"   ⚠ Will use CPU")
except Exception as e:
    print(f"   ✗ Error: {e}")

# NudeNet
print("\n5. NudeNet (NSFW Detection):")
try:
    from nudenet import NudeDetector
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print(f"   ✓ Will use GPU (10x faster)")
    else:
        print(f"   ⚠ Will use CPU (slow)")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*60)
print("COMPLETE")
print("="*60 + "\n")
PYEOF

# ==================== SUMMARY ====================

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}INSTALLATION COMPLETE!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"

echo ""
echo -e "${BLUE}Installed:${NC}"
echo "  ✓ TensorFlow 2.20.0 (GPU)"
echo "  ✓ ONNX Runtime GPU 1.20.1"
echo "  ✓ InsightFace 0.7.3 (GPU)"
echo "  ✓ DeepFace 0.0.93 (GPU)"
echo "  ✓ NudeNet 3.4.2 (GPU)"

echo ""
echo -e "${YELLOW}NEXT STEPS:${NC}"
echo ""
echo "1. Activate GPU environment:"
echo -e "   ${GREEN}source gpu_env_config.sh${NC}"
echo ""
echo "2. Start API:"
echo -e "   ${GREEN}./start_gpu_api.sh${NC}"
echo ""
echo "3. Monitor GPU:"
echo -e "   ${GREEN}watch -n 1 nvidia-smi${NC}"
echo ""
echo "4. Test API:"
echo -e "   ${GREEN}curl http://localhost:8000/health${NC}"

echo ""
echo -e "${GREEN}Ready to go!${NC}"
echo ""
