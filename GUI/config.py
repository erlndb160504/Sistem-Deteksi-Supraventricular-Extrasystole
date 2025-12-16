# config.py
"""
Configuration module for SVE Detection System
Contains all constants and settings
"""

DEFAULT_MODEL_PATH = "C://IMPORTANT//KULIAH//SKRIPSI//final_ithink//unet_svdb_FINAL (1).h5"

# =====================================================
# SIGNAL PROCESSING CONFIGURATION
# =====================================================
FS = 128  # 
WINDOW_SIZE = 512  
OVERLAP = 256  
STRIDE = WINDOW_SIZE - OVERLAP
PROB_THRESHOLD = 0.5  
SVE_BURDEN_THRESHOLD = 5.0  

LOW_FREQ = 0.5       # Bandpass lower cutoff (Hz)
HIGH_FREQ = 50       # Bandpass upper cutoff (Hz)
BP_ORDER = 4         # Bandpass filter order

# =====================================================
# BEAT-BASED EVALUATION CONFIG
# =====================================================
PRE_R_MS = 200      # ms sebelum R-peak
POST_R_MS = 400     # ms sesudah R-peak

PRE_R_SAMPLES = int(PRE_R_MS / 1000 * FS)
POST_R_SAMPLES = int(POST_R_MS / 1000 * FS)

BEAT_DECISION_THRESHOLD = 0.3


# =====================================================
# SHIMMER CONFIGURATION
# =====================================================
V_REF = 2.42
GAIN = 6
SENS_MV = V_REF / (GAIN * (2**23 - 1)) * 1000
DEFAULT_BAUDRATE = 115200

# =====================================================
# RECORDING DURATIONS
# =====================================================
RECORDING_DURATIONS = {
    "1 minute": 60,
    "5 minutes": 300,
    "10 minutes": 600,
    "15 minutes": 900,
    "30 minutes": 1800,
}

# =====================================================
# UI CONFIGURATION
# =====================================================
COLORS = {
    'primary': '#6C63FF',
    'success': '#6BCB77',
    'danger': '#FF6B9D',
    'warning': '#FFD93D',
    'info': '#4D96FF',
    'dark': '#2C3E50',
    'light': '#ECF0F1',
    'white': '#FFFFFF',
    'bg': '#f5f7fa'
}

SEGMENT_WIDTH = 30  # 30 seconds per segment

# =====================================================
# PROCESSING CONFIGURATION
# =====================================================
BATCH_SIZE = 32  # Batch size for model inference
PROCESS_INTERVAL = 5  # seconds between buffer processing