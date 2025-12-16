# # main.py
# """
# SVE Detection System - Main Application
# UNet-based Segmentation for SVE Detection

# Modular architecture with separated concerns:
# - config.py: Configuration constants
# - utils.py: Utility functions
# - preprocessor.py: ECG preprocessing
# - segmentation.py: UNet segmentation
# - shimmer_manager.py: Device management
# - widgets.py: Custom UI widgets
# - exporter.py: Data export
# - ui_components.py: UI layout
# - processing.py: Signal processing
# - visualization.py: Plot visualization
# """
# import tkinter as tk
# from tkinter import filedialog, messagebox
# import numpy as np
# import tensorflow as tf
# import threading
# import time
# import os
# import atexit

# # Import custom modules
# from config import *
# from utils import setup_gpu, check_libraries, format_time
# from preprocessor import ECGPreprocessor
# from segmentation import UNetSegmentation
# from shimmer_manager import ShimmerManager
# from wfdb_evaluator import WFDBEvaluator
# #from exporter import ResultExporter
# from ui_components import UIComponents
# from processing import SignalProcessor
# from visualization import Visualizer    

# # Check available libraries
# LIBRARIES = check_libraries()
# WFDB_AVAILABLE = LIBRARIES['wfdb']
# SHIMMER_AVAILABLE = LIBRARIES['shimmer']

# if WFDB_AVAILABLE:
#     import wfdb

# # Setup GPU
# setup_gpu()


# class SVEDetectionGUI:
#     """Main GUI Application - UNet Segmentation"""
    
#     def __init__(self, root):
#         self.root = root
#         self.root.title("SVE Detection System - UNet Segmentation")
        
#         # Window setup
#         self._setup_window()
        
#         # State variables
#         self._init_state_variables()
        
#         # Data storage
#         self._init_data_storage()
        
#         # Initialize managers FIRST
#         self.shimmer = ShimmerManager()
#         self.preprocessor = ECGPreprocessor()

#         # Initialize evaluator
#         self.evaluator = WFDBEvaluator()
#         self.ground_truth_loaded = False

#         #self.exporter = ResultExporter()
#         self.model = None
#         self.segmentation = None
#         self.model_loaded = False
        
#         # Visualization
#         self.segment_width = SEGMENT_WIDTH
#         self.current_segment = 0
#         self.total_segments = 0
        
#         # Threading
#         self.results_lock = threading.Lock()
#         self.buffer_lock = threading.Lock()
#         self.processing_thread = None
#         self.stop_processing = False
        
#         # Initialize helper classes BEFORE setup_ui()
#         self.processor = SignalProcessor(self)
#         self.visualizer = Visualizer(self)

#         # Initialize trimmer
#         self.init_trimmer()
#         self.ui_components = None
        
#         # Setup
#         atexit.register(self.cleanup)
#         self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
#         # NOW setup UI
#         self.setup_ui()
#         self.update_loop()
    
#     def _setup_window(self):
#         """Setup main window"""
#         screen_width = self.root.winfo_screenwidth()
#         screen_height = self.root.winfo_screenheight()
#         window_width = max(min(int(screen_width * 0.95), 1600), 1000)
#         window_height = max(min(int(screen_height * 0.95), 900), 700)
#         x = (screen_width - window_width) // 2
#         y = (screen_height - window_height) // 2
#         self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
#         self.root.configure(bg=COLORS['bg'])
#         self.scale_factor = window_width / 1400
#         self.colors = COLORS
    
#     def _init_state_variables(self):
#         """Initialize state variables - FIXED VERSION"""
#         self.is_recording = False
#         self.is_processing = False
#         self.source_type = None
#         self.recording_start_time = 0
#         self.processing_start_time = 0
#         self.target_duration = 0
#         self.processing_time = 0.0
#         self.inference_start_time = None
#         self.last_plot_update = 0
#         self.analyze_btn_enabled = False
#         self.start_record_enabled = False
#         self.stop_record_enabled = False
        
#         # ===== PENTING! Initialize recording tracking variables =====
#         self.recording_start_samples = 0  # TAMBAH INI!
#         self.actual_recorded_samples = 0  # TAMBAH INI!
#         self.actual_recording_duration = 0.0  # TAMBAH INI!
    
#     def _init_data_storage(self):
#         """Initialize data storage"""
#         self.raw_signal = np.array([], dtype=np.float32)
#         self.preprocessed_signal = None
#         self.signal_before_norm = None
#         self.probability_mask = None
#         self.binary_mask = None
#         self.sve_burden = 0.0
#         #self.total_episodes = 0
#         self.classification = "Waiting..."
#         self.mean_amplitude = 0.0
#         self.sve_samples = 0
#         self.total_samples = 0
        
#         self.buffer_results = {
#             'all_masks': [],
#             'all_probs': [],
#             'processed_samples': 0,
#             'last_segment_end': 0
#         }
    
#     def scale(self, value):
#         """Scale value based on window size"""
#         return int(value * self.scale_factor)
    
#     # =====================================================
#     # UI SETUP
#     # =====================================================
#     def setup_ui(self):
#         """Setup entire UI"""
#         self.ui_components = UIComponents(self)
#         self.ui_components.setup_header()
#         self.ui_components.setup_main_container()
    
#     # # =====================================================
#     # # MODEL MANAGEMENT
#     # # =====================================================
#     # def load_model(self):
#     #     """Load UNet model"""
#     #     try:
#     #         file_path = filedialog.askopenfilename(
#     #             title="Select UNet Model File",
#     #             filetypes=[("HDF5 files", "*.h5"), 
#     #                       ("Keras files", "*.keras"), 
#     #                       ("All files", "*.*")]
#     #         )
            
#     #         if not file_path:
#     #             return
            
#     #         self.model = tf.keras.models.load_model(file_path, compile=False)
#     #         self.segmentation = UNetSegmentation(self.model)
#     #         self.model_loaded = True
            
#     #         filename = os.path.basename(file_path)
#     #         self.model_status.config(text=f"{filename}", fg=self.colors['success'])
#     #         self.model_indicator.config(text="Model: Ready", fg=self.colors['success'])
#     #         messagebox.showinfo("Success", "UNet model loaded successfully!")
            
#     #     except Exception as e:
#     #         self.model_loaded = False
#     #         self.model_status.config(text="Failed to load", fg=self.colors['danger'])
#     #         messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
    
#     # =====================================================
#     # MODEL MANAGEMENT - AUTO LOAD
#     # =====================================================
# # main.py
# """
# SVE Detection System - Main Application
# UNet-based Segmentation for SVE Detection
# """
# import tkinter as tk
# from tkinter import filedialog, messagebox
# import numpy as np
# import tensorflow as tf
# import threading
# import time
# import os
# import atexit

# # Import custom modules
# from config import *
# from utils import setup_gpu, check_libraries, format_time
# from preprocessor import ECGPreprocessor
# from segmentation import UNetSegmentation
# from shimmer_manager import ShimmerManager
# from wfdb_evaluator import WFDBEvaluator
# #from exporter import ResultExporter
# from ui_components import UIComponents
# from processing import SignalProcessor
# from visualization import Visualizer    

# # Check available libraries
# LIBRARIES = check_libraries()
# WFDB_AVAILABLE = LIBRARIES['wfdb']
# SHIMMER_AVAILABLE = LIBRARIES['shimmer']

# if WFDB_AVAILABLE:
#     import wfdb

# # Setup GPU
# setup_gpu()


# class SVEDetectionGUI:
#     """Main GUI Application - UNet Segmentation"""
    
#     def __init__(self, root):
#         self.root = root
#         self.root.title("SVE Detection System - UNet Segmentation")
        
#         # Window setup
#         self._setup_window()
        
#         # State variables
#         self._init_state_variables()
        
#         # Data storage
#         self._init_data_storage()
        
#         # Initialize managers FIRST
#         self.shimmer = ShimmerManager()
#         self.preprocessor = ECGPreprocessor()

#         # Initialize evaluator
#         self.evaluator = WFDBEvaluator()
#         self.ground_truth_loaded = False

#         #self.exporter = ResultExporter()
#         self.model = None
#         self.segmentation = None
#         self.model_loaded = False
        
#         # Visualization
#         self.segment_width = SEGMENT_WIDTH
#         self.current_segment = 0
#         self.total_segments = 0
        
#         # Threading
#         self.results_lock = threading.Lock()
#         self.buffer_lock = threading.Lock()
#         self.processing_thread = None
#         self.stop_processing = False
        
#         # Initialize helper classes BEFORE setup_ui()
#         self.processor = SignalProcessor(self)
#         self.visualizer = Visualizer(self)
#         self.ui_components = None
        
#         # Setup
#         atexit.register(self.cleanup)
#         self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
#         # NOW setup UI
#         self.setup_ui()
        
#         # AUTO LOAD MODEL
#         self.root.after(500, self.auto_load_model)
        
#         self.update_loop()
    
#     def _setup_window(self):
#         """Setup main window"""
#         screen_width = self.root.winfo_screenwidth()
#         screen_height = self.root.winfo_screenheight()
#         window_width = max(min(int(screen_width * 0.95), 1600), 1000)
#         window_height = max(min(int(screen_height * 0.95), 900), 700)
#         x = (screen_width - window_width) // 2
#         y = (screen_height - window_height) // 2
#         self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
#         self.root.configure(bg=COLORS['bg'])
#         self.scale_factor = window_width / 1400
#         self.colors = COLORS
    
#     def _init_state_variables(self):
#         """Initialize state variables"""
#         self.is_recording = False
#         self.is_processing = False
#         self.source_type = None
#         self.recording_start_time = 0
#         self.processing_start_time = 0
#         self.target_duration = 0
#         self.processing_time = 0.0
#         self.inference_start_time = None
#         self.last_plot_update = 0
#         self.analyze_btn_enabled = False
#         self.start_record_enabled = False
#         self.stop_record_enabled = False
        
#         # Recording tracking variables
#         self.recording_start_samples = 0
#         self.actual_recorded_samples = 0
#         self.actual_recording_duration = 0.0
    
#     def _init_data_storage(self):
#         """Initialize data storage"""
#         self.raw_signal = np.array([], dtype=np.float32)
#         self.preprocessed_signal = None
#         self.signal_before_norm = None
#         self.probability_mask = None
#         self.binary_mask = None
#         self.sve_burden = 0.0
#         #self.total_episodes = 0
#         self.classification = "Waiting..."
#         self.mean_amplitude = 0.0
#         self.sve_samples = 0
#         self.total_samples = 0
        
#         self.buffer_results = {
#             'all_masks': [],
#             'all_probs': [],
#             'processed_samples': 0,
#             'last_segment_end': 0
#         }
    
#     def scale(self, value):
#         """Scale value based on window size"""
#         return int(value * self.scale_factor)
    
#     # =====================================================
#     # UI SETUP
#     # =====================================================
#     def setup_ui(self):
#         """Setup entire UI"""
#         self.ui_components = UIComponents(self)
#         self.ui_components.setup_header()
#         self.ui_components.setup_main_container()
    
#     # =====================================================
#     # MODEL MANAGEMENT - AUTO LOAD
#     # =====================================================
#     def auto_load_model(self):
#         """Auto load model from default path"""
#         def load_thread():
#             try:
#                 # Update UI - loading state
#                 self.root.after(0, lambda: self.model_status.config(
#                     text="⏳ Loading model...",
#                     fg="#F57C00"
#                 ))
#                 self.root.after(0, lambda: self.model_indicator.config(
#                     text="● Model: Loading...",
#                     fg="#F57C00"
#                 ))
                
#                 # Check if file exists
#                 if not os.path.exists(DEFAULT_MODEL_PATH):
#                     print(f"[AUTO-LOAD] ❌ Model file not found: {DEFAULT_MODEL_PATH}")
#                     self.root.after(0, lambda: self._on_model_not_found())
#                     return
                
#                 print(f"[AUTO-LOAD] Loading model from {DEFAULT_MODEL_PATH}...")
                
#                 # Load model
#                 model = tf.keras.models.load_model(DEFAULT_MODEL_PATH, compile=False)
                
#                 print(f"[AUTO-LOAD] ✓ Model loaded successfully!")
                
#                 # Update app state
#                 self.root.after(0, lambda m=model: self._on_model_loaded(m))
                
#             except Exception as e:
#                 print(f"[AUTO-LOAD ERROR] {str(e)}")
#                 import traceback
#                 traceback.print_exc()
#                 self.root.after(0, lambda err=str(e): self._on_model_load_error(err))
        
#         # Run in background thread
#         threading.Thread(target=load_thread, daemon=True).start()
    
#     def _on_model_not_found(self):
#         """Callback: Model file not found"""
#         self.model_loaded = False
#         self.model_status.config(
#             text=f"❌ {DEFAULT_MODEL_PATH} not found",
#             fg=self.colors['danger']
#         )
#         self.model_indicator.config(
#             text="● Model: Not Found",
#             fg=self.colors['danger']
#         )
#         messagebox.showwarning(
#             "Model Not Found",
#             f"⚠️ Default model '{DEFAULT_MODEL_PATH}' not found.\n\n"
#             f"Please place your model file in the same directory as main.py\n"
#             f"and name it '{DEFAULT_MODEL_PATH}'\n\n"
#             f"Supported formats: .h5, .keras"
#         )
    
#     def _on_model_loaded(self, model):
#         """Callback: Model loaded successfully"""
#         self.model = model
#         self.segmentation = UNetSegmentation(self.model)
#         self.model_loaded = True
        
#         filename = os.path.basename(DEFAULT_MODEL_PATH)
#         self.model_status.config(
#             text=f"✓ {filename}",
#             fg=self.colors['success']
#         )
#         self.model_indicator.config(
#             text="● Model: Ready",
#             fg=self.colors['success']
#         )
        
#         print(f"[AUTO-LOAD] ✓ Model ready for inference!")
#         messagebox.showinfo(
#             "Model Loaded",
#             f"✓ Model loaded successfully!\n\n"
#             f"File: {filename}\n"
#             f"Status: Ready for analysis\n\n"
#             f"You can now:\n"
#             f"• Load WFDB files and analyze\n"
#             f"• Connect Shimmer and record ECG"
#         )
    
#     def _on_model_load_error(self, error_msg):
#         """Callback: Model load error"""
#         self.model_loaded = False
#         self.model_status.config(
#             text="❌ Failed to load",
#             fg=self.colors['danger']
#         )
#         self.model_indicator.config(
#             text="● Model: Error",
#             fg=self.colors['danger']
#         )
#         messagebox.showerror(
#             "Model Load Error",
#             f"❌ Failed to auto-load model:\n\n"
#             f"{error_msg}\n\n"
#             f"Please check:\n"
#             f"• File '{DEFAULT_MODEL_PATH}' exists\n"
#             f"• File is a valid Keras/TensorFlow model\n"
#             f"• File is not corrupted"
#         )
        
#     # =====================================================
#     # WFDB FILE HANDLING
#     # =====================================================
#     def load_wfdb(self):
#         """Load WFDB file"""
#         try:
#             if not WFDB_AVAILABLE:
#                 messagebox.showerror("Error", 
#                     "WFDB not installed.\nInstall: pip install wfdb")
#                 return
            
#             file_path = filedialog.askopenfilename(
#                 title="Select WFDB Record",
#                 filetypes=[("WFDB files", "*.dat"), ("All files", "*.*")]
#             )
            
#             if not file_path:
#                 return
            
#             record_name = os.path.splitext(file_path)[0]
            
#             def load_thread():
#                 try:
#                     record = wfdb.rdrecord(record_name)
#                     signal = record.p_signal[:, 1].astype(np.float32)
#                     self.root.after(0, lambda: self._on_wfdb_loaded(signal, record_name))
#                 except Exception as e:
#                     self.root.after(0, lambda: self._on_wfdb_load_error(str(e)))
            
#             threading.Thread(target=load_thread, daemon=True).start()
            
#         except Exception as e:
#             messagebox.showerror("Error", f"Failed to load WFDB:\n{str(e)}")
    
#     def _on_wfdb_loaded(self, signal, record_name):
#         """Callback: WFDB loaded"""
#         self.raw_signal = signal
#         self.source_type = 'wfdb'
        
#         filename = os.path.basename(record_name)
#         self.wfdb_status.config(text=f"{filename}", fg=self.colors['success'])
#         self.source_indicator.config(text="Source: WFDB", fg=self.colors['success'])
        
#         # Enable analyze button
#         self.analyze_btn.bg_color = self.colors['warning']
#         self.analyze_btn.fg_color = self.colors['dark']
#         self.analyze_btn.draw_button()
#         self.analyze_btn_enabled = True
        
#         duration = len(signal) / FS
#         messagebox.showinfo("Success", 
#             f"WFDB loaded!\n\n"
#             f"Samples: {len(signal)}\n"
#             f"Duration: {duration:.2f}s\n\n"
#             f"Click 'Analyze WFDB' to process.")
        
#             # ✅ ADD SAFETY CHECK HERE
#         if hasattr(self, 'trimmer') and self.trimmer is not None:
#             self.update_trim_display()
#         # Update trim UI display
#         self.update_trim_display()
    
#     def _on_wfdb_load_error(self, error_msg):
#         """Callback: WFDB load error"""
#         self.wfdb_status.config(text="Failed", fg=self.colors['danger'])
#         messagebox.showerror("Error", f"Failed to load WFDB:\n{error_msg}")
    
#     # =====================================================
#     # DELEGATE METHODS TO PROCESSORS
#     # =====================================================
#     def refresh_com_ports(self):
#         """Refresh COM ports"""
#         self.processor.refresh_com_ports()
    
#     def connect_shimmer(self):
#         threading.Thread(
#             target=self.processor.connect_shimmer,
#             daemon=True
#         ).start()

    
#     def disconnect_shimmer(self):
#         """Disconnect from Shimmer"""
#         self.processor.disconnect_shimmer()
    
#     # def start_recording(self):
#     #     """Start recording"""
#     #     self.processor.start_recording()

#     def start_recording(self):
#         """Start recording (run in background thread, NOT GUI thread)"""
#         threading.Thread(
#             target=self.processor.start_recording,
#             daemon=True
#         ).start()
    
#     def stop_recording(self):
#         threading.Thread(
#             target=self.processor.stop_recording,
#             daemon=True
#         ).start()

    
#     def analyze_signal(self):
#         """Analyze signal"""
#         self.processor.analyze_signal()

#     # ============================================================
# # TAMBAHKAN METHOD BARU DI CLASS SVEDetectionGUI (setelah analyze_signal)
# # ============================================================
#     def load_ground_truth(self):
#         """Load WFDB ground truth for evaluation"""
#         try:
#             if not WFDB_AVAILABLE:
#                 messagebox.showerror("Error", "WFDB not installed")
#                 return
            
#             file_path = filedialog.askopenfilename(
#                 title="Select WFDB Ground Truth Record",
#                 filetypes=[("WFDB files", "*.dat"), ("All files", "*.*")]
#             )
            
#             if not file_path:
#                 return
            
#             record_name = os.path.splitext(file_path)[0]
            
#             # Load in background thread
#             def load_thread():
#                 try:
#                     success = self.evaluator.load_ground_truth(record_name)
#                     self.root.after(0, lambda s=success, n=record_name: self._on_gt_loaded(s, n))
#                 except Exception as e:
#                     self.root.after(0, lambda err=str(e): self._on_gt_error(err))
            
#             threading.Thread(target=load_thread, daemon=True).start()
            
#             # Update UI - loading state
#             self.gt_status.config(text="⏳ Loading ground truth...", fg="#F57C00")
            
#         except Exception as e:
#             messagebox.showerror("Error", f"Failed to load ground truth:\n{str(e)}")
    
#     def _on_gt_loaded(self, success, record_name):
#         """Callback: Ground truth loaded"""
#         if success:
#             self.ground_truth_loaded = True
#             filename = os.path.basename(record_name)
            
#             gt_burden = (np.sum(self.evaluator.ground_truth_mask) / 
#                         len(self.evaluator.ground_truth_mask) * 100)
            
#             self.gt_status.config(
#                 text=f"✓ {filename} ({gt_burden:.1f}% SVE)",
#                 fg=self.colors['success']
#             )
            
#             messagebox.showinfo("Ground Truth Loaded", 
#                 f"✓ Ground truth loaded successfully!\n\n"
#                 f"Samples: {len(self.evaluator.ground_truth_mask):,}\n"
#                 f"SVE samples: {np.sum(self.evaluator.ground_truth_mask):,}\n"
#                 f"SVE burden: {gt_burden:.2f}%\n\n"
#                 f"Annotation symbols: S, A, J\n\n"
#                 f"Now you can:\n"
#                 f"1. Analyze a WFDB signal\n"
#                 f"2. Click 'Evaluate Model' to compare")
#         else:
#             self.ground_truth_loaded = False
#             self.gt_status.config(text="⚠️ No annotation found", fg=self.colors['warning'])
#             messagebox.showwarning("No Annotation",
#                 "Record loaded but no annotation (.atr) file found.\n\n"
#                 "You need a WFDB record with .atr annotation file\n"
#                 "containing beat labels (S, A, J for SVE beats).")
    
#     def _on_gt_error(self, error_msg):
#         """Callback: Ground truth load error"""
#         self.ground_truth_loaded = False
#         self.gt_status.config(text="✗ Load failed", fg=self.colors['danger'])
#         messagebox.showerror("Error", f"Failed to load ground truth:\n\n{error_msg}")
    
#     def evaluate_model(self):
#         """Evaluate model predictions against ground truth"""
#         try:
#             if not self.ground_truth_loaded:
#                 messagebox.showwarning("Warning", 
#                     "⚠️ No ground truth loaded!\n\n"
#                     "Please load a WFDB record with annotation first.")
#                 return
            
#             if self.binary_mask is None or len(self.binary_mask) == 0:
#                 messagebox.showwarning("Warning", 
#                     "⚠️ No prediction available!\n\n"
#                     "Please analyze a signal first.")
#                 return
            
#             # Get threshold from UI
#             threshold = self.threshold_var.get()
            
#             print(f"\n{'='*60}")
#             print(f"🎯 STARTING MODEL EVALUATION")
#             print(f"{'='*60}")
#             print(f"Threshold: {threshold}")
#             print(f"Predicted mask length: {len(self.binary_mask)}")
#             print(f"Ground truth mask length: {len(self.evaluator.ground_truth_mask)}")
            
#             # Evaluate
#             metrics = self.evaluator.evaluate(
#                 self.binary_mask,
#                 threshold=threshold
#             )
            
#             # Update UI with results
#             cm = metrics['confusion_matrix']
#             self.eval_tp_label.config(text=f"{cm['TP']:,}")
#             self.eval_tn_label.config(text=f"{cm['TN']:,}")
#             self.eval_fp_label.config(text=f"{cm['FP']:,}")
#             self.eval_fn_label.config(text=f"{cm['FN']:,}")
            
#             m = metrics['metrics']
#             self.eval_acc_label.config(text=f"{m['accuracy']*100:.1f}%")
#             self.eval_prec_label.config(text=f"{m['precision']*100:.1f}%")
#             self.eval_recall_label.config(text=f"{m['recall']*100:.1f}%")
#             self.eval_f1_label.config(text=f"{m['f1_score']*100:.1f}%")
            
#             # Print to console
#             self.evaluator.print_evaluation(metrics)
            
#             # Show summary dialog
#             b = metrics['burden']
#             messagebox.showinfo("Evaluation Complete",
#                 f"📊 Model Evaluation Results\n\n"
#                 f"═══ Performance Metrics ═══\n"
#                 f"Accuracy : {m['accuracy']*100:.2f}%\n"
#                 f"Precision: {m['precision']*100:.2f}%\n"
#                 f"Recall   : {m['recall']*100:.2f}%\n"
#                 f"F1-Score : {m['f1_score']*100:.2f}%\n\n"
#                 f"═══ Confusion Matrix ═══\n"
#                 f"TP: {cm['TP']:>8,} | FP: {cm['FP']:>8,}\n"
#                 f"FN: {cm['FN']:>8,} | TN: {cm['TN']:>8,}\n\n"
#                 f"═══ SVE Burden ═══\n"
#                 f"Predicted : {b['predicted']:.2f}%\n"
#                 f"GT        : {b['ground_truth']:.2f}%\n"
#                 f"Error     : {b['error']:.2f}%")
            
#         except Exception as e:
#             print(f"[EVAL ERROR] {str(e)}")
#             import traceback
#             traceback.print_exc()
#             messagebox.showerror("Error", f"Evaluation failed:\n{str(e)}")
    
#     def prev_segment(self):
#         """Previous segment"""
#         self.visualizer.prev_segment()
    
#     def next_segment(self):
#         """Next segment"""
#         self.visualizer.next_segment()

#     def reset_all(self):
#         """Reset all data and UI"""
#         try:
#             self.is_recording = False
#             self.is_processing = False
#             self.stop_processing = True
#             import time
#             time.sleep(0.3)
            
#             # Reset timing
#             self.inference_start_time = None
#             self.processing_time = 0.0
            
#             with self.results_lock:
#                 self.raw_signal = np.array([], dtype=np.float32)
#                 self.preprocessed_signal = None
#                 self.signal_before_norm = None
#                 self.probability_mask = None
#                 self.binary_mask = None
#                 self.sve_burden = 0.0
#                 self.classification = "Waiting..."
#                 self.mean_amplitude = 0.0
#                 self.sve_samples = 0
#                 self.total_samples = 0
#                 #self.total_episodes = 0
#                 self.current_segment = 0
#                 self.total_segments = 0
#                 self.buffer_results = {
#                     'all_masks': [],
#                     'all_probs': [],
#                     'processed_samples': 0,
#                     'last_segment_end': 0
#                 }
            
#             # Reset UI
#             self.classification_label.config(text="Waiting...", fg=self.colors['dark'])
#             self.burden_label.config(text="0.00%")
#             self.sve_samples_label.config(text="0 / 0")
#             #self.episodes_label.config(text="0")
#             self.duration_label.config(text="0.00")
#             self.mean_label.config(text="0.00")
#             self.sample_count_label.config(text="0")
#             self.timer_label.config(text="0:00 / 0:00")
#             self.proc_time_label.config(text="0.00 sec")
#             #self.export_status.config(text="Ready to export", fg="#95a5a6")
            
#             self.start_record_enabled = self.source_type == 'shimmer'
#             if self.start_record_enabled:
#                 self.start_record_btn.bg_color = self.colors['success']
#                 self.start_record_btn.fg_color = self.colors['white']
#             else:
#                 self.start_record_btn.bg_color = "#dfe6e9"
#                 self.start_record_btn.fg_color = "#636e72"
#             self.start_record_btn.draw_button()
            
#             self.stop_record_enabled = False
#             self.stop_record_btn.bg_color = "#ffcccb"
#             self.stop_record_btn.fg_color = "#999999"
#             self.stop_record_btn.draw_button()
            
#             self.ax1.clear()
#             self.ax2.clear()
#             self.ui_components._setup_plots()
#             self.canvas.draw()
            
#             messagebox.showinfo("Reset", "All data reset successfully!")
#         except Exception as e:
#             messagebox.showerror("Error", f"Failed to reset:\n{str(e)}")


#     def update_loop(self):
#         """Main update loop - FIXED BUFFER READING"""
#         try:
#             # =====================================================
#             # UPDATE RECORDING TIMER & DISPLAY ACTUAL SAMPLES
#             # =====================================================
#             if self.is_recording and self.target_duration > 0:
#                 import time
#                 elapsed = time.time() - self.recording_start_time
                
#                 # Format time display
#                 timer_text = f"{format_time(elapsed)} / {format_time(self.target_duration)}"
#                 self.timer_label.config(text=timer_text)
                
#                 # Color change when close to end
#                 remaining = self.target_duration - elapsed
#                 if remaining < 5:
#                     self.timer_label.config(fg="#B71C1C")  # Dark red
#                 else:
#                     self.timer_label.config(fg="#E65100")  # Orange
                
#                 # ===== FIX: Read buffer dengan tracking start index =====
#                 if self.source_type == 'shimmer' and self.shimmer.is_streaming:
#                     full_buffer = self.shimmer.get_buffer_copy()
                    
#                     if len(full_buffer) > 0:
#                         # PENTING: Hanya ambil samples SETELAH recording start
#                         if hasattr(self, 'recording_start_samples'):
#                             start_idx = self.recording_start_samples
#                             if len(full_buffer) > start_idx:
#                                 # Samples yg benar-benar direkam setelah START
#                                 actual_recorded = len(full_buffer) - start_idx
#                                 self.raw_signal = full_buffer[start_idx:].copy()
#                             else:
#                                 # Buffer belum exceed start index
#                                 self.raw_signal = np.array([], dtype=np.float32)
#                                 actual_recorded = 0
#                         else:
#                             # Fallback jika start index belum set
#                             self.raw_signal = full_buffer.copy()
#                             actual_recorded = len(full_buffer)
                        
#                         # Update sample count YANG BENAR
#                         self.sample_count_label.config(text=str(actual_recorded))
                
#                 # Update live plot every 1 second
#                 import time
#                 if not hasattr(self, 'last_plot_update'):
#                     self.last_plot_update = 0
                
#                 current_time = time.time()
#                 if current_time - self.last_plot_update >= 1.0:
#                     self.visualizer.update_live_plot()
#                     self.last_plot_update = current_time
                
#                 # Auto-stop when target duration reached
#                 if elapsed >= self.target_duration:
#                     self.stop_recording()
        
#         except Exception as e:
#             print(f"[UPDATE-LOOP ERROR] {str(e)}")
        
#         self.root.after(500, self.update_loop)
    
#     # =====================================================
#     # CLEANUP
#     # =====================================================
#     def cleanup(self):
#         """Cleanup resources"""
#         try:
#             self.stop_processing = True
            
#             if self.is_recording:
#                 self.is_recording = False
#                 import time
#                 time.sleep(0.2)
            
#             if self.source_type == 'shimmer':
#                 try:
#                     if self.shimmer.is_streaming:
#                         self.shimmer.stop_streaming()
#                     import time
#                     time.sleep(0.3)
#                     self.shimmer.disconnect()
#                 except:
#                     pass
#         except:
#             pass
    
#     def on_closing(self):
#         """Handle window closing"""
#         try:
#             self.cleanup()
#             self.root.destroy()
#         except:
#             self.root.destroy()

#     # Tambahkan ini ke SVEDetectionGUI class di main.py

#     # =====================================================
#     # SIGNAL TRIMMING
#     # =====================================================
#     def init_trimmer(self):
#         """Initialize signal trimmer"""
#         try:
#             from signal_trimmer import SignalTrimmer
#             self.trimmer = SignalTrimmer()
#             print("[TRIMMER] Initialized successfully")
#         except Exception as e:
#             print(f"[TRIMMER ERROR] Failed to initialize: {e}")
#             self.trimmer = None

#     def apply_signal_trim(self):
#         """Apply trimming to loaded WFDB signal"""
#         try:
#             # ✅ CHECK if trimmer exists
#             if not hasattr(self, 'trimmer') or self.trimmer is None:
#                 messagebox.showerror("Error", "Trimmer not initialized!")
#                 return
            
#             if self.raw_signal is None or len(self.raw_signal) == 0:
#                 messagebox.showwarning("Warning", "Load WFDB file first!")
#                 return
            
#             # Get trim range from UI
#             try:
#                 start_pct = float(self.trim_start_var.get())
#                 end_pct = float(self.trim_end_var.get())
#             except ValueError:
#                 messagebox.showerror("Error", "Invalid start/end percentages")
#                 return
            
#             # Convert percentage to actual seconds
#             total_duration = len(self.raw_signal) / FS
#             start_sec = (start_pct / 100.0) * total_duration
#             end_sec = (end_pct / 100.0) * total_duration
            
#             # Validate
#             if start_sec >= end_sec:
#                 messagebox.showerror("Error", 
#                     f"Start time ({start_sec:.2f}s) must be before end time ({end_sec:.2f}s)")
#                 return
            
#             if end_sec - start_sec < 0.1:
#                 messagebox.showerror("Error", "Trimmed duration must be > 0.1 seconds")
#                 return
            
#             # Apply trim
#             self.trimmer.set_original_signal(self.raw_signal)
#             trimmed_signal = self.trimmer.trim_signal(start_sec, end_sec, in_place=True)
            
#             # Update raw_signal
#             self.raw_signal = trimmed_signal
            
#             # Show confirmation
#             trimmed_duration = len(trimmed_signal) / FS
#             messagebox.showinfo("Trim Complete",
#                 f"✂️ Signal trimmed successfully!\n\n"
#                 f"Original duration: {total_duration:.2f}s\n"
#                 f"Trimmed duration : {trimmed_duration:.2f}s ({len(trimmed_signal):,} samples)\n"
#                 f"Range: {start_sec:.2f}s - {end_sec:.2f}s\n\n"
#                 f"Ready to analyze!")
            
#             # Update UI to reflect trimmed state
#             self.wfdb_status.config(
#                 text=f"✓ Trimmed: {trimmed_duration:.2f}s ({len(trimmed_signal):,} samples)",
#                 fg=self.colors['success']
#             )
            
#             print(f"[TRIM] Applied: {start_sec:.2f}s - {end_sec:.2f}s ({len(trimmed_signal)} samples)")
            
#         except Exception as e:
#             print(f"[TRIM ERROR] {str(e)}")
#             import traceback
#             traceback.print_exc()
#             messagebox.showerror("Error", f"Trim failed:\n{str(e)}")


#     def reset_signal_trim(self):
#         """Reset to original signal"""
#         try:
#             # ✅ CHECK if trimmer exists
#             if not hasattr(self, 'trimmer') or self.trimmer is None:
#                 messagebox.showerror("Error", "Trimmer not initialized!")
#                 return
            
#             if self.raw_signal is None:
#                 messagebox.showwarning("Warning", "No signal loaded")
#                 return
            
#             # Reset trimmer
#             self.trimmer.reset_to_original()
#             if self.trimmer.original_signal is not None:
#                 self.raw_signal = self.trimmer.original_signal.copy()
            
#             # Update UI
#             total_duration = len(self.raw_signal) / FS
#             self.wfdb_status.config(
#                 text=f"✓ Original: {total_duration:.2f}s ({len(self.raw_signal):,} samples)",
#                 fg=self.colors['success']
#             )
            
#             # Reset slider UI
#             self.trim_start_var.set("0")
#             self.trim_end_var.set("100")
            
#             # Update trim display labels
#             self.trim_start_label.config(text="0.00s")
#             self.trim_end_label.config(text=f"{total_duration:.2f}s")
            
#             messagebox.showinfo("Reset Complete",
#                 f"Signal reset to original\n\n"
#                 f"Duration: {total_duration:.2f}s\n"
#                 f"Samples: {len(self.raw_signal):,}")
            
#             print(f"[TRIM] Reset to original signal ({len(self.raw_signal)} samples)")
            
#         except Exception as e:
#             print(f"[TRIM RESET ERROR] {str(e)}")
#             messagebox.showerror("Error", f"Reset failed:\n{str(e)}")


#     def update_trim_display(self):
#         """Update trim UI display based on signal duration"""
#         try:
#             # ✅ CHECK if UI elements exist
#             if not hasattr(self, 'trim_duration_label'):
#                 print("[TRIM-DISPLAY] UI not ready yet, skipping")
#                 return
            
#             if self.raw_signal is None or len(self.raw_signal) == 0:
#                 self.trim_duration_label.config(text="Signal Duration: Not loaded")
#                 self.trim_start_scale.config(to=100)
#                 self.trim_end_scale.config(to=100)
#                 return
            
#             total_duration = len(self.raw_signal) / FS
            
#             # Update duration label
#             self.trim_duration_label.config(
#                 text=f"Signal Duration: {total_duration:.2f}s ({len(self.raw_signal):,} samples)"
#             )
            
#             # Update scale ranges
#             self.trim_start_scale.config(to=100)
#             self.trim_end_scale.config(to=100)
            
#             # Initial label update
#             start_pct = float(self.trim_start_var.get())
#             end_pct = float(self.trim_end_var.get())
            
#             start_sec = (start_pct / 100.0) * total_duration
#             end_sec = (end_pct / 100.0) * total_duration
            
#             self.trim_start_label.config(text=f"{start_sec:.2f}s")
#             self.trim_end_label.config(text=f"{end_sec:.2f}s")
            
#             print(f"[TRIM-DISPLAY] Updated: {total_duration:.2f}s signal loaded")
            
#         except Exception as e:
#             print(f"[TRIM-DISPLAY ERROR] {e}")

# # =====================================================
# # MAIN ENTRY POINT
# # =====================================================
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = SVEDetectionGUI(root)
#     root.mainloop()

# main.py
"""
SVE Detection System - Main Application
UNet-based Segmentation for SVE Detection with Evaluation
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import tensorflow as tf
import threading
import time
import os
import atexit

# Import custom modules
from config import *
from utils import setup_gpu, check_libraries, format_time
from preprocessor import ECGPreprocessor
from segmentation import UNetSegmentation
from shimmer_manager import ShimmerManager
#from wfdb_evaluator import WFDBEvaluator
from ui_components import UIComponents
from processing import SignalProcessor
from visualization import Visualizer
from signal_trimmer import SignalTrimmer

# Check available libraries
LIBRARIES = check_libraries()
WFDB_AVAILABLE = LIBRARIES['wfdb']
SHIMMER_AVAILABLE = LIBRARIES['shimmer']

if WFDB_AVAILABLE:
    import wfdb

# Setup GPU
setup_gpu()


class SVEDetectionGUI:
    """Main GUI Application - UNet Segmentation"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("SVE Detection System - UNet Segmentation")
        
        # Window setup
        self._setup_window()
        
        # State variables
        self._init_state_variables()
        
        # Data storage
        self._init_data_storage()
        
        # Initialize managers FIRST
        self.shimmer = ShimmerManager()
        self.preprocessor = ECGPreprocessor()
        #self.evaluator = WFDBEvaluator()
        self.ground_truth_loaded = False
        
        # Initialize trimmer
        self.trimmer = SignalTrimmer()
        print("[INIT] ✓ Trimmer initialized")
        
        # Model
        self.model = None
        self.segmentation = None
        self.model_loaded = False
        
        # Visualization
        self.segment_width = SEGMENT_WIDTH
        self.current_segment = 0
        self.total_segments = 0
        
        # Threading
        self.results_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        self.processing_thread = None
        self.stop_processing = False
        
        # Initialize helper classes
        self.processor = SignalProcessor(self)
        self.visualizer = Visualizer(self)
        self.ui_components = None
        
        # Setup
        atexit.register(self.cleanup)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Setup UI
        self.setup_ui()
        
        # Auto-load model
        self.root.after(500, self.auto_load_model)
        
        # Start update loop
        self.update_loop()
    
    def _setup_window(self):
        """Setup main window"""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = max(min(int(screen_width * 0.95), 1600), 1000)
        window_height = max(min(int(screen_height * 0.95), 900), 700)
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.configure(bg=COLORS['bg'])
        self.scale_factor = window_width / 1400
        self.colors = COLORS
    
    def _init_state_variables(self):
        """Initialize state variables"""
        self.is_recording = False
        self.is_processing = False
        self.source_type = None
        self.recording_start_time = 0
        self.processing_start_time = 0
        self.target_duration = 0
        self.processing_time = 0.0
        self.inference_start_time = None
        self.last_plot_update = 0
        self.analyze_btn_enabled = False
        self.start_record_enabled = False
        self.stop_record_enabled = False
        self.recording_start_samples = 0
        self.actual_recorded_samples = 0
        self.actual_recording_duration = 0.0
    
    def _init_data_storage(self):
        """Initialize data storage"""
        self.raw_signal = np.array([], dtype=np.float32)
        self.preprocessed_signal = None
        self.signal_before_norm = None
        self.probability_mask = None
        self.binary_mask = None
        self.sve_burden = 0.0
        self.classification = "Waiting..."
        self.mean_amplitude = 0.0
        self.sve_samples = 0
        self.total_samples = 0
        
        self.buffer_results = {
            'all_masks': [],
            'all_probs': [],
            'processed_samples': 0,
            'last_segment_end': 0
        }
    
    def scale(self, value):
        """Scale value based on window size"""
        return int(value * self.scale_factor)
    
    # =====================================================
    # UI SETUP
    # =====================================================
    def setup_ui(self):
        """Setup entire UI"""
        self.ui_components = UIComponents(self)
        self.ui_components.setup_header()
        self.ui_components.setup_main_container()
    
    # =====================================================
    # MODEL MANAGEMENT
    # =====================================================
    def auto_load_model(self):
        """Auto load model from default path"""
        def load_thread():
            try:
                self.root.after(0, lambda: self.model_status.config(
                    text="⏳ Loading model...", fg="#F57C00"))
                self.root.after(0, lambda: self.model_indicator.config(
                    text="● Model: Loading...", fg="#F57C00"))
                
                if not os.path.exists(DEFAULT_MODEL_PATH):
                    print(f"[AUTO-LOAD] ❌ Model not found: {DEFAULT_MODEL_PATH}")
                    self.root.after(0, lambda: self._on_model_not_found())
                    return
                
                print(f"[AUTO-LOAD] Loading model...")
                model = tf.keras.models.load_model(DEFAULT_MODEL_PATH, compile=False)
                print(f"[AUTO-LOAD] ✓ Model loaded!")
                
                self.root.after(0, lambda m=model: self._on_model_loaded(m))
                
            except Exception as e:
                print(f"[AUTO-LOAD ERROR] {str(e)}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda err=str(e): self._on_model_load_error(err))
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def _on_model_not_found(self):
        """Callback: Model not found"""
        self.model_loaded = False
        self.model_status.config(text=f"❌ Model not found", fg=self.colors['danger'])
        self.model_indicator.config(text="● Model: Not Found", fg=self.colors['danger'])
        messagebox.showwarning("Model Not Found",
            f"⚠️ Default model not found:\n{DEFAULT_MODEL_PATH}\n\n"
            f"Place your .h5 model file in the same directory.")
    
    def _on_model_loaded(self, model):
        """Callback: Model loaded"""
        self.model = model
        self.segmentation = UNetSegmentation(self.model)
        self.model_loaded = True
        
        filename = os.path.basename(DEFAULT_MODEL_PATH)
        self.model_status.config(text=f"✓ {filename}", fg=self.colors['success'])
        self.model_indicator.config(text="● Model: Ready", fg=self.colors['success'])
        
        print(f"[AUTO-LOAD] ✓ Ready for inference!")
        messagebox.showinfo("Model Loaded",
            f"✓ Model loaded!\n\nFile: {filename}\nReady for analysis!")
    
    def _on_model_load_error(self, error_msg):
        """Callback: Model load error"""
        self.model_loaded = False
        self.model_status.config(text="❌ Load failed", fg=self.colors['danger'])
        self.model_indicator.config(text="● Model: Error", fg=self.colors['danger'])
        messagebox.showerror("Model Error", f"Failed to load model:\n\n{error_msg}")
    
    # =====================================================
    # WFDB FILE HANDLING
    # =====================================================
    def load_wfdb(self):
        """Load WFDB file"""
        try:
            if not WFDB_AVAILABLE:
                messagebox.showerror("Error", "WFDB not installed.\nInstall: pip install wfdb")
                return
            
            file_path = filedialog.askopenfilename(
                title="Select WFDB Record",
                filetypes=[("WFDB files", "*.dat"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
            
            record_name = os.path.splitext(file_path)[0]
            
            def load_thread():
                try:
                    record = wfdb.rdrecord(record_name)
                    signal = record.p_signal[:, 1].astype(np.float32)
                    self.root.after(0, lambda: self._on_wfdb_loaded(signal, record_name))
                except Exception as e:
                    self.root.after(0, lambda: self._on_wfdb_load_error(str(e)))
            
            threading.Thread(target=load_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load WFDB:\n{str(e)}")
    
    def _on_wfdb_loaded(self, signal, record_name):
        """Callback: WFDB loaded"""
        self.raw_signal = signal
        self.source_type = 'wfdb'
        
        # Store in trimmer
        if self.trimmer is not None:
            self.trimmer.set_original_signal(signal)
        
        filename = os.path.basename(record_name)
        self.wfdb_status.config(text=f"✓ {filename}", fg=self.colors['success'])
        self.source_indicator.config(text="Source: WFDB", fg=self.colors['success'])
        
        # Enable analyze button
        self.analyze_btn.bg_color = self.colors['warning']
        self.analyze_btn.fg_color = self.colors['dark']
        self.analyze_btn.draw_button()
        self.analyze_btn_enabled = True
        
        duration = len(signal) / FS
        messagebox.showinfo("Success", 
            f"WFDB loaded!\n\nSamples: {len(signal):,}\nDuration: {duration:.2f}s\n\n"
            f"Click 'Analyze WFDB' to process.")
        
        # Update trim UI
        self.update_trim_display()
    
    def _on_wfdb_load_error(self, error_msg):
        """Callback: WFDB load error"""
        self.wfdb_status.config(text="Failed", fg=self.colors['danger'])
        messagebox.showerror("Error", f"Failed to load WFDB:\n{error_msg}")
    
    # =====================================================
    # SIGNAL TRIMMING
    # =====================================================
    def apply_signal_trim(self):
        """Apply trimming"""
        try:
            if self.raw_signal is None or len(self.raw_signal) == 0:
                messagebox.showwarning("Warning", "Load WFDB file first!")
                return
            
            start_pct = float(self.trim_start_var.get())
            end_pct = float(self.trim_end_var.get())
            
            total_duration = len(self.raw_signal) / FS
            start_sec = (start_pct / 100.0) * total_duration
            end_sec = (end_pct / 100.0) * total_duration
            
            if start_sec >= end_sec:
                messagebox.showerror("Error", 
                    f"Start ({start_sec:.2f}s) must be < End ({end_sec:.2f}s)")
                return
            
            trimmed_signal = self.trimmer.trim_signal(start_sec, end_sec, in_place=True)
            self.raw_signal = trimmed_signal
            
            trimmed_duration = len(trimmed_signal) / FS
            messagebox.showinfo("Trim Complete",
                f"✂️ Trimmed!\n\n"
                f"Duration: {trimmed_duration:.2f}s ({len(trimmed_signal):,} samples)\n"
                f"Range: {start_sec:.2f}s - {end_sec:.2f}s")
            
            self.wfdb_status.config(
                text=f"✓ Trimmed: {trimmed_duration:.2f}s",
                fg=self.colors['success'])
            
        except Exception as e:
            messagebox.showerror("Error", f"Trim failed:\n{str(e)}")
    
    def reset_signal_trim(self):
        """Reset to original"""
        try:
            if self.raw_signal is None:
                messagebox.showwarning("Warning", "No signal loaded")
                return
            
            self.trimmer.reset_to_original()
            if self.trimmer.original_signal is not None:
                self.raw_signal = self.trimmer.original_signal.copy()
            
            total_duration = len(self.raw_signal) / FS
            self.wfdb_status.config(
                text=f"✓ Original: {total_duration:.2f}s",
                fg=self.colors['success'])
            
            self.trim_start_var.set("0")
            self.trim_end_var.set("100")
            self.update_trim_display()
            
            messagebox.showinfo("Reset", f"Reset to original\n\nDuration: {total_duration:.2f}s")
            
        except Exception as e:
            messagebox.showerror("Error", f"Reset failed:\n{str(e)}")
    
    def update_trim_display(self):
        """Update trim UI"""
        try:
            if not hasattr(self, 'trim_duration_label'):
                return
            
            if self.raw_signal is None or len(self.raw_signal) == 0:
                self.trim_duration_label.config(text="Signal Duration: Not loaded")
                return
            
            total_duration = len(self.raw_signal) / FS
            self.trim_duration_label.config(
                text=f"Signal Duration: {total_duration:.2f}s ({len(self.raw_signal):,} samples)")
            
            start_pct = float(self.trim_start_var.get())
            end_pct = float(self.trim_end_var.get())
            start_sec = (start_pct / 100.0) * total_duration
            end_sec = (end_pct / 100.0) * total_duration
            
            self.trim_start_label.config(text=f"{start_sec:.2f}s")
            self.trim_end_label.config(text=f"{end_sec:.2f}s")
            
        except Exception as e:
            print(f"[TRIM-DISPLAY ERROR] {e}")
    
    # =====================================================
    # GROUND TRUTH & EVALUATION
    # =====================================================
    def load_ground_truth(self):
        """Load ground truth"""
        try:
            if not WFDB_AVAILABLE:
                messagebox.showerror("Error", "WFDB not installed")
                return
            
            file_path = filedialog.askopenfilename(
                title="Select WFDB Ground Truth",
                filetypes=[("WFDB files", "*.dat"), ("All files", "*.*")])
            
            if not file_path:
                return
            
            record_name = os.path.splitext(file_path)[0]
            
            def load_thread():
                try:
                    success = self.evaluator.load_ground_truth(record_name)
                    self.root.after(0, lambda s=success, n=record_name: self._on_gt_loaded(s, n))
                except Exception as e:
                    self.root.after(0, lambda err=str(e): self._on_gt_error(err))
            
            threading.Thread(target=load_thread, daemon=True).start()
            self.gt_status.config(text="⏳ Loading GT...", fg="#F57C00")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed:\n{str(e)}")
    
    def _on_gt_loaded(self, success, record_name):
        """Callback: GT loaded"""
        if success:
            self.ground_truth_loaded = True
            filename = os.path.basename(record_name)
            
            gt_burden = (np.sum(self.evaluator.ground_truth_mask) / 
                        len(self.evaluator.ground_truth_mask) * 100)
            
            self.gt_status.config(text=f"✓ {filename} ({gt_burden:.1f}% SVE)",
                                 fg=self.colors['success'])
            
            messagebox.showinfo("Ground Truth Loaded",
                f"✓ GT loaded!\n\n"
                f"Samples: {len(self.evaluator.ground_truth_mask):,}\n"
                f"SVE burden: {gt_burden:.2f}%\n"
                f"Symbols: S, A, J")
        else:
            self.ground_truth_loaded = False
            self.gt_status.config(text="⚠️ No annotation", fg=self.colors['warning'])
            messagebox.showwarning("No Annotation", "No .atr file found")
    
    def _on_gt_error(self, error_msg):
        """Callback: GT error"""
        self.ground_truth_loaded = False
        self.gt_status.config(text="✗ Failed", fg=self.colors['danger'])
        messagebox.showerror("Error", f"GT load failed:\n{error_msg}")
    
    def evaluate_model(self):
        """Evaluate predictions"""
        try:
            if not self.ground_truth_loaded:
                messagebox.showwarning("Warning", "Load ground truth first!")
                return
            
            if self.binary_mask is None or len(self.binary_mask) == 0:
                messagebox.showwarning("Warning", "Analyze signal first!")
                return
            
            threshold = self.threshold_var.get()
            metrics = self.evaluator.evaluate(self.binary_mask, threshold=threshold)
            
            # Update UI
            cm = metrics['confusion_matrix']
            self.eval_tp_label.config(text=f"{cm['TP']:,}")
            self.eval_tn_label.config(text=f"{cm['TN']:,}")
            self.eval_fp_label.config(text=f"{cm['FP']:,}")
            self.eval_fn_label.config(text=f"{cm['FN']:,}")
            
            m = metrics['metrics']
            self.eval_acc_label.config(text=f"{m['accuracy']*100:.1f}%")
            self.eval_prec_label.config(text=f"{m['precision']*100:.1f}%")
            self.eval_recall_label.config(text=f"{m['recall']*100:.1f}%")
            self.eval_f1_label.config(text=f"{m['f1_score']*100:.1f}%")
            
            self.evaluator.print_evaluation(metrics)
            
            b = metrics['burden']
            messagebox.showinfo("Evaluation Complete",
                f"📊 Results\n\n"
                f"Accuracy : {m['accuracy']*100:.2f}%\n"
                f"Precision: {m['precision']*100:.2f}%\n"
                f"Recall   : {m['recall']*100:.2f}%\n"
                f"F1-Score : {m['f1_score']*100:.2f}%\n\n"
                f"TP:{cm['TP']:,} FP:{cm['FP']:,}\n"
                f"FN:{cm['FN']:,} TN:{cm['TN']:,}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Evaluation failed:\n{str(e)}")
    
    # =====================================================
    # DELEGATE TO PROCESSORS
    # =====================================================
    def refresh_com_ports(self):
        self.processor.refresh_com_ports()
    
    def connect_shimmer(self):
        threading.Thread(target=self.processor.connect_shimmer, daemon=True).start()
    
    def disconnect_shimmer(self):
        self.processor.disconnect_shimmer()
    
    def start_recording(self):
        threading.Thread(target=self.processor.start_recording, daemon=True).start()
    
    def stop_recording(self):
        threading.Thread(target=self.processor.stop_recording, daemon=True).start()
    
    def analyze_signal(self):
        self.processor.analyze_signal()
    
    def prev_segment(self):
        self.visualizer.prev_segment()
    
    def next_segment(self):
        self.visualizer.next_segment()
    
    # =====================================================
    # RESET & UPDATE
    # =====================================================
    def reset_all(self):
        """Reset all"""
        try:
            self.is_recording = False
            self.is_processing = False
            self.stop_processing = True
            time.sleep(0.3)
            
            with self.results_lock:
                self.raw_signal = np.array([], dtype=np.float32)
                self.preprocessed_signal = None
                self.binary_mask = None
                self.probability_mask = None
                self.sve_burden = 0.0
                self.classification = "Waiting..."
                self.sve_samples = 0
                self.total_samples = 0
                self.current_segment = 0
                self.total_segments = 0
            
            # Reset UI
            self.classification_label.config(text="Waiting...", fg=self.colors['dark'])
            self.burden_label.config(text="0.00%")
            self.sve_samples_label.config(text="0 / 0")
            self.duration_label.config(text="0.00")
            self.mean_label.config(text="0.00")
            self.sample_count_label.config(text="0")
            self.timer_label.config(text="0:00 / 0:00")
            self.proc_time_label.config(text="0.00 sec")
            
            self.ax1.clear()
            self.ax2.clear()
            self.ui_components._setup_plots()
            self.canvas.draw()
            
            messagebox.showinfo("Reset", "All data reset!")
        except Exception as e:
            messagebox.showerror("Error", f"Reset failed:\n{str(e)}")
    
    def update_loop(self):
        """Main update loop"""
        try:
            if self.is_recording and self.target_duration > 0:
                elapsed = time.time() - self.recording_start_time
                timer_text = f"{format_time(elapsed)} / {format_time(self.target_duration)}"
                self.timer_label.config(text=timer_text)
                
                remaining = self.target_duration - elapsed
                if remaining < 5:
                    self.timer_label.config(fg="#B71C1C")
                else:
                    self.timer_label.config(fg="#E65100")
                
                if self.source_type == 'shimmer' and self.shimmer.is_streaming:
                    full_buffer = self.shimmer.get_buffer_copy()
                    if len(full_buffer) > 0 and hasattr(self, 'recording_start_samples'):
                        start_idx = self.recording_start_samples
                        if len(full_buffer) > start_idx:
                            actual_recorded = len(full_buffer) - start_idx
                            self.raw_signal = full_buffer[start_idx:].copy()
                        else:
                            actual_recorded = 0
                        self.sample_count_label.config(text=str(actual_recorded))
                
                current_time = time.time()
                if not hasattr(self, 'last_plot_update'):
                    self.last_plot_update = 0
                if current_time - self.last_plot_update >= 1.0:
                    self.visualizer.update_live_plot()
                    self.last_plot_update = current_time
                
                if elapsed >= self.target_duration:
                    self.stop_recording()
        except Exception as e:
            print(f"[UPDATE-LOOP ERROR] {str(e)}")
        
        self.root.after(500, self.update_loop)
    
    # =====================================================
    # CLEANUP
    # =====================================================
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.stop_processing = True
            if self.is_recording:
                self.is_recording = False
                time.sleep(0.2)
            if self.source_type == 'shimmer':
                try:
                    if self.shimmer.is_streaming:
                        self.shimmer.stop_streaming()
                    time.sleep(0.3)
                    self.shimmer.disconnect()
                except:
                    pass
        except:
            pass
    
    def on_closing(self):
        """Handle window closing"""
        try:
            self.cleanup()
            self.root.destroy()
        except:
            self.root.destroy()


# =====================================================
# MAIN ENTRY POINT
# =====================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = SVEDetectionGUI(root)
    root.mainloop()