# ui_components.py
"""
UI Components Module
Handles all UI layout and widget creation
"""
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from widgets import RoundedButton
from config import RECORDING_DURATIONS


class UIComponents:
    """UI Components Manager"""
    
    def __init__(self, app):
        self.app = app
        self.root = app.root
        self.colors = app.colors
    
    # =====================================================
    # HEADER SETUP
    # =====================================================
    def setup_header(self):
        """Setup header section"""
        header = tk.Frame(self.root, bg=self.colors['white'], height=90)
        header.pack(fill=tk.X, padx=0, pady=0)
        header.pack_propagate(False)
        
        # Title
        title_frame = tk.Frame(header, bg=self.colors['white'])
        title_frame.pack(side=tk.LEFT, fill=tk.Y, pady=20, padx=30)
        
        tk.Label(
            title_frame,
            text="SVE Detection System - UNet Segmentation",
            font=("Segoe UI", max(14, int(16 * self.app.scale_factor)), "bold"),
            bg=self.colors['white'],
            fg=self.colors['dark']
        ).pack(anchor=tk.W)
        
        tk.Label(
            title_frame,
            text="FAKULTAS ILMU KOMPUTER - UNIVERSITAS BRAWIJAYA",
            font=("Segoe UI", max(8, int(10 * self.app.scale_factor))),
            bg=self.colors['white'],
            fg="#7f8c8d"
        ).pack(anchor=tk.W, pady=(3, 0))
        
        # Status indicators
        status_frame = tk.Frame(header, bg=self.colors['white'])
        status_frame.pack(side=tk.RIGHT, padx=30, pady=20)
        
        self.app.model_indicator = tk.Label(
            status_frame,
            text="● Model: Not Loaded",
            font=("Segoe UI", max(8, int(10 * self.app.scale_factor)), "bold"),
            bg=self.colors['white'],
            fg="#95a5a6"
        )
        self.app.model_indicator.pack(anchor=tk.E)
        
        self.app.source_indicator = tk.Label(
            status_frame,
            text="● Source: Disconnected",
            font=("Segoe UI", max(7, int(9 * self.app.scale_factor))),
            bg=self.colors['white'],
            fg="#95a5a6"
        )
        self.app.source_indicator.pack(anchor=tk.E, pady=(3, 0))
    
    # =====================================================
    # MAIN CONTAINER
    # =====================================================
    def setup_main_container(self):
        """Setup main content area"""
        main = tk.Frame(self.root, bg=self.colors['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        self.setup_left_panel(main)
        self.setup_middle_panel(main)
        self.setup_right_panel(main)
    
    # =====================================================
    # LEFT PANEL - CONTROL PANEL
    # =====================================================
    def setup_left_panel(self, parent):
        """Setup control panel (left)"""
        left_frame = tk.Frame(parent, bg=self.colors['white'])
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_frame.pack_propagate(False)
        left_frame.config(width=280)
        
        # Scrollable canvas
        scrollbar = ttk.Scrollbar(left_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        canvas = tk.Canvas(left_frame, bg=self.colors['white'], 
                          yscrollcommand=scrollbar.set, highlightthickness=0)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=canvas.yview)
        
        content = tk.Frame(canvas, bg=self.colors['white'])
        canvas_window = canvas.create_window(0, 0, window=content, anchor='nw')
        
        def on_frame_config(event=None):
            canvas.configure(scrollregion=canvas.bbox('all'))
        
        def on_canvas_config(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        content.bind('<Configure>', on_frame_config)
        canvas.bind('<Configure>', on_canvas_config)
        canvas.bind('<MouseWheel>', lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        # Setup sections
        self._setup_model_section(content)
        self._add_divider(content)
        # self._setup_evaluation_section(content)  # ← Tambah di sini
        # self._add_divider(content)
        self._setup_wfdb_section(content)
        self._add_divider(content)
        self._setup_shimmer_section(content)
        self._add_divider(content)
        self._setup_trimming_section(content)
        self._add_divider(content)
        self._setup_duration_section(content)
        self._add_divider(content)
        self._setup_recording_section(content)
        self._add_divider(content)
        #self._setup_export_section(content)
    
    def _setup_model_section(self, parent):
        """Model loading section - AUTO LOAD"""
        self._add_section_title(parent, "Model Configuration")
        
        self.app.model_status = tk.Label(
            parent, text="⏳ Auto-loading model...",
            font=("Segoe UI", 8), 
            bg=self.colors['white'], fg="#95a5a6"
        )
        self.app.model_status.pack(anchor=tk.W, pady=(0, 15), padx=15)

    # #EFDB EVALUATOR
    # def _setup_evaluation_section(self, parent):
    #     """Evaluation configuration section - NEW"""
    #     self._add_section_title(parent, "📊 WFDB Evaluation")
        
    #     # Ground truth selection
    #     self.app.load_gt_btn = RoundedButton(
    #         parent, text="📥 Load Ground Truth (WFDB)",
    #         command=self.app.load_ground_truth, 
    #         bg_color=self.colors['info'],
    #         fg_color=self.colors['white'], 
    #         width=220, height=32
    #     )
    #     self.app.load_gt_btn.pack(pady=(0, 8), padx=15)
        
    #     self.app.gt_status = tk.Label(
    #         parent, text="❌ No ground truth loaded",
    #         font=("Segoe UI", 8), 
    #         bg=self.colors['white'], fg="#95a5a6"
    #     )
    #     self.app.gt_status.pack(anchor=tk.W, pady=(0, 8), padx=15)
        
    #     # Threshold slider
    #     tk.Label(parent, text="Threshold (0.0 - 1.0):",
    #             font=("Segoe UI", 8), bg=self.colors['white'], 
    #             fg="#7f8c8d").pack(anchor=tk.W, pady=(0, 4), padx=15)
        
    #     threshold_frame = tk.Frame(parent, bg=self.colors['white'])
    #     threshold_frame.pack(fill=tk.X, pady=(0, 10), padx=15)
        
    #     self.app.threshold_var = tk.DoubleVar(value=0.5)
    #     threshold_scale = tk.Scale(threshold_frame, from_=0.0, to=1.0, resolution=0.05,
    #                             orient=tk.HORIZONTAL, variable=self.app.threshold_var,
    #                             bg=self.colors['white'], fg=self.colors['dark'],
    #                             troughcolor="#e0e0e0", highlightthickness=0)
    #     threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
    #     self.app.threshold_label = tk.Label(threshold_frame, text="0.50",
    #                                         font=("Segoe UI", 9, "bold"),
    #                                         bg=self.colors['white'], fg=self.colors['primary'])
    #     self.app.threshold_label.pack(side=tk.LEFT)
        
    #     # Update label ketika threshold berubah
    #     def update_threshold_label(value):
    #         self.app.threshold_label.config(text=f"{float(value):.2f}")
        
    #     threshold_scale.config(command=update_threshold_label)
        
    #     # Evaluate button
    #     self.app.evaluate_btn = RoundedButton(
    #         parent, text="🔍 Evaluate Model",
    #         command=self.app.evaluate_model, 
    #         bg_color="#FFB300",
    #         fg_color=self.colors['white'], 
    #         width=220, height=32
    #     )
    #     self.app.evaluate_btn.pack(pady=(0, 15), padx=15)


    # Tambahkan ini ke UIComponents class di ui_components.py
# Letakkan method ini sebelum _setup_wfdb_section() di setup_left_panel()

    def _setup_trimming_section(self, parent):
        """Signal trimming/cutting section"""
        self._add_section_title(parent, "✂️ Signal Trimming")
        
        tk.Label(
            parent, 
            text="(Optional: Trim WFDB signal to specific time range)",
            font=("Segoe UI", 8), 
            bg=self.colors['white'], 
            fg="#95a5a6"
        ).pack(anchor=tk.W, pady=(0, 8), padx=15)
        
        # ===== START TIME INPUT =====
        tk.Label(
            parent, 
            text="Start Time (%):",
            font=("Segoe UI", 8), 
            bg=self.colors['white'], 
            fg="#7f8c8d"
        ).pack(anchor=tk.W, pady=(0, 3), padx=15)
        
        start_frame = tk.Frame(parent, bg=self.colors['white'])
        start_frame.pack(fill=tk.X, pady=(0, 8), padx=15)
        
        self.app.trim_start_var = tk.StringVar(value="0")
        self.app.trim_start_scale = tk.Scale(
            start_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL,
            variable=self.app.trim_start_var,
            bg=self.colors['white'], 
            fg=self.colors['dark'],
            troughcolor="#e0e0e0", 
            highlightthickness=0,
            length=150,
            command=lambda v: self._update_trim_label_start()
        )
        self.app.trim_start_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        
        self.app.trim_start_label = tk.Label(
            start_frame, 
            text="0.00s",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['white'], 
            fg=self.colors['primary'],
            width=8
        )
        self.app.trim_start_label.pack(side=tk.LEFT)
        
        # ===== END TIME INPUT =====
        tk.Label(
            parent, 
            text="End Time (%):",
            font=("Segoe UI", 8), 
            bg=self.colors['white'], 
            fg="#7f8c8d"
        ).pack(anchor=tk.W, pady=(0, 3), padx=15)
        
        end_frame = tk.Frame(parent, bg=self.colors['white'])
        end_frame.pack(fill=tk.X, pady=(0, 12), padx=15)
        
        self.app.trim_end_var = tk.StringVar(value="100")
        self.app.trim_end_scale = tk.Scale(
            end_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL,
            variable=self.app.trim_end_var,
            bg=self.colors['white'], 
            fg=self.colors['dark'],
            troughcolor="#e0e0e0", 
            highlightthickness=0,
            length=150,
            command=lambda v: self._update_trim_label_end()
        )
        self.app.trim_end_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        
        self.app.trim_end_label = tk.Label(
            end_frame, 
            text="0.00s",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['white'], 
            fg=self.colors['primary'],
            width=8
        )
        self.app.trim_end_label.pack(side=tk.LEFT)
        
        # ===== SIGNAL DURATION INFO =====
        self.app.trim_duration_label = tk.Label(
            parent, 
            text="Signal Duration: Not loaded",
            font=("Segoe UI", 8), 
            bg=self.colors['white'], 
            fg="#7f8c8d"
        )
        self.app.trim_duration_label.pack(anchor=tk.W, pady=(0, 8), padx=15)
        
        # ===== TRIMMING BUTTONS =====
        trim_btn_frame = tk.Frame(parent, bg=self.colors['white'])
        trim_btn_frame.pack(fill=tk.X, pady=(0, 15), padx=15)
        
        self.app.trim_btn = RoundedButton(
            trim_btn_frame, 
            text="✂️ Apply Trim",
            command=self.app.apply_signal_trim, 
            bg_color=self.colors['info'],
            fg_color=self.colors['white'], 
            width=105, 
            height=32
        )
        self.app.trim_btn.pack(side=tk.LEFT, padx=(0, 6))
        
        self.app.trim_reset_btn = RoundedButton(
            trim_btn_frame, 
            text="🔄 Reset",
            command=self.app.reset_signal_trim, 
            bg_color="#95a5a6",
            fg_color=self.colors['white'], 
            width=105, 
            height=32
        )
        self.app.trim_reset_btn.pack(side=tk.LEFT)


    def _update_trim_label_start(self):
        """Update start time label when slider changes"""
        try:
            if self.app.raw_signal is None or len(self.app.raw_signal) == 0:
                return
            from config import FS
            total_duration = len(self.app.raw_signal) / FS
            start_pct = float(self.app.trim_start_var.get())
            start_sec = (start_pct / 100.0) * total_duration
            self.app.trim_start_label.config(text=f"{start_sec:.2f}s")
        except:
            pass


    def _update_trim_label_end(self):
        """Update end time label when slider changes"""
        try:
            if self.app.raw_signal is None or len(self.app.raw_signal) == 0:
                return
            from config import FS
            total_duration = len(self.app.raw_signal) / FS
            end_pct = float(self.app.trim_end_var.get())
            end_sec = (end_pct / 100.0) * total_duration
            self.app.trim_end_label.config(text=f"{end_sec:.2f}s")
        except:
            pass

    def _setup_wfdb_section(self, parent):
        """WFDB file input section"""
        self._add_section_title(parent, "WFDB File Input")
        
        self.app.load_wfdb_btn = RoundedButton(
            parent, text="📁 Load WFDB File",
            command=self.app.load_wfdb, 
            bg_color=self.colors['info'],
            fg_color=self.colors['white'], 
            width=220, height=32
        )
        self.app.load_wfdb_btn.pack(pady=(0, 8), padx=15)
        
        self.app.wfdb_status = tk.Label(
            parent, text="✗ No file loaded",
            font=("Segoe UI", 8), 
            bg=self.colors['white'], fg="#95a5a6"
        )
        self.app.wfdb_status.pack(anchor=tk.W, pady=(0, 8), padx=15)
        
        self.app.analyze_btn = RoundedButton(
            parent, text="🔍 Analyze WFDB",
            command=self.app.analyze_signal, 
            bg_color="#dfe6e9",
            fg_color="#636e72", 
            width=220, height=32
        )
        self.app.analyze_btn.pack(pady=(0, 15), padx=15)
    
    def _setup_shimmer_section(self, parent):
        """Shimmer connection section"""
        self._add_section_title(parent, "Shimmer Connection")
        
        self.app.shimmer_status = tk.Label(
            parent, text="✗ Disconnected",
            font=("Segoe UI", 8), 
            bg=self.colors['white'], fg="#95a5a6"
        )
        self.app.shimmer_status.pack(anchor=tk.W, pady=(0, 8), padx=15)
        
        tk.Label(parent, text="COM Port:",
                font=("Segoe UI", 8), bg=self.colors['white'], 
                fg="#7f8c8d").pack(anchor=tk.W, pady=(0, 4), padx=15)
        
        com_frame = tk.Frame(parent, bg=self.colors['white'])
        com_frame.pack(fill=tk.X, pady=(0, 8), padx=15)
        
        self.app.com_var = tk.StringVar()
        self.app.com_combo = ttk.Combobox(com_frame, textvariable=self.app.com_var, 
                                          state='readonly', width=15)
        self.app.com_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.app.refresh_com_ports()
        
        refresh_btn = tk.Button(
            com_frame, text="🔄", command=self.app.refresh_com_ports,
            bg=self.colors['info'], fg=self.colors['white'],
            font=("Segoe UI", 9, "bold"), padx=6, pady=1
        )
        refresh_btn.pack(side=tk.LEFT)
        
        btn_frame = tk.Frame(parent, bg=self.colors['white'])
        btn_frame.pack(fill=tk.X, pady=(0, 15), padx=15)
        
        self.app.connect_btn = RoundedButton(
            btn_frame, text="🔗 Connect",
            command=self.app.connect_shimmer, 
            bg_color=self.colors['success'],
            fg_color=self.colors['white'], 
            width=100, height=32
        )
        self.app.connect_btn.pack(side=tk.LEFT, padx=(0, 6))
        
        self.app.disconnect_btn = RoundedButton(
            btn_frame, text="✕ Disconnect",
            command=self.app.disconnect_shimmer, 
            bg_color=self.colors['danger'],
            fg_color=self.colors['white'], 
            width=100, height=32
        )
        self.app.disconnect_btn.pack(side=tk.LEFT)
    
    def _setup_duration_section(self, parent):
        """Recording duration section"""
        self._add_section_title(parent, "Recording Duration")
        
        self.app.duration_var = tk.StringVar(value="1 minute")
        duration_combo = ttk.Combobox(
            parent, textvariable=self.app.duration_var,
            values=list(RECORDING_DURATIONS.keys()), 
            state='readonly', width=18
        )
        duration_combo.pack(fill=tk.X, pady=(0, 12), padx=15)
    
    def _setup_recording_section(self, parent):
        """Recording control section"""
        self._add_section_title(parent, "Recording Control")
        
        record_btn_frame = tk.Frame(parent, bg=self.colors['white'])
        record_btn_frame.pack(fill=tk.X, pady=(0, 6), padx=15)
        
        self.app.start_record_btn = RoundedButton(
            record_btn_frame, text="▶ Start",
            command=self.app.start_recording, 
            bg_color="#dfe6e9",
            fg_color="#636e72", 
            width=100, height=32
        )
        self.app.start_record_btn.pack(side=tk.LEFT, padx=(0, 6))
        
        self.app.stop_record_btn = RoundedButton(
            record_btn_frame, text="■ Stop",
            command=self.app.stop_recording, 
            bg_color="#ffcccb",
            fg_color="#999999", 
            width=100, height=32
        )
        self.app.stop_record_btn.pack(side=tk.LEFT)
        
        self.app.reset_btn = RoundedButton(
            parent, text="🔄 Reset All",
            command=self.app.reset_all, 
            bg_color=self.colors['warning'],
            fg_color=self.colors['dark'], 
            width=220, height=32
        )
        self.app.reset_btn.pack(pady=(6, 15), padx=15)
    
    # def _setup_export_section(self, parent):
    #     """Export section"""
    #     self._add_section_title(parent, "Export Results")
        
    #     self.app.export_btn = RoundedButton(
    #         parent, text="💾 Export to CSV",
    #         command=self.app.export_results, 
    #         bg_color=self.colors['success'],
    #         fg_color=self.colors['white'], 
    #         width=220, height=32
    #     )
    #     self.app.export_btn.pack(padx=15, pady=(0, 8))
        
    #     self.app.export_status = tk.Label(
    #         parent, text="✓ Ready to export",
    #         font=("Segoe UI", 8), 
    #         bg=self.colors['white'], fg="#95a5a6"
    #     )
    #     self.app.export_status.pack(anchor=tk.W, pady=(0, 15), padx=15)
    
    # =====================================================
    # MIDDLE PANEL - VISUALIZATION
    # =====================================================
    def setup_middle_panel(self, parent):
        """Setup visualization panel (middle)"""
        middle = tk.Frame(parent, bg=self.colors['white'])
        middle.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Header with navigation
        header = tk.Frame(middle, bg=self.colors['white'])
        header.pack(fill=tk.X, padx=15, pady=(12, 8))
        
        title_frame = tk.Frame(header, bg=self.colors['white'])
        title_frame.pack(side=tk.LEFT)
        
        tk.Label(title_frame, text="Signal Visualization (30-second segments)",
                font=("Segoe UI", 10, "bold"),
                bg=self.colors['white'], fg=self.colors['dark']).pack(anchor=tk.W)
        
        self.app.segment_info_label = tk.Label(title_frame, text="Segment: 0/0",
                font=("Segoe UI", 9),
                bg=self.colors['white'], fg="#7f8c8d")
        self.app.segment_info_label.pack(anchor=tk.W, pady=(2, 0))
        
        # Navigation buttons
        nav_frame = tk.Frame(header, bg=self.colors['white'])
        nav_frame.pack(side=tk.RIGHT)
        
        self.app.prev_seg_btn = tk.Button(nav_frame, text="◀ Prev", 
                                          command=self.app.prev_segment,
                                          bg=self.colors['info'], fg=self.colors['white'],
                                          font=("Segoe UI", 9, "bold"), padx=10)
        self.app.prev_seg_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.app.next_seg_btn = tk.Button(nav_frame, text="Next ▶",
                                          command=self.app.next_segment,
                                          bg=self.colors['info'], fg=self.colors['white'],
                                          font=("Segoe UI", 9, "bold"), padx=10)
        self.app.next_seg_btn.pack(side=tk.LEFT)
        
        # Matplotlib figure
        canvas_frame = tk.Frame(middle, bg=self.colors['white'])
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=15, pady=(0, 12))
        
        self.app.fig = Figure(figsize=(12, 8), facecolor='white', dpi=80)
        self.app.fig.subplots_adjust(hspace=0.35, left=0.08, right=0.96, top=0.95, bottom=0.07)
        
        self.app.ax1 = self.app.fig.add_subplot(211)
        self.app.ax2 = self.app.fig.add_subplot(212)
        
        self.app.canvas = FigureCanvasTkAgg(self.app.fig, canvas_frame)
        self.app.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self._setup_plots()
    
    def _setup_plots(self):
        """Setup matplotlib plots"""
        # Plot 1: Raw Signal
        self.app.ax1.set_title("Raw ECG Signal", fontsize=10, fontweight='bold', pad=10, loc='left')
        self.app.ax1.set_ylabel("Amplitude (mV)", fontsize=9)
        self.app.ax1.set_xlabel("Time (s)", fontsize=9)
        self.app.ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        self.app.ax1.spines['top'].set_visible(False)
        self.app.ax1.spines['right'].set_visible(False)
        
        # Plot 2: Preprocessed + Mask Overlay
        self.app.ax2.set_title("Preprocessed Signal + SVE Mask (Red = SVE Region)", 
                               fontsize=10, fontweight='bold', pad=10, loc='left')
        self.app.ax2.set_ylabel("Amplitude (Normalized)", fontsize=9)
        self.app.ax2.set_xlabel("Time (s)", fontsize=9)
        self.app.ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        self.app.ax2.spines['top'].set_visible(False)
        self.app.ax2.spines['right'].set_visible(False)
    
    # =====================================================
    # RIGHT PANEL - RESULTS
    # =====================================================
    def setup_right_panel(self, parent):
        """Setup results panel (right)"""
        right_frame = tk.Frame(parent, bg=self.colors['white'])
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        right_frame.pack_propagate(False)
        right_frame.config(width=320)
        
        # Scrollable canvas
        scrollbar = ttk.Scrollbar(right_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        canvas = tk.Canvas(right_frame, bg=self.colors['white'],
                          yscrollcommand=scrollbar.set, highlightthickness=0)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=canvas.yview)
        
        content = tk.Frame(canvas, bg=self.colors['white'])
        canvas_window = canvas.create_window(0, 0, window=content, anchor='nw')
        
        def on_frame_config(event=None):
            canvas.configure(scrollregion=canvas.bbox('all'))
        
        def on_canvas_config(event):
            canvas.itemconfig(canvas_window, width=event.width - 5)
        
        content.bind('<Configure>', on_frame_config)
        canvas.bind('<Configure>', on_canvas_config)
        canvas.bind('<MouseWheel>', lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        inner = tk.Frame(content, bg=self.colors['white'])
        inner.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        
        # Results title
        tk.Label(inner, text="Analysis Results",
                font=("Segoe UI", 11, "bold"), bg=self.colors['white'],
                fg=self.colors['dark']).pack(anchor=tk.W, pady=(0, 12))
        
        # Setup result cards
        self._setup_timer_card(inner)
        self._setup_processing_time_card(inner)
        self._setup_classification_card(inner)
        self._setup_burden_card(inner)
        self._setup_samples_card(inner)
        #self._setup_episodes_card(inner)
        # self._setup_evaluation_metrics_card(inner)
        self._setup_statistics_card(inner)
    
    def _setup_timer_card(self, parent):
        """Recording timer card"""
        card = tk.Frame(parent, bg="#FFE0B2", bd=0)
        card.pack(fill=tk.X, pady=(0, 10))
        
        inner = tk.Frame(card, bg="#FFE0B2")
        inner.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        
        tk.Label(inner, text="Recording Timer:",
                font=("Segoe UI", 8), bg="#FFE0B2", fg="#7f8c8d").pack(anchor=tk.W)
        
        self.app.timer_label = tk.Label(
            inner, text="0:00 / 0:00",
            font=("Segoe UI", 12, "bold"), bg="#FFE0B2", fg="#E65100"
        )
        self.app.timer_label.pack(anchor=tk.W, pady=(3, 0))
    
    def _setup_processing_time_card(self, parent):
        """Processing time card"""
        card = tk.Frame(parent, bg="#C8E6C9", bd=0)
        card.pack(fill=tk.X, pady=(0, 10))
        
        inner = tk.Frame(card, bg="#C8E6C9")
        inner.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        
        tk.Label(inner, text="Processing Time:",
                font=("Segoe UI", 8), bg="#C8E6C9", fg="#7f8c8d").pack(anchor=tk.W)
        
        self.app.proc_time_label = tk.Label(
            inner, text="0.00 sec",
            font=("Segoe UI", 12, "bold"), bg="#C8E6C9", fg="#2E7D32"
        )
        self.app.proc_time_label.pack(anchor=tk.W, pady=(3, 0))
    
    def _setup_classification_card(self, parent):
        """Classification card"""
        card = tk.Frame(parent, bg="#FFF9E6", bd=0)
        card.pack(fill=tk.X, pady=(0, 10))
        
        inner = tk.Frame(card, bg="#FFF9E6")
        inner.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        
        tk.Label(inner, text="Classification:",
                font=("Segoe UI", 8), bg="#FFF9E6", fg="#7f8c8d").pack(anchor=tk.W)
        
        self.app.classification_label = tk.Label(
            inner, text="Waiting...",
            font=("Segoe UI", 14, "bold"), bg="#FFF9E6", fg=self.colors['dark']
        )
        self.app.classification_label.pack(anchor=tk.W, pady=(3, 0))
    
    def _setup_burden_card(self, parent):
        """SVE Burden card"""
        card = tk.Frame(parent, bg="#FFE8F0", bd=0)
        card.pack(fill=tk.X, pady=(0, 10))
        
        inner = tk.Frame(card, bg="#FFE8F0")
        inner.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        
        tk.Label(inner, text="SVE Burden (%):",
                font=("Segoe UI", 8), bg="#FFE8F0", fg="#7f8c8d").pack(anchor=tk.W)
        
        self.app.burden_label = tk.Label(
            inner, text="0.00%",
            font=("Segoe UI", 14, "bold"), bg="#FFE8F0", fg=self.colors['danger']
        )
        self.app.burden_label.pack(anchor=tk.W, pady=(3, 0))
    
    def _setup_samples_card(self, parent):
        """SVE samples card"""
        card = tk.Frame(parent, bg="#E8F4FD", bd=0)
        card.pack(fill=tk.X, pady=(0, 10))
        
        inner = tk.Frame(card, bg="#E8F4FD")
        inner.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        
        tk.Label(inner, text="SVE Region (samples):",
                font=("Segoe UI", 8), bg="#E8F4FD", fg="#7f8c8d").pack(anchor=tk.W)
        
        self.app.sve_samples_label = tk.Label(
            inner, text="0 / 0",
            font=("Segoe UI", 12, "bold"), bg="#E8F4FD", fg=self.colors['info']
        )
        self.app.sve_samples_label.pack(anchor=tk.W, pady=(3, 0))
    
    # def _setup_episodes_card(self, parent):
    #     """Episode counter card"""
    #     card = tk.Frame(parent, bg="#F3E5F5", bd=0)
    #     card.pack(fill=tk.X, pady=(0, 10))
        
    #     inner = tk.Frame(card, bg="#F3E5F5")
    #     inner.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        
    #     tk.Label(inner, text="SVE Episodes:",
    #             font=("Segoe UI", 8), bg="#F3E5F5", fg="#7f8c8d").pack(anchor=tk.W)
        
    #     self.app.episodes_label = tk.Label(
    #         inner, text="0",
    #         font=("Segoe UI", 14, "bold"), bg="#F3E5F5", fg="#7B1FA2"
    #     )
    #     self.app.episodes_label.pack(anchor=tk.W, pady=(3, 0))
    
    def _setup_statistics_card(self, parent):
        """Signal statistics card"""
        card = tk.Frame(parent, bg="#E8F5E9", bd=0)
        card.pack(fill=tk.X, pady=(0, 10))
        
        inner = tk.Frame(card, bg="#E8F5E9")
        inner.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        
        tk.Label(inner, text="Signal Statistics:",
                font=("Segoe UI", 9, "bold"), bg="#E8F5E9", fg="#7f8c8d").pack(anchor=tk.W)
        
        tk.Label(inner, text="Duration (sec):",
                font=("Segoe UI", 8), bg="#E8F5E9", fg="#7f8c8d").pack(anchor=tk.W, pady=(6, 0))
        
        self.app.duration_label = tk.Label(
            inner, text="0.00",
            font=("Segoe UI", 11, "bold"), bg="#E8F5E9", fg=self.colors['success']
        )
        self.app.duration_label.pack(anchor=tk.W, pady=(2, 0))
        
        tk.Label(inner, text="Mean Amplitude (mV):",
                font=("Segoe UI", 8), bg="#E8F5E9", fg="#7f8c8d").pack(anchor=tk.W, pady=(6, 0))
        
        self.app.mean_label = tk.Label(
            inner, text="0.00",
            font=("Segoe UI", 11, "bold"), bg="#E8F5E9", fg=self.colors['success']
        )
        self.app.mean_label.pack(anchor=tk.W, pady=(2, 0))
        
        tk.Label(inner, text="Samples Recorded:",
                font=("Segoe UI", 8), bg="#E8F5E9", fg="#7f8c8d").pack(anchor=tk.W, pady=(6, 0))
        
        self.app.sample_count_label = tk.Label(
            inner, text="0",
            font=("Segoe UI", 11, "bold"), bg="#E8F5E9", fg=self.colors['success']
        )
        self.app.sample_count_label.pack(anchor=tk.W, pady=(2, 0))

    # def _setup_evaluation_metrics_card(self, parent):
    #     """Evaluation metrics card - NEW"""
    #     card = tk.Frame(parent, bg="#F5DEB3", bd=0)
    #     card.pack(fill=tk.X, pady=(0, 10))
        
    #     inner = tk.Frame(card, bg="#F5DEB3")
    #     inner.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        
    #     tk.Label(inner, text="🎯 Model Evaluation Metrics",
    #             font=("Segoe UI", 9, "bold"), bg="#F5DEB3", fg="#7f8c8d").pack(anchor=tk.W)
        
    #     # Confusion matrix grid
    #     matrix_frame = tk.Frame(inner, bg="#F5DEB3")
    #     matrix_frame.pack(fill=tk.X, pady=(6, 0))
        
    #     # Row 1: Labels + TP/FP
    #     tk.Label(matrix_frame, text="TP:", font=("Segoe UI", 8), 
    #             bg="#F5DEB3", fg="#7f8c8d").grid(row=0, column=0, sticky="w", padx=5)
    #     self.app.eval_tp_label = tk.Label(matrix_frame, text="-", 
    #                                     font=("Segoe UI", 10, "bold"), 
    #                                     bg="#F5DEB3", fg="#2E7D32")
    #     self.app.eval_tp_label.grid(row=0, column=1, sticky="w", padx=5)
        
    #     tk.Label(matrix_frame, text="FP:", font=("Segoe UI", 8), 
    #             bg="#F5DEB3", fg="#7f8c8d").grid(row=0, column=2, sticky="w", padx=5)
    #     self.app.eval_fp_label = tk.Label(matrix_frame, text="-", 
    #                                     font=("Segoe UI", 10, "bold"), 
    #                                     bg="#F5DEB3", fg="#D32F2F")
    #     self.app.eval_fp_label.grid(row=0, column=3, sticky="w", padx=5)
        
    #     # Row 2: TN/FN
    #     tk.Label(matrix_frame, text="TN:", font=("Segoe UI", 8), 
    #             bg="#F5DEB3", fg="#7f8c8d").grid(row=1, column=0, sticky="w", padx=5, pady=(3, 0))
    #     self.app.eval_tn_label = tk.Label(matrix_frame, text="-", 
    #                                     font=("Segoe UI", 10, "bold"), 
    #                                     bg="#F5DEB3", fg="#1976D2")
    #     self.app.eval_tn_label.grid(row=1, column=1, sticky="w", padx=5, pady=(3, 0))
        
    #     tk.Label(matrix_frame, text="FN:", font=("Segoe UI", 8), 
    #             bg="#F5DEB3", fg="#7f8c8d").grid(row=1, column=2, sticky="w", padx=5, pady=(3, 0))
    #     self.app.eval_fn_label = tk.Label(matrix_frame, text="-", 
    #                                     font=("Segoe UI", 10, "bold"), 
    #                                     bg="#F5DEB3", fg="#F57F17")
    #     self.app.eval_fn_label.grid(row=1, column=3, sticky="w", padx=5, pady=(3, 0))
        
    #     # Performance metrics
    #     perf_frame = tk.Frame(inner, bg="#F5DEB3")
    #     perf_frame.pack(fill=tk.X, pady=(6, 0))
        
    #     tk.Label(perf_frame, text="Acc:", font=("Segoe UI", 8), 
    #             bg="#F5DEB3", fg="#7f8c8d").grid(row=0, column=0, sticky="w", padx=5)
    #     self.app.eval_acc_label = tk.Label(perf_frame, text="-", 
    #                                         font=("Segoe UI", 10, "bold"), 
    #                                         bg="#F5DEB3", fg="#388E3C")
    #     self.app.eval_acc_label.grid(row=0, column=1, sticky="w", padx=5)
        
    #     tk.Label(perf_frame, text="Pre:", font=("Segoe UI", 8), 
    #             bg="#F5DEB3", fg="#7f8c8d").grid(row=0, column=2, sticky="w", padx=5)
    #     self.app.eval_prec_label = tk.Label(perf_frame, text="-", 
    #                                         font=("Segoe UI", 10, "bold"), 
    #                                         bg="#F5DEB3", fg="#6A1B9A")
    #     self.app.eval_prec_label.grid(row=0, column=3, sticky="w", padx=5)
        
    #     tk.Label(perf_frame, text="Rec:", font=("Segoe UI", 8), 
    #             bg="#F5DEB3", fg="#7f8c8d").grid(row=1, column=0, sticky="w", padx=5, pady=(3, 0))
    #     self.app.eval_recall_label = tk.Label(perf_frame, text="-", 
    #                                         font=("Segoe UI", 10, "bold"), 
    #                                         bg="#F5DEB3", fg="#F57C00")
    #     self.app.eval_recall_label.grid(row=1, column=1, sticky="w", padx=5, pady=(3, 0))
        
    #     tk.Label(perf_frame, text="F1:", font=("Segoe UI", 8), 
    #             bg="#F5DEB3", fg="#7f8c8d").grid(row=1, column=2, sticky="w", padx=5, pady=(3, 0))
    #     self.app.eval_f1_label = tk.Label(perf_frame, text="-", 
    #                                     font=("Segoe UI", 10, "bold"), 
    #                                     bg="#F5DEB3", fg="#0097A7")
    #     self.app.eval_f1_label.grid(row=1, column=3, sticky="w", padx=5, pady=(3, 0))
    
    # =====================================================
    # HELPER METHODS
    # =====================================================
    def _add_section_title(self, parent, text):
        """Add section title"""
        tk.Label(parent, text=text,
                font=("Segoe UI", 11, "bold"),
                bg=self.colors['white'], fg=self.colors['dark']).pack(
                    anchor=tk.W, pady=(8, 10), padx=15)
    
    def _add_divider(self, parent):
        """Add horizontal divider"""
        tk.Frame(parent, bg="#e0e0e0", height=1).pack(fill=tk.X, pady=12)