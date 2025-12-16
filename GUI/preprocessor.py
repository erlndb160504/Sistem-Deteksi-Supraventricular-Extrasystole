# ============================================================
# FILE 1: preprocessor.py - COMPLETE UPDATED VERSION
# ============================================================

import numpy as np
from scipy.signal import filtfilt, firwin, butter
from config import FS, WINDOW_SIZE, OVERLAP


class ECGPreprocessor:
    """ECG Preprocessing - MATCHES TRAINING PIPELINE"""
    
    def __init__(self, fs=FS, notch_freq=50, 
                 window_size=WINDOW_SIZE, overlap=OVERLAP):
        self.fs = fs
        self.notch_freq = notch_freq
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap
        self.signal_before_norm = None
    
    def notch_filter_50hz(self, x, Q=30):
        """Notch filter for 50 Hz power line interference"""
        if len(x) < 2:
            return x
        try:
            w0 = 2 * np.pi * self.notch_freq / self.fs
            r = 1 - (w0 / Q)
            b = np.array([1, -2*np.cos(w0), 1])
            a = np.array([1, -2*r*np.cos(w0), r**2])
            return filtfilt(b, a, x)
        except:
            return x
    
    def bandpass_filter(self, x, low_freq=0.5, high_freq=50, order=4):
        """Bandpass filter 0.5-50 Hz (MATCHES TRAINING)"""
        if len(x) < 2:
            return x
        try:
            nyquist = 0.5 * self.fs
            b, a = butter(order, [low_freq/nyquist, high_freq/nyquist], btype='band')
            return filtfilt(b, a, x)
        except:
            return x
    
    def normalize_robust(self, x):
        """Robust normalization using percentiles 1-99 (MATCHES TRAINING)"""
        if len(x) == 0:
            return x
        q1, q99 = np.percentile(x, [1, 99])
        if q99 - q1 < 1e-8:
            return np.zeros_like(x)
        x_clipped = np.clip(x, q1, q99)
        return (x_clipped - q1) / (q99 - q1)
    
    def segment_signal(self, x):
        """Segment signal into overlapping windows"""
        windows = []
        for start in range(0, max(len(x) - self.window_size + 1, 1), self.stride):
            end = start + self.window_size
            if end > len(x):
                window = np.zeros(self.window_size)
                window[:len(x)-start] = x[start:]
            else:
                window = x[start:end]
            windows.append({'start': start, 'window': window})
        return windows
    
    def dc_removal(self, signal, window=128):
        """Remove DC using moving average subtraction (ONLY FOR SHIMMER LIVE PLOT)"""
        if len(signal) < window:
            return signal - np.mean(signal)
        kernel = np.ones(window) / window
        moving_mean = np.convolve(signal, kernel, mode='same')
        return signal - moving_mean

    def preprocess_pipeline(self, raw_signal, remove_dc=False):
        """
        Complete preprocessing pipeline (MATCHES TRAINING)
        
        Args:
            raw_signal: Input ECG signal
            remove_dc: If True, apply DC removal. Only for WFDB with very low frequency noise
        
        Returns:
            signal_normalized: Normalized signal for model input
            windows: List of windowed segments
            signal_before_norm: Filtered signal before normalization (for plotting)
        """
        if len(raw_signal) < 100:
            return None, [], None
        
        # Convert raw to numpy
        signal = np.array(raw_signal, dtype=np.float32)
        
        # Step 0: DC Removal (OPTIONAL - skip for Shimmer)
        if remove_dc:
            signal = self.dc_removal(signal)
        
        # Step 1: Notch filter 50 Hz
        signal = self.notch_filter_50hz(signal)
        
        # Step 2: Bandpass filter 0.5-50 Hz (MATCHES TRAINING!)
        signal = self.bandpass_filter(signal, low_freq=0.5, high_freq=50, order=4)
        
        # Keep filtered signal for plotting (before normalization)
        signal_before_norm = signal.copy()
        self.signal_before_norm = signal_before_norm
        
        # Step 3: Robust normalization (MATCHES TRAINING!)
        signal_normalized = self.normalize_robust(signal)
        
        # Step 4: Windowing with overlap
        windows = self.segment_signal(signal_normalized)
        
        return signal_normalized, windows, signal_before_norm