# signal_trimmer.py
"""
Signal Trimming/Cutting Module
Allows users to specify start/end times and cut WFDB signals
"""
import numpy as np
from config import FS


class SignalTrimmer:
    """Handle signal trimming and cutting operations"""
    
    def __init__(self):
        self.original_signal = None
        self.trim_start_sec = 0.0
        self.trim_end_sec = None
        self.trimmed_signal = None
    
    def set_original_signal(self, signal):
        """Store original signal"""
        self.original_signal = np.array(signal, dtype=np.float32)
        self.trim_end_sec = len(signal) / FS
    
    def get_signal_duration(self):
        """Get total duration in seconds"""
        if self.original_signal is None:
            return 0.0
        return len(self.original_signal) / FS
    
    def trim_signal(self, start_sec=0.0, end_sec=None, in_place=False):
        """
        Trim signal to specified range
        
        Args:
            start_sec: Start time in seconds
            end_sec: End time in seconds (None = to end)
            in_place: If True, replace original signal
        
        Returns:
            Trimmed signal
        """
        if self.original_signal is None:
            raise ValueError("No signal loaded")
        
        # Validate inputs
        start_sec = max(0, float(start_sec))
        total_duration = len(self.original_signal) / FS
        end_sec = min(total_duration, float(end_sec)) if end_sec else total_duration
        
        # Validate range
        if start_sec >= end_sec:
            raise ValueError(f"Invalid range: {start_sec}s >= {end_sec}s")
        
        # Convert to sample indices
        start_idx = int(start_sec * FS)
        end_idx = int(end_sec * FS)
        
        # Extract signal segment
        trimmed = self.original_signal[start_idx:end_idx].copy()
        
        # Store trimmed signal
        if in_place:
            self.original_signal = trimmed
            self.trim_start_sec = 0.0
            self.trim_end_sec = len(trimmed) / FS
        
        self.trimmed_signal = trimmed
        
        print(f"[TRIMMER] Signal trimmed: {start_sec:.2f}s - {end_sec:.2f}s "
              f"({len(trimmed)} samples, {len(trimmed)/FS:.2f}s)")
        
        return trimmed
    
    def get_trimmed_signal(self):
        """Get last trimmed signal"""
        return self.trimmed_signal if self.trimmed_signal is not None else self.original_signal
    
    def reset_to_original(self):
        """Reset to original signal"""
        self.trimmed_signal = None
        self.trim_start_sec = 0.0
        if self.original_signal is not None:
            self.trim_end_sec = len(self.original_signal) / FS