# segmentation.py
"""
UNet Segmentation Module
Handles SVE detection using UNet model
"""
import numpy as np
from config import WINDOW_SIZE, OVERLAP, PROB_THRESHOLD, SVE_BURDEN_THRESHOLD, PRE_R_SAMPLES, POST_R_SAMPLES, BEAT_DECISION_THRESHOLD
import numpy as np
from scipy.signal import find_peaks



class UNetSegmentation:
    """UNet-based SVE Segmentation"""
    
    def __init__(self, model):
        self.model = model
    
    def predict_window(self, window):
        """Predict mask probability for single window"""
        X_input = window.reshape(1, WINDOW_SIZE, 1)
        mask_prob = self.model.predict(X_input, verbose=0)
        
        # Handle various output formats
        if isinstance(mask_prob, np.ndarray):
            if mask_prob.ndim == 3:
                mask_prob = mask_prob.squeeze()  # (1, 512, 1) -> (512,)
            elif mask_prob.ndim == 2:
                mask_prob = mask_prob[0]  # (1, 512) -> (512,)
        
        # Ensure size matches window size
        if mask_prob.size != WINDOW_SIZE:
            indices = np.linspace(0, mask_prob.size-1, WINDOW_SIZE)
            mask_prob = np.interp(indices, np.arange(mask_prob.size), mask_prob)
        
        return mask_prob
    
    def merge_overlapping_masks(self, window_predictions, signal_length):
        """Merge overlapping window predictions with averaging"""
        merged_prob = np.zeros(signal_length)
        counts = np.zeros(signal_length)
        
        for wp in window_predictions:
            start = wp['start']
            end = min(start + WINDOW_SIZE, signal_length)
            length = end - start
            
            merged_prob[start:end] += wp['prob'][:length]
            counts[start:end] += 1
        
        # Average overlapping regions
        mask = counts > 0
        merged_prob[mask] /= counts[mask]
        
        return merged_prob
    
    def apply_threshold(self, mask_prob, threshold=PROB_THRESHOLD):
        """Apply sigmoid threshold to get binary mask"""
        return (mask_prob > threshold).astype(int)

    # def average_peak_amplitude(self, signal, fs):
    #     """
    #     Hitung rata-rata amplitudo peak sinyal ECG
    #     """
    #     peaks, _ = find_peaks(signal, distance=0.4 * fs)
        
    #     if len(peaks) == 0:
    #         return 0.0
        
    #     peak_amplitudes = np.abs(signal[peaks])
    #     return np.mean(peak_amplitudes)
    
    # def apply_threshold(self,
    #                 mask_prob,
    #                 signal=None,
    #                 fs=180,
    #                 base_threshold=0.5,
    #                 alpha=0.2):
    #     """
    #     Threshold adaptif berbasis rata-rata amplitudo peak ECG
    #     """

    #     # fallback ke threshold statis
    #     if signal is None:
    #         return (mask_prob > base_threshold).astype(int)

    #     avg_peak = self.average_peak_amplitude(signal, fs)

    #     # Normalisasi sederhana (aman untuk ECG)
    #     peak_norm = avg_peak / (np.max(np.abs(signal)) + 1e-6)

    #     # Threshold adaptif
    #     adaptive_threshold = base_threshold - alpha * peak_norm
    #     adaptive_threshold = np.clip(adaptive_threshold, 0.3, 0.7)

    #     return (mask_prob > adaptive_threshold).astype(int)

    
    def calculate_sve_burden(self, binary_mask):
        """Calculate SVE Burden (%) from binary mask"""
        if len(binary_mask) == 0:
            return 0.0
        sve_samples = np.sum(binary_mask)
        burden = (sve_samples / len(binary_mask)) * 100.0
        return burden
    
    def classify_signal(self, sve_burden, threshold=SVE_BURDEN_THRESHOLD):
        """Classify signal based on SVE Burden"""
        if sve_burden >= threshold:
            return "SVE Detected"
        else:
            return "Normal"
    
    # def count_episodes(self, binary_mask, window_size=WINDOW_SIZE, stride=None):
    #     """Count SVE episodes based on window transitions"""
    #     if stride is None:
    #         stride = window_size - OVERLAP
        
    #     if len(binary_mask) < window_size:
    #         return 1 if np.sum(binary_mask) > 0 else 0
        
    #     # Step 1: Create window mask
    #     window_mask = []
        
    #     for start in range(0, len(binary_mask), stride):
    #         end = min(start + window_size, len(binary_mask))
    #         window_segment = binary_mask[start:end]
            
    #         # Mark window as SVE if it has at least 1 SVE sample
    #         has_sve = np.sum(window_segment) > 0
    #         window_mask.append(1 if has_sve else 0)
        
    #     window_mask = np.array(window_mask)
        
    #     if len(window_mask) == 0:
    #         return 0
        
    #     # Step 2: Count episodes (0→1 transitions)
    #     episodes = 0
    #     for i in range(len(window_mask)):
    #         if window_mask[i] == 1 and (i == 0 or window_mask[i-1] == 0):
    #             episodes += 1
        
    #     return episodes
    
    # def mask_to_beats(self, binary_mask, signal, fs=180):
    #     """
    #     Convert sample-level segmentation mask → beat-level SVE detection.
    #     Returns:
    #         beat_labels : list of 0/1 per beat
    #         r_peaks     : index posisi R-peak
    #     """
    #     binary_mask = np.array(binary_mask)
    #     signal = np.array(signal)

    #     # 1. R-peak detection
    #     r_peaks, _ = find_peaks(signal, distance=0.46 * fs)

    #     if len(r_peaks) < 2:
    #         return [], r_peaks

    #     beat_labels = []

    #     # 2. Evaluasi interval antar R-peaks
    #     for i in range(len(r_peaks) - 1):
    #         start = r_peaks[i]
    #         end   = r_peaks[i + 1]

    #         if end > len(binary_mask):
    #             end = len(binary_mask)

    #         interval_mask = binary_mask[start:end]

    #         # Beat dianggap SVE kalau ada ≥1 mask aktif
    #         if np.sum(interval_mask) > 0:
    #             beat_labels.append(1)
    #         else:
    #             beat_labels.append(0)

    #     return beat_labels, r_peaks

    def calculate_beat_burden(self, beat_labels):
        if len(beat_labels) == 0:
            return 0.0
        sve_beats = np.sum(beat_labels)
        return (sve_beats / len(beat_labels)) * 100.0

    def classify_beats(self, prob_mask, r_peaks):
        """
        Convert sample-based probability mask into beat-based decisions
        """
        beat_results = []

        for r in r_peaks:
            if r - PRE_R_SAMPLES < 0 or r + POST_R_SAMPLES >= len(prob_mask):
                continue

            segment = prob_mask[
                r - PRE_R_SAMPLES : r + POST_R_SAMPLES
            ]

            mean_prob = np.mean(segment)
            is_sve = mean_prob > BEAT_DECISION_THRESHOLD

            beat_results.append({
                "r_peak": r,
                "score": mean_prob,
                "is_sve": is_sve
            })

        return beat_results
    
    