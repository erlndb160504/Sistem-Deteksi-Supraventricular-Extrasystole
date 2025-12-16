# exporter.py
"""
Data Export Module
Handles exporting analysis results to CSV
"""
import csv
from datetime import datetime
from config import FS, WINDOW_SIZE, OVERLAP, PROB_THRESHOLD, SVE_BURDEN_THRESHOLD


class ResultExporter:
    """Export analysis results to CSV"""
    
    @staticmethod
    def export_to_csv(file_path, data_dict):
        """
        Export results to CSV file
        
        Args:
            file_path: Path to save CSV file
            data_dict: Dictionary containing all analysis data
        """
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['SVE Detection Analysis - UNet Segmentation'])
            writer.writerow(['Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow(['Source:', data_dict.get('source_type', 'Unknown')])
            writer.writerow([])
            
            # Configuration
            writer.writerow(['=== CONFIGURATION ==='])
            writer.writerow(['Sampling Frequency:', f'{FS} Hz'])
            writer.writerow(['Window Size:', WINDOW_SIZE])
            writer.writerow(['Overlap:', OVERLAP])
            writer.writerow(['Probability Threshold:', PROB_THRESHOLD])
            writer.writerow(['SVE Burden Threshold:', f'{SVE_BURDEN_THRESHOLD}%'])
            writer.writerow([])
            
            # Results Summary
            writer.writerow(['=== RESULTS SUMMARY ==='])
            writer.writerow(['Classification:', data_dict.get('classification', 'N/A')])
            writer.writerow(['SVE Burden (%):', f"{data_dict.get('sve_burden', 0):.2f}"])
            writer.writerow(['SVE Region Samples:', data_dict.get('sve_samples', 0)])
            writer.writerow(['Total Episodes:', data_dict.get('total_episodes', 0)])
            writer.writerow(['Total Samples:', data_dict.get('total_samples', 0)])
            writer.writerow(['Duration (sec):', f"{data_dict.get('duration_sec', 0):.2f}"])
            writer.writerow(['Mean Amplitude (mV):', f"{data_dict.get('mean_amplitude', 0):.2f}"])
            writer.writerow(['Processing Time (sec):', f"{data_dict.get('processing_time', 0):.4f}"])
            writer.writerow([])
            
            # Signal data
            writer.writerow(['=== SIGNAL DATA ==='])
            writer.writerow(['Index', 'Time(s)', 'Raw(mV)', 'Preprocessed', 
                           'Prob_Mask', 'Binary_Mask', 'SVE_Region'])
            
            raw_signal = data_dict.get('raw_signal', [])
            preprocessed_signal = data_dict.get('preprocessed_signal')
            probability_mask = data_dict.get('probability_mask')
            binary_mask = data_dict.get('binary_mask')
            
            for i in range(len(raw_signal)):
                time_s = i / FS
                raw = raw_signal[i]
                
                prep = 0.0
                prob = 0.0
                mask_val = 0
                
                if preprocessed_signal is not None and i < len(preprocessed_signal):
                    prep = preprocessed_signal[i]
                
                if probability_mask is not None and i < len(probability_mask):
                    prob = probability_mask[i]
                
                if binary_mask is not None and i < len(binary_mask):
                    mask_val = int(binary_mask[i])
                
                writer.writerow([
                    i, 
                    f'{time_s:.4f}', 
                    f'{raw:.6f}', 
                    f'{prep:.6f}',
                    f'{prob:.6f}',
                    mask_val, 
                    'Y' if mask_val == 1 else 'N'
                ])