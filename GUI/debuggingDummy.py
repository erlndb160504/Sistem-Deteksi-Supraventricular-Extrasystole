# test_modules.py
"""
Module Testing Script
Test individual modules independently
"""
import numpy as np
import sys

def test_config():
    """Test config module"""
    print("=" * 50)
    print("Testing config.py...")
    try:
        from config import FS, WINDOW_SIZE, COLORS, BATCH_SIZE
        print(f"✓ FS = {FS} Hz")
        print(f"✓ WINDOW_SIZE = {WINDOW_SIZE}")
        print(f"✓ BATCH_SIZE = {BATCH_SIZE}")
        print(f"✓ COLORS loaded: {len(COLORS)} colors")
        print("✓ config.py OK")
        return True
    except Exception as e:
        print(f"✗ config.py FAILED: {e}")
        return False

def test_utils():
    """Test utils module"""
    print("\n" + "=" * 50)
    print("Testing utils.py...")
    try:
        from utils import setup_gpu, check_libraries, format_time, to_signed24
        
        # Test GPU setup
        gpu_ok = setup_gpu()
        print(f"✓ GPU setup: {gpu_ok}")
        
        # Test library check
        libs = check_libraries()
        print(f"✓ Libraries checked: {libs}")
        
        # Test time formatting
        time_str = format_time(125)
        print(f"✓ format_time(125) = {time_str}")
        assert time_str == "2:05", "Time format error"
        
        # Test 24-bit conversion
        val = to_signed24(0x800000)
        print(f"✓ to_signed24(0x800000) = {val}")
        
        print("✓ utils.py OK")
        return True
    except Exception as e:
        print(f"✗ utils.py FAILED: {e}")
        return False

def test_preprocessor():
    """Test preprocessor module"""
    print("\n" + "=" * 50)
    print("Testing preprocessor.py...")
    try:
        from preprocessor import ECGPreprocessor
        from config import FS
        
        # Create preprocessor
        preprocessor = ECGPreprocessor()
        print("✓ ECGPreprocessor created")
        
        # Test with dummy signal
        dummy_signal = np.sin(2 * np.pi * 1 * np.arange(0, 10, 1/FS))  # 1 Hz sine wave
        dummy_signal += np.random.normal(0, 0.1, len(dummy_signal))
        
        signal_norm, windows, signal_plot = preprocessor.preprocess_pipeline(dummy_signal)
        
        print(f"✓ Input signal: {len(dummy_signal)} samples")
        print(f"✓ Normalized signal: {len(signal_norm)} samples")
        print(f"✓ Windows created: {len(windows)}")
        print(f"✓ Signal for plot: {len(signal_plot)} samples")
        
        assert signal_norm is not None, "Normalized signal is None"
        assert len(windows) > 0, "No windows created"
        
        print("✓ preprocessor.py OK")
        return True
    except Exception as e:
        print(f"✗ preprocessor.py FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_segmentation():
    """Test segmentation module"""
    print("\n" + "=" * 50)
    print("Testing segmentation.py...")
    try:
        from segmentation import UNetSegmentation
        from config import WINDOW_SIZE
        
        # Create dummy model (just for structure test)
        class DummyModel:
            def predict(self, x, verbose=0):
                # Return dummy predictions
                batch_size = x.shape[0]
                return np.random.random((batch_size, WINDOW_SIZE, 1))
        
        model = DummyModel()
        segmentation = UNetSegmentation(model)
        print("✓ UNetSegmentation created")
        
        # Test prediction
        dummy_window = np.random.random(WINDOW_SIZE)
        prob = segmentation.predict_window(dummy_window)
        print(f"✓ predict_window: output shape {prob.shape}")
        
        # Test merging
        window_preds = [
            {'start': 0, 'prob': np.random.random(WINDOW_SIZE)},
            {'start': 256, 'prob': np.random.random(WINDOW_SIZE)}
        ]
        merged = segmentation.merge_overlapping_masks(window_preds, 768)
        print(f"✓ merge_overlapping_masks: output shape {merged.shape}")
        
        # Test threshold
        binary = segmentation.apply_threshold(merged, 0.5)
        print(f"✓ apply_threshold: {np.sum(binary)} positive samples")
        
        # Test burden calculation
        burden = segmentation.calculate_sve_burden(binary)
        print(f"✓ calculate_sve_burden: {burden:.2f}%")
        
        # Test classification
        classification = segmentation.classify_signal(burden)
        print(f"✓ classify_signal: {classification}")
        
        # Test episode counting
        episodes = segmentation.count_episodes(binary)
        print(f"✓ count_episodes: {episodes} episodes")
        
        print("✓ segmentation.py OK")
        return True
    except Exception as e:
        print(f"✗ segmentation.py FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_exporter():
    """Test exporter module"""
    print("\n" + "=" * 50)
    print("Testing exporter.py...")
    try:
        from exporter import ResultExporter
        import tempfile
        import os
        
        exporter = ResultExporter()
        print("✓ ResultExporter created")
        
        # Create dummy data
        data_dict = {
            'source_type': 'test',
            'classification': 'Normal',
            'sve_burden': 3.5,
            'sve_samples': 100,
            'total_episodes': 2,
            'total_samples': 1000,
            'duration_sec': 10.0,
            'mean_amplitude': 0.5,
            'processing_time': 1.234,
            'raw_signal': np.random.random(1000),
            'preprocessed_signal': np.random.random(1000),
            'probability_mask': np.random.random(1000),
            'binary_mask': np.random.randint(0, 2, 1000)
        }
        
        # Export to temp file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.close()
        
        exporter.export_to_csv(temp_file.name, data_dict)
        
        # Check file exists and has content
        assert os.path.exists(temp_file.name), "Export file not created"
        file_size = os.path.getsize(temp_file.name)
        print(f"✓ CSV exported: {file_size} bytes")
        
        # Cleanup
        os.unlink(temp_file.name)
        
        print("✓ exporter.py OK")
        return True
    except Exception as e:
        print(f"✗ exporter.py FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_widgets():
    """Test widgets module (basic import)"""
    print("\n" + "=" * 50)
    print("Testing widgets.py...")
    try:
        from widgets import RoundedButton
        print("✓ RoundedButton imported")
        print("✓ widgets.py OK (GUI test requires display)")
        return True
    except Exception as e:
        print(f"✗ widgets.py FAILED: {e}")
        return False

def test_all():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("SVE DETECTION SYSTEM - MODULE TESTING")
    print("=" * 70)
    
    results = []
    
    results.append(("config.py", test_config()))
    results.append(("utils.py", test_utils()))
    results.append(("preprocessor.py", test_preprocessor()))
    results.append(("segmentation.py", test_segmentation()))
    results.append(("exporter.py", test_exporter()))
    results.append(("widgets.py", test_widgets()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for module_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{module_name:25} {status}")
    
    total_pass = sum(1 for _, r in results if r)
    total_tests = len(results)
    
    print("=" * 70)
    print(f"Results: {total_pass}/{total_tests} modules passed")
    
    if total_pass == total_tests:
        print("\n🎉 All modules working correctly!")
        return 0
    else:
        print("\n⚠️  Some modules need attention")
        return 1

if __name__ == "__main__":
    sys.exit(test_all())