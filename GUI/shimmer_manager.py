# shimmer_manager.py
"""
Shimmer Device Manager Module
Handles connection and data streaming from Shimmer ECG device
"""
import threading
import time
import numpy as np
from collections import deque
from config import DEFAULT_BAUDRATE, SENS_MV
from utils import to_signed24
from serial.tools import list_ports
import serial

try:
    from pyshimmer import ShimmerBluetooth, DataPacket, EChannelType
    CH_ECG = EChannelType.EXG_ADS1292R_1_CH1_24BIT  # ECG Channel
    SHIMMER_AVAILABLE = True
except ImportError:
    SHIMMER_AVAILABLE = False

class ShimmerManager:
    """Shimmer ECG Device Manager"""
    
    def __init__(self):
        self.ser = None
        self.shim = None
        self.is_streaming = False
        self.data_buffer = deque(maxlen=2000000)
        self.lock = threading.Lock()
        self.connection_error = None
        self.dc_window = deque(maxlen=128)
    
    def get_available_ports(self):
        """Get list of available COM ports"""
        try:
            ports = []
            for port_info in list_ports.comports():
                ports.append(port_info.device)
            return sorted(ports)
        except:
            return []
    
    def connect(self, port):
        """Connect to Shimmer device"""
        if not SHIMMER_AVAILABLE:
            raise Exception("pyshimmer library not installed")
        
        try:
            if self.ser:
                try:
                    self.disconnect()
                    time.sleep(1)
                except:
                    pass
            
            self.ser = serial.Serial(port, baudrate=DEFAULT_BAUDRATE, timeout=None)
            time.sleep(1.5)
            
            self.shim = ShimmerBluetooth(self.ser)
            self.shim.initialize() 
            self.connection_error = None
            return True
        except Exception as e:
            self.connection_error = str(e)
            if self.ser:
                try:
                    self.ser.close()
                except:
                    pass
            self.ser = None
            self.shim = None
            raise e
    
    def disconnect(self):
        """Disconnect from Shimmer device"""
        try:
            if self.is_streaming and self.shim:
                try:
                    self.shim.stop_streaming()
                    time.sleep(0.5)
                except:
                    pass
        except:
            pass
        
        self.is_streaming = False
        
        try:
            if self.ser and self.ser.is_open:
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
                time.sleep(0.3)
                self.ser.close()
                time.sleep(0.3)
        except:
            pass
        
        self.shim = None
        self.ser = None
        self.connection_error = None
    
    def handler(self, pkt):
        """Data packet handler - DEBUG MODE"""
        try:
            if pkt is None:
                return
            try:
                raw_counts = pkt[CH_ECG]
                if not hasattr(self, 'packet_count'):
                    self.packet_count = 0
                
                self.packet_count += 1
                
                if self.packet_count % 10 == 0:
                    print(f"[PACKET {self.packet_count}] Raw: {raw_counts}")
                
                signed = raw_counts
                
                if signed < -8388608 or signed > 8388607:
                    print(f"[ERROR] Signed out of range: {signed}")
                    return
                
                voltage_mv = signed * SENS_MV

                if abs(voltage_mv) > 20:  # Extreme value
                    print(f"[WARNING] Extreme voltage: {voltage_mv:.2f} mV")
                
                with self.lock:
                    self.data_buffer.append(voltage_mv)
                    
                    if len(self.data_buffer) % 128 == 0:
                        recent_1s = list(self.data_buffer)[-128:]
                        mean_v = np.mean(recent_1s)
                        std_v = np.std(recent_1s)
                        print(f"[STATS] Mean: {mean_v:.4f} mV, Std: {std_v:.6f} mV")
            
            except Exception as e:
                print(f"[HANDLER ERROR] {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
        
        except:
            pass

    def start_streaming(self):
        """Safe start streaming"""
        try:
            if not self.shim:
                raise Exception("Not connected")
            try:
                self.shim.stop_streaming()
                time.sleep(0.1)
            except:
                pass

            try:
                if hasattr(self.shim, "clear_stream_callbacks"):
                    self.shim.clear_stream_callbacks()
            except:
                pass

            try:
                self.shim.add_stream_callback(self.handler)
            except:
                pass

            for retry in range(3):
                try:
                    self.shim.start_streaming()
                    time.sleep(0.2)
                    self.is_streaming = True
                    print("[STREAM] Started successfully")
                    return True
                except Exception as e:
                    print(f"[STREAM-RETRY {retry}] {e}")
                    time.sleep(0.2)

            raise Exception("Failed to start stream after retries")

        except Exception as e:
            self.is_streaming = False
            print("[START-ERROR]", e)
            raise e

        
    def stop_streaming(self):
        """Stop ECG streaming"""
        try:
            if self.is_streaming and self.shim:
                self.shim.stop_streaming()
            self.is_streaming = False
        except:
            pass
    
    def get_buffer_copy(self):
        """Get copy of current buffer"""
        with self.lock:
            return np.array(list(self.data_buffer), dtype=np.float32)
    
    def clear_buffer(self):
        """Clear data buffer"""
        with self.lock:
            self.data_buffer.clear()
    
    def is_connected(self):
        """Check if device is connected"""
        return self.ser is not None and self.ser.is_open if self.ser else False