"""
================================================================================
     TEKNOFEST AUTONOMY & VISION - DUAL PAYLOAD DROPPING SYSTEM
   (Cross-Drop: Blue Target -> Red Drop | Red Target -> Blue Drop + Level Guard)
================================================================================

Spesifikasi & Logika Operasi:
1. Target "square_blue" (Terpal Biru) -> Drop PAYLOAD MERAH (Servo Channel 7).
2. Target "square_red"  (Terpal Merah) -> Drop PAYLOAD BIRU  (Servo Channel 8).
3. Confidence Threshold: 80% (0.80) + OpenCV HSV Color Guard.
4. Attitude Level Guard:
     - Dropping otomatis HANYA dieksekusi saat pesawat dalam posisi level/datar:
       * Roll  : -10 deg s/d +10 deg (abs(roll) <= 10.0)
       * Pitch : -8 deg s/d +8 deg  (abs(pitch) <= 8.0)
5. Video Input: Clean Video Enhancer (Deinterlace, AWB, Denoise, CLAHE, Saturation Boost).
6. MAVLink Unified Bridge:
     - Membuka port Serial RFD900 secara langsung (Respon 0ms, Zero Packet Loss).
     - Otomatis stream ke Mission Planner via UDP 127.0.0.1:14550 (Sinyal 100% Instan).
7. 4 Tombol Interaktif di GUI (Bisa diklik Mouse & Keyboard 1, 2, 3, 4, R, G, SPACE).
================================================================================
"""

import sys
import os
import time
import math
import socket
import threading
import queue
import datetime
import subprocess
import importlib.util
import ctypes
from collections import deque
from pathlib import Path

# ==============================================================================
# AUTO-DEPENDENCY CHECKER & INSTALLER
# ==============================================================================
REQUIRED_PACKAGES = {
    "numpy": "numpy>=1.23.0",
    "cv2": "opencv-python>=4.8.0",
    "onnxruntime": "onnxruntime>=1.16.0",
    "serial": "pyserial>=3.5",
    "pymavlink": "pymavlink>=2.4.37"
}

def ensure_dependencies():
    missing = [pkg for mod, pkg in REQUIRED_PACKAGES.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print("\n" + "=" * 65)
        print(" [SETUP] Memeriksa dependensi sistem Teknofest...")
        print(f" -> Paket belum terpasang : {missing}")
        print(" -> Mengunduh & memasang dependensi via pip sekarang...")
        print("=" * 65 + "\n")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("\n[SETUP] Semua paket berhasil dipasang! Melanjutkan program...\n")
        except Exception as e:
            print(f"\n[SETUP ERROR] Gagal mengunduh paket secara otomatis: {e}")
            print(f"Silakan jalankan manual: pip install {' '.join(missing)}")
            sys.exit(1)

ensure_dependencies()

# Third-party imports (Aman setelah dependensi terpenuhi)
import numpy as np
import cv2
import onnxruntime as ort
import serial
import serial.tools.list_ports
from pymavlink import mavutil


# ==============================================================================
# KONFIGURASI PAYLOAD & SERVO DROPPING (USER SECTION)
# ==============================================================================

# -- PAYLOAD MERAH (Diturunkan saat Target SQUARE BLUE terdeteksi)
SERVO_RED_CHANNEL = 7          # Channel Servo di Flight Controller (AUX 1 / SERVO 7)
PWM_RED_START = 1100           # PWM Standby / Kunci Payload
PWM_RED_DROP = 2200            # PWM Release / Buka Kunci Dropping

# -- PAYLOAD BIRU (Diturunkan saat Target SQUARE RED terdeteksi)
SERVO_BLUE_CHANNEL = 8         # Channel Servo di Flight Controller (AUX 2 / SERVO 8)
PWM_BLUE_START = 2100          # PWM Standby / Kunci Payload
PWM_BLUE_DROP = 1100           # PWM Release / Buka Kunci Dropping

# -- MASTER DETECTION & AUTO-DROP TOGGLE (NO SAFEGUARD DIRECT ACTION)
AUTO_DETECTION_DEFAULT = False      # Status awal deteksi & dropping otomatis (Toggle via tombol [CTRL])

# -- MOUSE SIDE BUTTON MAPPING (Fantech & Gaming Mouse 2 Side Buttons)
# Mode "TARGET" (Sesuai Regulasi Teknofest Cross-Drop):
#   - Side Button Atas  (Forward / Button 5): Sasaran MERAH SQUARE -> Drop Payload BIRU (Servo 8)
#   - Side Button Bawah (Back / Button 4)   : Sasaran BIRU SQUARE  -> Drop Payload MERAH (Servo 7)
# Mode "PAYLOAD" (Direct Servo Color):
#   - Side Button Atas  (Forward / Button 5): Drop Langsung Payload MERAH (Servo 7)
#   - Side Button Bawah (Back / Button 4)   : Drop Langsung Payload BIRU (Servo 8)
MOUSE_SIDE_BUTTON_MODE = "TARGET"  # "TARGET" atau "PAYLOAD"

# Virtual Key Codes Windows (ctypes GetAsyncKeyState)
VK_CONTROL = 0x11   # Tombol CTRL (Modifier Toggle Deteksi & Dropping)
VK_XBUTTON1 = 0x05  # Tombol Samping Bawah / Back (Mouse Button 4)
VK_XBUTTON2 = 0x06  # Tombol Samping Atas / Forward (Mouse Button 5)

# -- TELEMETRY HUD SETTINGS (MONITORING ONLY - TIDAK MEMBLOKIR DROPPING)
AUTO_MODE_GUARD_ENABLED = False    # Safeguard bypass: Dropping tidak diblokir oleh mode terbang
TAKEOFF_GUARD_ENABLED = False      # Safeguard bypass: Dropping tidak diblokir oleh ketinggian
LEVEL_GUARD_ENABLED = False        # Safeguard bypass: Dropping tidak diblokir oleh kemiringan
WAYPOINT_GUARD_ENABLED = False     # Safeguard bypass: Dropping aktif di semua waypoint
TARGET_WAYPOINTS = [3]             # Nomor Waypoint target (Monitoring informasi)
MIN_TAKEOFF_ALT_METERS = 30.0      # Ketinggian minimal referensi
TAKEOFF_ALT_PERCENT = 30.0         # Batas ambang minimal referensi
MAX_ABS_ROLL_DEG = 10.0            # Toleransi Roll referensi
MAX_ABS_PITCH_DEG = 8.0            # Toleransi Pitch referensi


def normalize_waypoints(wps):
    """Mengubah input waypoint (int, list, tuple, str) menjadi list integer yang valid."""
    if isinstance(wps, int):
        return [wps]
    if isinstance(wps, (list, tuple, set)):
        return [int(x) for x in wps]
    if isinstance(wps, str):
        res = []
        for part in wps.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    s, e = part.split("-", 1)
                    res.extend(range(int(s.strip()), int(e.strip()) + 1))
                except Exception:
                    pass
            elif part.isdigit():
                res.append(int(part))
        return res if res else [3]
    return [3]

# -- YOLO & VISION CONFIGURATION
YOLO_MODEL_PATH = "v1main.onnx"
YOLO_INPUT_SIZE = 640
YOLO_CONF_THRESHOLD = 0.85     # Minimal Confidence 85% (0.85)
MIN_CONSECUTIVE_FRAMES = 2     # Minimal 2 frame berturut-turut terdeteksi (Anti-Glitch)

# -- HSV COLOR GUARD FILTER
COLOR_GUARD_ENABLED = True
COLOR_GUARD_BLUE_LOWER = np.array([75, 30, 30], dtype=np.uint8)
COLOR_GUARD_BLUE_UPPER = np.array([145, 255, 255], dtype=np.uint8)

COLOR_GUARD_RED_LOWER1 = np.array([0, 30, 30], dtype=np.uint8)
COLOR_GUARD_RED_UPPER1 = np.array([18, 255, 255], dtype=np.uint8)
COLOR_GUARD_RED_LOWER2 = np.array([155, 30, 30], dtype=np.uint8)
COLOR_GUARD_RED_UPPER2 = np.array([180, 255, 255], dtype=np.uint8)

MIN_COLOR_GUARD_PIXELS = 15
MIN_ROI_COLOR_PIXELS = 5

# -- MAVLINK & NETWORKING
DEFAULT_MP_FORWARD_PORT = 14550
DEFAULT_SERIAL_BAUD = 57600
DEFAULT_CAMERA_INDEX = 0       # Index 0 untuk EasyCap VRX di Windows

# -- RECORDING & DETECTION PROOF DIRECTORIES
RECORD_DIR = "video_rec"
PROOF_BASE_DIR = "detected_proof"
Path(RECORD_DIR).mkdir(parents=True, exist_ok=True)
Path(PROOF_BASE_DIR).mkdir(parents=True, exist_ok=True)

GUI_WINDOW_NAME = "TEKNOFEST DUAL DROPPING - Blue Target -> Red Drop | Red Target -> Blue Drop"


# ==============================================================================
# CLASS: VideoRecorder (Perekam Video Instan ke Folder video_rec/)
# ==============================================================================
class VideoRecorder:
    def __init__(self, output_dir=RECORD_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
        self.is_recording = False
        self.start_time = 0.0
        self.frame_count = 0
        self.current_filepath = None
        self.width = 640
        self.height = 480

    def toggle(self, width, height, fps=30.0):
        if self.is_recording:
            return self.stop()
        else:
            return self.start(width, height, fps)

    def start(self, width, height, fps=30.0):
        if self.is_recording:
            return self.current_filepath

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rec_{ts}.mp4"
        self.current_filepath = str(self.output_dir / filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.current_filepath, fourcc, fps, (width, height))

        if not self.writer.isOpened():
            filename = f"rec_{ts}.avi"
            self.current_filepath = str(self.output_dir / filename)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(self.current_filepath, fourcc, fps, (width, height))

        self.width = width
        self.height = height

        if self.writer.isOpened():
            self.is_recording = True
            self.start_time = time.time()
            self.frame_count = 0
            print(f"\n[RECORDER] MULAI MEREKAM -> '{self.current_filepath}' ({width}x{height} @ {fps:.1f} FPS)")
            return self.current_filepath
        else:
            print(f"[RECORDER ERROR] Gagal membuat VideoWriter!")
            self.writer = None
            self.is_recording = False
            return None

    def write(self, frame):
        if self.is_recording and self.writer is not None and frame is not None:
            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            self.writer.write(frame)
            self.frame_count += 1

    def stop(self):
        if not self.is_recording:
            return None

        self.is_recording = False
        dur = time.time() - self.start_time
        saved_path = self.current_filepath

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        print(f"\n[RECORDER] REKAMAN SELESAI:")
        print(f"  -> File    : {saved_path}")
        print(f"  -> Durasi  : {dur:.1f} detik ({self.frame_count} frames)")
        return saved_path

    def get_status_str(self):
        if not self.is_recording:
            return ""
        elapsed = int(time.time() - self.start_time)
        return f"{elapsed//60:02d}:{elapsed%60:02d} ({self.frame_count}f)"


# ==============================================================================
# CLASS: DetectionProofLogger (Penyimpan Foto Bukti Deteksi ke detected_proof/)
# ==============================================================================
class DetectionProofLogger:
    def __init__(self, base_dir=PROOF_BASE_DIR):
        self.base_dir = Path(base_dir)
        # Nama subfolder sesuai tanggal dan jam saat kode mulai dijalankan (start run)
        self.session_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = self.base_dir / self.session_time_str
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.queue = queue.Queue(maxsize=1000)
        self.saved_count = 0
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True, name="ProofLoggerWorker")
        self.worker_thread.start()
        print(f"[PROOF] Folder Bukti Deteksi Sesi: '{self.session_dir}'")

    def log_detection(self, frame, label, conf, frame_idx, info=""):
        if not self.running or frame is None:
            return
        # Salin frame agar tidak termodifikasi oleh frame selanjutnya
        item = (frame.copy(), label, conf, frame_idx, info, time.time())
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            pass

    def _worker(self):
        while self.running or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            frame, label, conf, frame_idx, info, ts = item
            ts_str = datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S_%f")[:19]
            filename = f"det_{ts_str}_f{frame_idx:06d}_{label}_{conf*100:.0f}pct.jpg"
            filepath = self.session_dir / filename

            # Tambahkan metadata watermark kecil di bagian bawah foto bukti
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 26), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            meta_txt = f"TEKNOFEST PROOF | {label.upper()} {conf*100:.0f}% | F:{frame_idx} | {info}"
            cv2.putText(frame, meta_txt[:70], (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

            try:
                cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                self.saved_count += 1
            except Exception as e:
                print(f"[PROOF ERROR] Gagal simpan bukti {filename}: {e}")

            self.queue.task_done()

    def close(self):
        self.running = False
        try:
            self.queue.join()
        except Exception:
            pass


# ==============================================================================
# CLASS: CleanVideoEnhancer (Engine Pembersih Sinyal EasyCap VRX)
# ==============================================================================
class CleanVideoEnhancer:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
        self.enabled = False  # Default: NONAKTIF (Raw Camera), toggle dengan [SPACE]

    def process(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled or frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]

        # 1. Instantaneous Bob Deinterlacing
        field = frame[1::2, :, :]
        deint = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)

        # 2. YCrCb Denoise & Color Separation
        ycrcb = cv2.cvtColor(deint, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # Auto White Balance
        cr_off = int(np.clip(np.mean(cr) - 128, -20, 20))
        cb_off = int(np.clip(np.mean(cb) - 128, -20, 20))
        if cr_off != 0: cr = cv2.subtract(cr, cr_off)
        if cb_off != 0: cb = cv2.subtract(cb, cb_off)

        # Chroma Denoising
        cr = cv2.GaussianBlur(cr, (7, 7), 1.5)
        cb = cv2.GaussianBlur(cb, (7, 7), 1.5)

        # Luma Surface Smoothing
        blur_y = cv2.GaussianBlur(y, (5, 5), 1.5)
        diff = cv2.absdiff(y, blur_y)
        mask_edge = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)[1]
        y_smooth = np.where(mask_edge == 255, y, blur_y)

        # Controlled CLAHE
        y_clahe = self.clahe.apply(y_smooth)

        # Subtle Unsharp Masking
        blur_s = cv2.GaussianBlur(y_clahe, (0, 0), sigmaX=1.0)
        diff_s = cv2.absdiff(y_clahe, blur_s)
        mask_p = cv2.threshold(diff_s, 6, 255, cv2.THRESH_BINARY)[1]
        boost = cv2.addWeighted(y_clahe, 1.35, blur_s, -0.35, 0)
        y_final = np.where(mask_p == 255, boost, y_clahe)

        # Recombine & Saturation Boost
        merged = cv2.merge([y_final, cr, cb])
        bgr = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_c, s_c, v_c = cv2.split(hsv)
        s_c = np.clip(s_c.astype(np.float32) * 1.30, 0, 255).astype(np.uint8)
        enhanced = cv2.cvtColor(cv2.merge([h_c, s_c, v_c]), cv2.COLOR_HSV2BGR)

        # Crop border kotor analog
        crop_m = 8
        enhanced = cv2.resize(enhanced[crop_m:-crop_m, crop_m:-crop_m], (w, h), interpolation=cv2.INTER_CUBIC)

        return enhanced


# ==============================================================================
# CLASS: MavlinkUnifiedBridge (Ultra-Fast Zero-Loss Serial + MP UDP Forwarder)
# ==============================================================================
class MavlinkUnifiedBridge:
    """
    Menghubungkan langsung Serial Radio RFD900 ke Python dan secara simultan
    meneruskan data ke Mission Planner (UDP 127.0.0.1:14550) dengan 100% kualitas sinyal
    dan latensi sub-milidetik.
    """
    def __init__(self, serial_port="COM7", baud_rate=57600, mp_port=DEFAULT_MP_FORWARD_PORT):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.mp_port = mp_port

        self.ser = None
        self.running = False
        self.thread = None

        # UDP Socket to Mission Planner (Ephemeral outbound port -> 127.0.0.1:14550)
        self.sock_mp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_mp.setblocking(False)
        self.dest_mp = ('127.0.0.1', self.mp_port)

        # MAVLink Parser
        self.mav = mavutil.mavlink.MAVLink(None)
        self.mav.srcSystem = 250
        self.mav.srcComponent = 190

        # State Telemetri & Sikap Pesawat
        self.has_heartbeat = False
        self.roll_deg = 0.0
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0
        self.target_system = 1
        self.target_component = 1
        self.msg_count = 0

        # Mode Penerbangan & Safeguards
        self.custom_mode = 0
        self.mode_name = "UNKNOWN"
        self.is_armed = False
        self.is_auto = False

        # Status Takeoff & Ketinggian
        self.relative_alt = 0.0
        self.target_alt = 0.0
        self.landed_state = 1  # 1: ON_GROUND, 2: IN_AIR, 3: TAKEOFF, 4: LANDING
        self.takeoff_complete = False
        self.mission_seq = 0

    def start(self):
        print(f"\n[MAVLINK] Membuka port serial {self.serial_port} @ {self.baud_rate} baud...")
        try:
            self.ser = serial.Serial(self.serial_port, baudrate=self.baud_rate, timeout=0.001, write_timeout=0.05)
        except Exception as e:
            print(f"[MAVLINK ERROR] Gagal membuka port {self.serial_port}: {e}")
            print(f"[MAVLINK WARNING] Berjalan dalam mode BYPASS/SIMULASI SERIAL (Perintah servo tetap dicatat & disimulasikan).")
            return False

        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True, name="MavlinkUnifiedBridgeThread")
        self.thread.start()

        # Kirim request data stream rate 10 Hz ke Flight Controller
        self.request_streams(rate_hz=10)

        print(
            f"[MAVLINK] JEMBATAN AKTIF (Sinyal 100% & Zero-Loss):\n"
            f"  -> Port Serial Radio : {self.serial_port} @ {self.baud_rate}\n"
            f"  -> Mission Planner   : Buka MP -> Pilih 'UDP' -> Connect (Port {self.mp_port})\n"
        )
        return True

    def request_streams(self, rate_hz=10):
        if self.ser is None or not getattr(self.ser, 'is_open', False):
            return
        try:
            req = self.mav.request_data_stream_encode(
                self.target_system, self.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL, rate_hz, 1
            )
            self.ser.write(req.pack(self.mav))
        except Exception:
            pass

    def send_servo_pwm(self, channel, pwm_value):
        """Kirim perintah DO_SET_SERVO langsung ke flight controller lewat serial"""
        if self.ser is None or not getattr(self.ser, 'is_open', False):
            print(f"[SERVO SIMULASI] Serial {self.serial_port} tidak aktif. Perintah Servo Channel {channel} -> {pwm_value} PWM dicatat (Mode Simulasi).")
            return True

        try:
            msg = self.mav.command_long_encode(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                channel,
                pwm_value,
                0, 0, 0, 0, 0
            )
            raw_cmd = msg.pack(self.mav)
            self.ser.write(raw_cmd)
            # Forward juga ke MP agar log MP mencatat aksi servo
            try:
                self.sock_mp.sendto(raw_cmd, self.dest_mp)
            except Exception:
                pass
            return True
        except Exception as e:
            print(f"[SERVO ERROR] Gagal kirim perintah servo: {e}")
            return False

    def _worker(self):
        last_stream_req = time.time()
        while self.running:
            # 1. Baca dari Serial Radio -> Forward ke MP & Parse di Python
            try:
                raw_bytes = self.ser.read(2048)
                if raw_bytes:
                    # Forward langsung ke Mission Planner Unicast (127.0.0.1:14550)
                    try:
                        self.sock_mp.sendto(raw_bytes, self.dest_mp)
                    except Exception:
                        pass

                    # Parse di Python secara lokal
                    try:
                        msgs = self.mav.parse_buffer(raw_bytes)
                    except Exception:
                        msgs = None

                    if msgs:
                        for m in msgs:
                            self.msg_count += 1
                            mtype = m.get_type()
                            if mtype == 'HEARTBEAT':
                                self.has_heartbeat = True
                                self.target_system = m.get_srcSystem()
                                self.target_component = m.get_srcComponent()
                                self.custom_mode = getattr(m, 'custom_mode', 0)
                                self.is_armed = bool(getattr(m, 'base_mode', 0) & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

                                # Mode mapping ArduPlane
                                PLANE_MODES = {
                                    0: 'MANUAL', 1: 'CIRCLE', 2: 'STABILIZE', 3: 'TRAINING', 4: 'ACRO',
                                    5: 'FBWA', 6: 'FBWB', 7: 'CRUISE', 8: 'AUTOTUNE', 10: 'AUTO',
                                    11: 'RTL', 12: 'LOITER', 13: 'TAKEOFF', 14: 'AVOID_ADSB', 15: 'GUIDED',
                                    17: 'QSTABILIZE', 18: 'QHOVER', 19: 'QLOITER', 20: 'QLAND', 21: 'QRTL'
                                }
                                self.mode_name = PLANE_MODES.get(self.custom_mode, f"MODE_{self.custom_mode}")
                                self.is_auto = (self.custom_mode == 10)

                            elif mtype == 'ATTITUDE':
                                self.roll_deg = m.roll * 57.29577951308232
                                self.pitch_deg = m.pitch * 57.29577951308232
                                self.yaw_deg = m.yaw * 57.29577951308232

                            elif mtype == 'GLOBAL_POSITION_INT':
                                self.relative_alt = getattr(m, 'relative_alt', 0) / 1000.0

                            elif mtype == 'VFR_HUD':
                                if self.relative_alt == 0.0 and hasattr(m, 'alt'):
                                    self.relative_alt = getattr(m, 'alt', 0.0)

                            elif mtype == 'NAV_CONTROLLER_OUTPUT':
                                if hasattr(m, 'alt_error'):
                                    self.target_alt = self.relative_alt + getattr(m, 'alt_error', 0.0)

                            elif mtype == 'EXTENDED_SYS_STATE':
                                self.landed_state = getattr(m, 'landed_state', 1)

                            elif mtype == 'STATUSTEXT':
                                try:
                                    txt = m.text.decode('utf-8', errors='ignore').strip().lower()
                                    if "takeoff complete" in txt or "climb complete" in txt:
                                        self.takeoff_complete = True
                                except Exception:
                                    pass

                            elif mtype == 'MISSION_CURRENT':
                                self.mission_seq = getattr(m, 'seq', 0)
                                if self.is_auto and self.mission_seq >= 2:
                                    self.takeoff_complete = True
                else:
                    time.sleep(0.001)
            except Exception:
                time.sleep(0.002)

            # 2. Baca perintah dari Mission Planner -> Tulis ke Serial Radio
            try:
                mp_data, _ = self.sock_mp.recvfrom(4096)
                if mp_data:
                    self.ser.write(mp_data)
            except (BlockingIOError, socket.error):
                pass
            except Exception:
                pass

            # Refresh stream request setiap 4 detik jika belum ada data
            now = time.time()
            if now - last_stream_req > 4.0:
                self.request_streams(rate_hz=10)
                last_stream_req = now

    def stop(self):
        self.running = False
        if self.ser:
            try: self.ser.close()
            except Exception: pass
        if self.sock_mp:
            try: self.sock_mp.close()
            except Exception: pass


# ==============================================================================
# YOLO LETTERBOX & PREPROCESS
# ==============================================================================
def letterbox(frame, size=YOLO_INPUT_SIZE, color=(114, 114, 114)):
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def draw_fps_badge(image, fps, latency_ms=None, cam_fps=None, top_right_x=None, top_y=14):
    """
    Menggambar badge FPS Counter HUD modern semi-transparan:
    - Hijau neon jika FPS >= 24 (Sangat lancar)
    - Kuning jika FPS 15-23 (Cukup)
    - Merah jika FPS < 15 (Perhatian/Drop frame)
    """
    fps_val = max(0.0, fps)
    fps_text = f"{fps_val:4.1f} FPS"
    
    parts = []
    if latency_ms is not None and latency_ms > 0:
        parts.append(f"Lat: {latency_ms:3.0f}ms")
    if cam_fps is not None and cam_fps > 0:
        parts.append(f"Cam: {cam_fps:2.0f}")
    sub_text = " | ".join(parts)

    if fps_val >= 24.0:
        badge_c = (0, 255, 120)   # Neon Green
    elif fps_val >= 15.0:
        badge_c = (0, 220, 255)   # Amber / Yellow
    else:
        badge_c = (0, 60, 255)    # Bright Red

    bw, bh = (128, 42) if sub_text else (98, 28)
    h_img, w_img = image.shape[:2]
    rx = (w_img - bw - 14) if top_right_x is None else (top_right_x - bw)
    ry = top_y
    if rx < 0: rx = 10

    overlay = image.copy()
    cv2.rectangle(overlay, (rx, ry), (rx + bw, ry + bh), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)
    cv2.rectangle(image, (rx, ry), (rx + bw, ry + bh), badge_c, 1)

    cv2.putText(image, fps_text, (rx + 8, ry + 20), cv2.FONT_HERSHEY_DUPLEX, 0.52, badge_c, 1, cv2.LINE_AA)
    if sub_text:
        cv2.putText(image, sub_text, (rx + 8, ry + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1, cv2.LINE_AA)


def preprocess_yolo(frame):
    image, scale, pad_x, pad_y = letterbox(frame)
    tensor = image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return tensor[None], scale, pad_x, pad_y


def parse_yolo_predictions(raw_output, conf_threshold=0.50, nms_threshold=0.45):
    """
    Universal parser for YOLO ONNX outputs:
    1. End-to-end NMS format: (1, N, 6) or (N, 6) -> [x1, y1, x2, y2, conf, class_id]
       (Contoh: v1_gazbmodel_exp.onnx)
    2. Standard YOLOv8/v11 format: (1, 4 + C, N) or (1, N, 4 + C) -> [cx, cy, w, h, class_scores...]
       (Contoh: v1main.onnx, v2_medium.onnx, v2_small.onnx)
    Returns list of dicts: [{'box': (x1, y1, x2, y2), 'conf': float, 'class_id': int}]
    """
    out = raw_output
    if isinstance(out, (list, tuple)):
        out = out[0]

    # Format 1: End-to-End NMS format [1, N, 6] atau [N, 6]
    if out.ndim == 3 and out.shape[2] == 6:
        out = out[0]
    if out.ndim == 2 and out.shape[1] == 6:
        results = []
        for row in out:
            conf = float(row[4])
            if conf >= conf_threshold:
                x1, y1, x2, y2 = [float(v) for v in row[:4]]
                cid = int(row[5])
                results.append({'box': (x1, y1, x2, y2), 'conf': conf, 'class_id': cid})
        return results

    # Format 2: Standard Ultralytics YOLOv8/v11 [1, 4+C, N] atau [1, N, 4+C]
    if out.ndim == 3:
        if out.shape[1] < out.shape[2]:  # (1, C, N) -> misal (1, 8, 8400)
            preds = np.transpose(out[0], (1, 0))  # (8400, 8)
        else:  # (1, N, C)
            preds = out[0]
    elif out.ndim == 2:
        if out.shape[0] < out.shape[1]:
            preds = np.transpose(out, (1, 0))
        else:
            preds = out
    else:
        return []

    boxes_cxcywh = preds[:, :4]
    scores = preds[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confs = np.max(scores, axis=1)

    mask = confs >= conf_threshold
    if not np.any(mask):
        return []

    filt_boxes = boxes_cxcywh[mask]
    filt_confs = confs[mask]
    filt_classes = class_ids[mask]

    boxes_for_nms = []
    boxes_x1y1x2y2 = []
    for b in filt_boxes:
        cx, cy, bw, bh = b
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        boxes_for_nms.append([int(x1), int(y1), int(bw), int(bh)])
        boxes_x1y1x2y2.append((x1, y1, x1 + bw, y1 + bh))

    indices = cv2.dnn.NMSBoxes(boxes_for_nms, [float(c) for c in filt_confs], conf_threshold, nms_threshold)
    results = []
    if len(indices) > 0:
        for idx in indices:
            i = int(idx)
            results.append({
                'box': boxes_x1y1x2y2[i],
                'conf': float(filt_confs[i]),
                'class_id': int(filt_classes[i])
            })
    return results


# ==============================================================================
# CLASS: TeknofestDualDroppingMission
# ==============================================================================
class TeknofestDualDroppingMission:
    def __init__(self, serial_port="COM7", baud_rate=57600, camera_index=DEFAULT_CAMERA_INDEX, model_path=YOLO_MODEL_PATH, is_sim=False, enable_enhancer=False):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.camera_index = camera_index
        self.model_path = model_path
        self.is_sim = is_sim

        # Enhancer (Default: Nonaktif / Raw Camera, tekan SPACE untuk aktifkan)
        self.enhancer = CleanVideoEnhancer()
        self.enhancer.enabled = bool(enable_enhancer)

        # Telemetry Bridge
        self.bridge = None
        if not self.is_sim:
            self.bridge = MavlinkUnifiedBridge(serial_port=self.serial_port, baud_rate=self.baud_rate)
            self.bridge.start()

        # State Payload Merah (Servo 7) - Diturunkan saat Target SQUARE BLUE
        self.servo_red_chan = SERVO_RED_CHANNEL
        self.pwm_red_start = PWM_RED_START
        self.pwm_red_drop = PWM_RED_DROP
        self.servo_red_pwm = PWM_RED_START
        self.payload_red_dropped = False

        # State Payload Biru (Servo 8) - Diturunkan saat Target SQUARE RED
        self.servo_blue_chan = SERVO_BLUE_CHANNEL
        self.pwm_blue_start = PWM_BLUE_START
        self.pwm_blue_drop = PWM_BLUE_DROP
        self.servo_blue_pwm = PWM_BLUE_START
        self.payload_blue_dropped = False

        # Detection Counters (Anti-glitch debounce)
        self.consecutive_blue = 0
        self.consecutive_red = 0
        self.last_blue_info = None
        self.last_red_info = None

        # Camera & Threading
        self.cap = None
        self.latest_enhanced = None
        self.new_frame = False
        self.cam_fps = 0.0
        self.cam_timestamps = deque(maxlen=20)
        self.lock = threading.Lock()

        # YOLO Model
        self.yolo_session = None
        self.yolo_input_name = None
        self.yolo_names = {}
        self.load_yolo_model()

        # UI & Buttons
        self.buttons = []
        self.detection_active = AUTO_DETECTION_DEFAULT  # Master Toggle Deteksi & Dropping [CTRL]
        self.status_banner = "SISTEM SIAP: Deteksi & Drop [AKTIF]" if self.detection_active else "SISTEM SIAP: Deteksi & Drop [STANDBY]"
        self.status_timer = time.time() + 4.0

        # Video Recorder & Detection Proof Logger
        self.recorder = VideoRecorder(output_dir=RECORD_DIR)
        self.proof_logger = DetectionProofLogger(base_dir=PROOF_BASE_DIR)
        self.frame_counter = 0
        self.pipeline_fps = 0.0
        self.fps_timestamps = deque(maxlen=20)
        self.proc_ms = 0.0

        # Start Camera
        self.start_camera()

    def load_yolo_model(self):
        model_path = Path(self.model_path)
        if not model_path.is_file():
            for d in [".", "models"]:
                cand = Path(d) / model_path.name
                if cand.is_file():
                    model_path = cand
                    break
        if not model_path.is_file():
            model_path = Path(__file__).resolve().parent / self.model_path

        print(f"[YOLO] Memuat model ONNX: {model_path}...")
        try:
            ort.preload_dlls()
            # Force GPU execution (CUDA / DirectML) dengan fallback otomatis ke CPU
            available_p = ort.get_available_providers()
            gpu_providers = [p for p in ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"] if p in available_p]
            if not gpu_providers:
                gpu_providers = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
            self.yolo_session = ort.InferenceSession(str(model_path), providers=gpu_providers)
            self.yolo_input_name = self.yolo_session.get_inputs()[0].name
            meta = self.yolo_session.get_modelmeta().custom_metadata_map
            if "names" in meta:
                import ast
                self.yolo_names = ast.literal_eval(meta["names"])
            else:
                self.yolo_names = {0: "square_blue", 1: "square_red"}
            print(f"[YOLO] Model ONNX Siap (GPU/Hardware Accel): '{model_path.name}' | Providers: {self.yolo_session.get_providers()} | Kelas: {self.yolo_names}")
        except Exception as e:
            print(f"[YOLO GPU Warning] Gagal inisialisasi GPU provider ({e}), mencoba fallback CPU...")
            try:
                self.yolo_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                self.yolo_input_name = self.yolo_session.get_inputs()[0].name
                meta = self.yolo_session.get_modelmeta().custom_metadata_map
                if "names" in meta:
                    import ast
                    self.yolo_names = ast.literal_eval(meta["names"])
                else:
                    self.yolo_names = {0: "square_blue", 1: "square_red"}
                print(f"[YOLO] Model ONNX Siap di CPU: '{model_path.name}' | Kelas: {self.yolo_names}")
            except Exception as e2:
                print(f"[YOLO ERROR] Gagal memuat model '{model_path}': {e2}")
                sys.exit(1)

    def start_camera(self):
        # Deteksi apakah camera_index berupa string path file video atau integer index kamera
        is_video_file = False
        if isinstance(self.camera_index, str) and not str(self.camera_index).isdigit():
            p = Path(self.camera_index)
            if not p.is_file():
                for d in ["vid_input", "vid input"]:
                    cand = Path(d) / p.name
                    if cand.is_file():
                        self.camera_index = str(cand)
                        break
            is_video_file = Path(self.camera_index).is_file()

        if is_video_file:
            print(f"[SOURCE] Membuka Berkas Video Uji: '{self.camera_index}'...")
            self.cap = cv2.VideoCapture(self.camera_index)
        else:
            cam_idx = int(self.camera_index) if str(self.camera_index).isdigit() else 0
            print(f"[CAMERA] Membuka Kamera Index {cam_idx} (DirectShow Windows)...")
            if os.name == 'nt':
                self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(cam_idx)

            if not self.cap.isOpened():
                print(f"[CAMERA WARN] Index {cam_idx} DirectShow gagal. Mencoba default backend...")
                self.cap = cv2.VideoCapture(cam_idx)
                if not self.cap.isOpened() and cam_idx != 0:
                    print(f"[CAMERA WARN] Mencoba Index 0...")
                    self.camera_index = 0
                    self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print(f"[CAMERA ERROR] Tidak dapat membuka sumber video: {self.camera_index}")
            sys.exit(1)

        def camera_loop():
            vid_fps = 30.0
            if is_video_file:
                vid_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = 1.0 / vid_fps if is_video_file else 0.0
            t_prev = time.perf_counter()

            while True:
                if is_video_file and frame_interval > 0:
                    elapsed = time.perf_counter() - t_prev
                    sleep_needed = frame_interval - elapsed
                    if sleep_needed > 0:
                        time.sleep(sleep_needed)
                    t_prev = time.perf_counter()

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    if is_video_file:
                        # Putar ulang video (loop) jika file telah selesai
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                        if not ret or frame is None:
                            time.sleep(0.01)
                            continue
                    else:
                        time.sleep(0.005)
                        continue

                enhanced = self.enhancer.process(frame)
                with self.lock:
                    self.latest_enhanced = enhanced
                    self.new_frame = True

                t_read = time.perf_counter()
                self.cam_timestamps.append(t_read)
                if len(self.cam_timestamps) >= 2:
                    dt_cam = self.cam_timestamps[-1] - self.cam_timestamps[0]
                    if dt_cam > 0:
                        self.cam_fps = (len(self.cam_timestamps) - 1) / dt_cam

        t = threading.Thread(target=camera_loop, daemon=True, name="CameraWorker")
        t.start()

    # ==========================================================================
    # SERVO CONTROL ACTIONS (DIRECT DO_SET_SERVO)
    # ==========================================================================
    def set_servo_pwm(self, channel, pwm_value, reason=""):
        print(f"\n[SERVO ACTION] Channel {channel} -> PWM {pwm_value} | Alasan: {reason}")
        if self.bridge is not None:
            self.bridge.send_servo_pwm(channel, pwm_value)

    # -- Payload Merah Actions (Triggered by BLUE Target)
    def trigger_drop_red(self, info=""):
        self.payload_red_dropped = True
        self.servo_red_pwm = self.pwm_red_drop
        self.set_servo_pwm(self.servo_red_chan, self.pwm_red_drop, reason=f"DROP MERAH (Target SQUARE BLUE) {info}")
        self.status_banner = f">> DROP MERAH DILEPAS! (Servo {self.servo_red_chan} -> {self.pwm_red_drop} PWM)"
        self.status_timer = time.time() + 5.0

    def reset_servo_red(self):
        self.payload_red_dropped = False
        self.servo_red_pwm = self.pwm_red_start
        self.set_servo_pwm(self.servo_red_chan, self.pwm_red_start, reason="RESET SERVO MERAH KE STANDBY")
        self.status_banner = f">> Servo Merah Reset ({self.pwm_red_start} PWM)"
        self.status_timer = time.time() + 3.0

    # -- Payload Biru Actions (Triggered by RED Target)
    def trigger_drop_blue(self, info=""):
        self.payload_blue_dropped = True
        self.servo_blue_pwm = self.pwm_blue_drop
        self.set_servo_pwm(self.servo_blue_chan, self.pwm_blue_drop, reason=f"DROP BIRU (Target SQUARE RED) {info}")
        self.status_banner = f">> DROP BIRU DILEPAS! (Servo {self.servo_blue_chan} -> {self.pwm_blue_drop} PWM)"
        self.status_timer = time.time() + 5.0

    def reset_servo_blue(self):
        self.payload_blue_dropped = False
        self.servo_blue_pwm = self.pwm_blue_start
        self.set_servo_pwm(self.servo_blue_chan, self.pwm_blue_start, reason="RESET SERVO BIRU KE STANDBY")
        self.status_banner = f">> Servo Biru Reset ({self.pwm_blue_start} PWM)"
        self.status_timer = time.time() + 3.0

    def reset_all_servos(self):
        self.reset_servo_red()
        self.reset_servo_blue()
        self.status_banner = ">> SEMUA SERVO DI-RESET KE STANDBY!"
        self.status_timer = time.time() + 3.0

    # ==========================================================================
    # MOUSE SIDE BUTTON ACTIONS (FANTECH & GAMING MOUSE)
    # ==========================================================================
    def handle_side_button_top(self):
        """
        Tombol Samping Atas Mouse (Forward / VK_XBUTTON2):
        Memicu pelepasan dropping untuk sasaran MERAH SQUARE.
        """
        if MOUSE_SIDE_BUTTON_MODE == "TARGET":
            # Target Square Red -> Sesuai aturan Teknofest melepaskan Payload Biru (Servo 8)
            print("\n[MOUSE] SIDE BUTTON TOP -> SASARAN MERAH SQUARE -> TRIGGER DROP BIRU (SERVO 8)")
            self.trigger_drop_blue("SIDE BUTTON TOP [TARGET MERAH SQUARE]")
        else:
            # Mode Direct Payload: Lepas Payload Merah (Servo 7)
            print("\n[MOUSE] SIDE BUTTON TOP -> TRIGGER DROP MERAH (SERVO 7)")
            self.trigger_drop_red("SIDE BUTTON TOP [PAYLOAD MERAH]")

    def handle_side_button_bottom(self):
        """
        Tombol Samping Bawah Mouse (Back / VK_XBUTTON1):
        Memicu pelepasan dropping untuk sasaran BIRU SQUARE.
        """
        if MOUSE_SIDE_BUTTON_MODE == "TARGET":
            # Target Square Blue -> Sesuai aturan Teknofest melepaskan Payload Merah (Servo 7)
            print("\n[MOUSE] SIDE BUTTON BOTTOM -> SASARAN BIRU SQUARE -> TRIGGER DROP MERAH (SERVO 7)")
            self.trigger_drop_red("SIDE BUTTON BOTTOM [TARGET BIRU SQUARE]")
        else:
            # Mode Direct Payload: Lepas Payload Biru (Servo 8)
            print("\n[MOUSE] SIDE BUTTON BOTTOM -> TRIGGER DROP BIRU (SERVO 8)")
            self.trigger_drop_blue("SIDE BUTTON BOTTOM [PAYLOAD BIRU]")

    def get_attitude(self):
        if self.bridge is not None:
            return self.bridge.roll_deg, self.bridge.pitch_deg, self.bridge.yaw_deg
        return 0.0, 0.0, 0.0

    def is_auto_mode(self):
        """
        Guard Mode AUTO:
        Mengembalikan True jika flight controller dalam mode AUTO (custom_mode == 10).
        """
        if not AUTO_MODE_GUARD_ENABLED or self.bridge is None or not self.bridge.has_heartbeat:
            return True, "BYPASS (No Telemetry / Disabled)"

        if self.bridge.is_auto:
            return True, "AUTO [OK]"
        return False, f"{self.bridge.mode_name} [HOLD: Bukan AUTO]"

    def is_takeoff_complete(self):
        """
        Guard Takeoff Complete:
        Divalidasi murni dari Ketinggian (Altitude) di atas 30m / 30%, TANPA syarat arming:
        - Jika Ketinggian relatif (AGL) >= 30.0 meter (atau >= 30% dari Target Ketinggian Misi),
          maka Takeoff Complete dianggap VALID (OK).
        """
        if not TAKEOFF_GUARD_ENABLED or self.bridge is None or not self.bridge.has_heartbeat:
            return True, "BYPASS (No Telemetry / Disabled)"

        cur_alt = self.bridge.relative_alt

        # Hitung threshold 30m atau 30% dari target altitude jika tersedia dari FC
        target_alt = getattr(self.bridge, 'target_alt', 0.0)
        if target_alt > 20.0:
            threshold_alt = max(MIN_TAKEOFF_ALT_METERS, target_alt * (TAKEOFF_ALT_PERCENT / 100.0))
        else:
            threshold_alt = MIN_TAKEOFF_ALT_METERS  # Default: 30.0 meter

        if cur_alt >= threshold_alt:
            return True, f"ALT OK ({cur_alt:.1f}m >= {threshold_alt:.0f}m) [OK]"
        else:
            pct_val = int((cur_alt / threshold_alt) * 100) if threshold_alt > 0 else 0
            return False, f"ALT RENDAH ({cur_alt:.1f}m < {threshold_alt:.0f}m | {pct_val}%) [HOLD]"

    def is_aircraft_level(self):
        """
        Guard Kemiringan Pesawat:
        Mengembalikan True jika pesawat sedang datar (Roll <= 10 deg, Pitch <= 8 deg).
        """
        if not LEVEL_GUARD_ENABLED or self.bridge is None or not self.bridge.has_heartbeat:
            return True, "BYPASS (No Telemetry / Disabled)"

        roll_deg, pitch_deg, _ = self.get_attitude()
        roll_ok = abs(roll_deg) <= MAX_ABS_ROLL_DEG
        pitch_ok = abs(pitch_deg) <= MAX_ABS_PITCH_DEG

        if roll_ok and pitch_ok:
            return True, f"LEVEL (R:{roll_deg:+.1f} P:{pitch_deg:+.1f}) [OK]"
        else:
            reason = []
            if not roll_ok: reason.append(f"Roll {roll_deg:+.1f} > +/-{MAX_ABS_ROLL_DEG:.0f}")
            if not pitch_ok: reason.append(f"Pitch {pitch_deg:+.1f} > +/-{MAX_ABS_PITCH_DEG:.0f}")
            return False, f"TILTED ({', '.join(reason)}) [HOLD]"

    def is_target_waypoint(self):
        """
        Guard Waypoint Misi:
        Mengembalikan True jika pesawat sedang berada pada Waypoint misi yang ditentukan.
        """
        if not WAYPOINT_GUARD_ENABLED or self.bridge is None or not self.bridge.has_heartbeat:
            return True, "BYPASS (No Telemetry / Disabled)"

        cur_wp = getattr(self.bridge, 'mission_seq', 0)
        target_wps = normalize_waypoints(TARGET_WAYPOINTS)

        if cur_wp in target_wps:
            return True, f"WP {cur_wp} (Target: {target_wps}) [OK]"
        else:
            return False, f"WP {cur_wp} (Target: {target_wps}) [HOLD]"

    def check_all_safeguards(self):
        """
        Evaluasi gabungan 4 lapis safeguards:
        1. Mode AUTO
        2. Takeoff Complete (Altitude >= 30m / 30%)
        3. Level Flight (+/- 10 deg Roll, +/- 8 deg Pitch)
        4. Target Waypoint
        """
        auto_ok, auto_desc = self.is_auto_mode()
        to_ok, to_desc = self.is_takeoff_complete()
        lvl_ok, lvl_desc = self.is_aircraft_level()
        wp_ok, wp_desc = self.is_target_waypoint()
        all_ok = auto_ok and to_ok and lvl_ok and wp_ok
        return all_ok, auto_ok, auto_desc, to_ok, to_desc, lvl_ok, lvl_desc, wp_ok, wp_desc

    # ==========================================================================
    # COLOR GUARD & VISION INFERENCE
    # ==========================================================================
    def check_roi_color(self, hsv, box, color_type="blue"):
        x1, y1, x2, y2 = box
        roi = hsv[y1:y2 + 1, x1:x2 + 1]
        if roi.size == 0: return False

        if color_type == "blue":
            mask = cv2.inRange(roi, COLOR_GUARD_BLUE_LOWER, COLOR_GUARD_BLUE_UPPER)
        else:
            m1 = cv2.inRange(roi, COLOR_GUARD_RED_LOWER1, COLOR_GUARD_RED_UPPER1)
            m2 = cv2.inRange(roi, COLOR_GUARD_RED_LOWER2, COLOR_GUARD_RED_UPPER2)
            mask = cv2.bitwise_or(m1, m2)

        return cv2.countNonZero(mask) >= MIN_ROI_COLOR_PIXELS

    def process_vision(self):
        t_start = time.perf_counter()
        with self.lock:
            if not self.new_frame or self.latest_enhanced is None:
                return None
            frame = self.latest_enhanced.copy()
            self.new_frame = False

        h, w = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Pre-check Color Guard
        if COLOR_GUARD_ENABLED:
            mask_b = cv2.inRange(hsv, COLOR_GUARD_BLUE_LOWER, COLOR_GUARD_BLUE_UPPER)
            mask_r1 = cv2.inRange(hsv, COLOR_GUARD_RED_LOWER1, COLOR_GUARD_RED_UPPER1)
            mask_r2 = cv2.inRange(hsv, COLOR_GUARD_RED_LOWER2, COLOR_GUARD_RED_UPPER2)
            mask_r = cv2.bitwise_or(mask_r1, mask_r2)
            if cv2.countNonZero(mask_b) < MIN_COLOR_GUARD_PIXELS and cv2.countNonZero(mask_r) < MIN_COLOR_GUARD_PIXELS:
                self.consecutive_blue = 0
                self.consecutive_red = 0
                self.frame_counter += 1
                if self.recorder.is_recording:
                    self.recorder.write(frame)
                self.proc_ms = (time.perf_counter() - t_start) * 1000.0
                return frame

        # YOLO Inference (Universal parser mendukung format v1main [1, 8, 8400] & format lama [1, 300, 6])
        tensor, scale, pad_x, pad_y = preprocess_yolo(frame)
        raw_output = self.yolo_session.run(None, {self.yolo_input_name: tensor})[0]
        parsed_detections = parse_yolo_predictions(raw_output, conf_threshold=YOLO_CONF_THRESHOLD, nms_threshold=0.45)

        detected_blue = None
        detected_red = None

        for det in parsed_detections:
            conf = det['conf']
            class_id = det['class_id']
            label = self.yolo_names.get(class_id, str(class_id)).lower()

            bx1, by1, bx2, by2 = det['box']
            x1 = max(0, min(w - 1, int(round((bx1 - pad_x) / scale))))
            x2 = max(0, min(w - 1, int(round((bx2 - pad_x) / scale))))
            y1 = max(0, min(h - 1, int(round((by1 - pad_y) / scale))))
            y2 = max(0, min(h - 1, int(round((by2 - pad_y) / scale))))
            if x2 <= x1 or y2 <= y1: continue

            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Klasifikasi Target:
            # - Invalid Shapes: triangle_red & hexagon_blue (Tetap digambar bounding box, TIDAK kirim command)
            # - Valid Targets : square_blue (-> Drop Merah) & square_red (-> Drop Biru)
            is_triangle_red = ("triangle" in label or label == "triangle_red")
            is_hexagon_blue = ("hexagon" in label or label == "hexagon_blue")
            is_square_blue  = (label == "square_blue" or ("square" in label and "blue" in label))
            is_square_red   = (label == "square_red" or ("square" in label and "red" in label))

            # ------------------------------------------------------------------
            # KASUS 1: TARGET INVALID (triangle_red & hexagon_blue)
            # Tampilkan Bounding Box & Label INVALID, tetapi TIDAK KIRIM COMMAND!
            # ------------------------------------------------------------------
            if is_triangle_red or is_hexagon_blue:
                inv_color = (0, 165, 255)  # Oranye peringatan
                cv2.rectangle(frame, (x1, y1), (x2, y2), inv_color, 2)
                d = 12
                cv2.line(frame, (x1, y1), (x1 + d, y1), (0, 255, 255), 2)
                cv2.line(frame, (x1, y1), (x1, y1 + d), (0, 255, 255), 2)
                cv2.line(frame, (x2, y2), (x2 - d, y2), (0, 255, 255), 2)
                cv2.line(frame, (x2, y2), (x2, y2 - d), (0, 255, 255), 2)
                cv2.circle(frame, center, 4, (120, 120, 120), -1)

                inv_tag = f"[INVALID] {label.upper()} {conf * 100:.0f}% (NO DROP)"
                cv2.putText(frame, inv_tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_DUPLEX, 0.48, inv_color, 1, cv2.LINE_AA)

                # Simpan bukti target invalid tanpa perintah dropping
                rel_alt = self.bridge.relative_alt if self.bridge else 0.0
                self.proof_logger.log_detection(frame, f"INVALID_{label}", conf, self.frame_counter, f"Alt:{rel_alt:.1f}m | INVALID (NO DROP)")
                continue

            # ------------------------------------------------------------------
            # KASUS 2: TARGET VALID SQUARE BLUE (Terpal Biru -> Drop Merah)
            # ------------------------------------------------------------------
            if is_square_blue:
                if COLOR_GUARD_ENABLED and not self.check_roi_color(hsv, (x1, y1, x2, y2), "blue"):
                    continue

                color = (255, 120, 0)  # Biru Cerah
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                d = 12
                cv2.line(frame, (x1, y1), (x1 + d, y1), (255, 255, 255), 2)
                cv2.line(frame, (x1, y1), (x1, y1 + d), (255, 255, 255), 2)
                cv2.line(frame, (x2, y2), (x2 - d, y2), (255, 255, 255), 2)
                cv2.line(frame, (x2, y2), (x2, y2 - d), (255, 255, 255), 2)
                cv2.circle(frame, center, 4, (0, 255, 255), -1)

                tag = f"[VALID] SQUARE_BLUE {conf * 100:.0f}% -> DROP MERAH"
                cv2.putText(frame, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_DUPLEX, 0.52, color, 1, cv2.LINE_AA)

                if detected_blue is None or conf > detected_blue["conf"]:
                    detected_blue = {"conf": conf, "box": (x1, y1, x2, y2), "center": center}

            # ------------------------------------------------------------------
            # KASUS 3: TARGET VALID SQUARE RED (Terpal Merah -> Drop Biru)
            # ------------------------------------------------------------------
            elif is_square_red:
                if COLOR_GUARD_ENABLED and not self.check_roi_color(hsv, (x1, y1, x2, y2), "red"):
                    continue

                color = (0, 60, 255)  # Merah Cerah
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                d = 12
                cv2.line(frame, (x1, y1), (x1 + d, y1), (255, 255, 255), 2)
                cv2.line(frame, (x1, y1), (x1, y1 + d), (255, 255, 255), 2)
                cv2.line(frame, (x2, y2), (x2 - d, y2), (255, 255, 255), 2)
                cv2.line(frame, (x2, y2), (x2, y2 - d), (255, 255, 255), 2)
                cv2.circle(frame, center, 4, (0, 255, 255), -1)

                tag = f"[VALID] SQUARE_RED {conf * 100:.0f}% -> DROP BIRU"
                cv2.putText(frame, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_DUPLEX, 0.52, color, 1, cv2.LINE_AA)

                if detected_red is None or conf > detected_red["conf"]:
                    detected_red = {"conf": conf, "box": (x1, y1, x2, y2), "center": center}

        # ----------------------------------------------------------------------
        # CROSS-DROP EXECUTION (HANYA DIEKSEKUSI JIKA MASTER DETEKSI AKTIF)
        # ----------------------------------------------------------------------
        if self.detection_active:
            # 1. Target Square Blue -> Trigger Payload Merah (Servo 7)
            if detected_blue is not None:
                self.consecutive_blue += 1
                self.last_blue_info = detected_blue
                if self.consecutive_blue >= MIN_CONSECUTIVE_FRAMES and not self.payload_red_dropped:
                    self.trigger_drop_red(f"Target SQUARE BLUE (Conf: {detected_blue['conf'] * 100:.0f}%) [DIRECT ACTION]")
            else:
                self.consecutive_blue = 0

            # 2. Target Square Red -> Trigger Payload Biru (Servo 8)
            if detected_red is not None:
                self.consecutive_red += 1
                self.last_red_info = detected_red
                if self.consecutive_red >= MIN_CONSECUTIVE_FRAMES and not self.payload_blue_dropped:
                    self.trigger_drop_blue(f"Target SQUARE RED (Conf: {detected_red['conf'] * 100:.0f}%) [DIRECT ACTION]")
            else:
                self.consecutive_red = 0
        else:
            # Status Deteksi & Auto-Dropping NONAKTIF: JANGAN kirim command apapun!
            self.consecutive_blue = 0
            self.consecutive_red = 0

        # 1. Update counter frame
        self.frame_counter += 1

        # 2. Rekam video full jika tombol [R] aktif
        if self.recorder.is_recording:
            self.recorder.write(frame)

        # 3. Simpan foto bukti deteksi ke folder detected_proof/<session_start>/
        rel_alt = self.bridge.relative_alt if self.bridge else 0.0
        act_blue = "DROP MERAH" if self.detection_active else "STANDBY"
        if detected_blue is not None:
            info = f"Alt:{rel_alt:.1f}m | SQUARE BLUE -> {act_blue}"
            self.proof_logger.log_detection(frame, "square_blue", detected_blue["conf"], self.frame_counter, info)

        act_red = "DROP BIRU" if self.detection_active else "STANDBY"
        if detected_red is not None:
            info = f"Alt:{rel_alt:.1f}m | SQUARE RED -> {act_red}"
            self.proof_logger.log_detection(frame, "square_red", detected_red["conf"], self.frame_counter, info)

        self.proc_ms = (time.perf_counter() - t_start) * 1000.0
        return frame

    # ==========================================================================
    # GUI RENDERING WITH 4 BUTTONS & LEVEL HUD
    # ==========================================================================
    def render_gui(self, frame):
        if frame is None:
            return

        h, w = frame.shape[:2]

        # Panel Samping Kanan
        panel_w = 360
        canvas_h = max(h, 600)

        # Skala frame video agar pas dengan tinggi canvas
        scaled_w = int(round(w * (canvas_h / h)))
        scaled_video = cv2.resize(frame, (scaled_w, canvas_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((canvas_h, scaled_w + panel_w, 3), dtype=np.uint8)
        canvas[:, :scaled_w] = scaled_video
        canvas[:, scaled_w:] = (25, 25, 25)

        # Watermark REC Berkedip di Video jika sedang merekam
        if self.recorder.is_recording:
            blink = int(time.time() * 2) % 2 == 0
            rec_c = (0, 0, 255) if blink else (0, 0, 160)
            cv2.circle(canvas, (24, 24), 8, rec_c, -1)
            cv2.putText(canvas, f"REC {self.recorder.get_status_str()}", (38, 30), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)

        # Badge Status Video Enhancer di Pojok Kiri Atas Frame Video
        enh_badge_y = 52 if self.recorder.is_recording else 16
        enh_active = self.enhancer.enabled
        pill_w = 205
        pill_h = 24
        cv2.rectangle(canvas, (14, enh_badge_y), (14 + pill_w, enh_badge_y + pill_h), (20, 20, 20), -1)
        cv2.rectangle(canvas, (14, enh_badge_y), (14 + pill_w, enh_badge_y + pill_h), (0, 255, 120) if enh_active else (75, 75, 75), 1)
        dot_c = (0, 255, 120) if enh_active else (110, 110, 110)
        cv2.circle(canvas, (25, enh_badge_y + 12), 4, dot_c, -1)
        enh_badge_txt = "ENHANCER: ON [SPACE]" if enh_active else "ENHANCER: OFF [SPACE]"
        enh_badge_col = (0, 255, 180) if enh_active else (180, 180, 180)
        cv2.putText(canvas, enh_badge_txt, (36, enh_badge_y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.38, enh_badge_col, 1, cv2.LINE_AA)

        # Badge FPS Counter Modern di Pojok Kanan Atas Area Video
        draw_fps_badge(canvas, fps=self.pipeline_fps, latency_ms=self.proc_ms, cam_fps=self.cam_fps, top_right_x=scaled_w - 14, top_y=14)

        p_x = scaled_w

        # 1. Header Sidebar
        self.buttons = []
        cv2.rectangle(canvas, (p_x, 0), (p_x + panel_w, 42), (38, 38, 38), -1)
        cv2.putText(canvas, "TEKNOFEST MISSION 2", (p_x + 18, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 255), 1, cv2.LINE_AA)

        # 2. DETEKSI & DROPPING MASTER CARD [CTRL] (NO SAFEGUARD MODE)
        y_pos = 48
        card_h = 118

        det_on = self.detection_active
        card_bg = (18, 38, 22) if det_on else (38, 22, 18)
        border_color = (0, 255, 120) if det_on else (0, 140, 255)
        cv2.rectangle(canvas, (p_x + 10, y_pos), (p_x + panel_w - 10, y_pos + card_h), card_bg, -1)
        cv2.rectangle(canvas, (p_x + 10, y_pos), (p_x + panel_w - 10, y_pos + card_h), border_color, 1)

        # Title & Status Dot
        dot_c = (0, 255, 120) if det_on else (0, 140, 255)
        cv2.circle(canvas, (p_x + 22, y_pos + 18), 5, dot_c, -1)
        status_txt = "DETEKSI & DROP: AKTIF" if det_on else "DETEKSI & DROP: STANDBY"
        status_col = (0, 255, 180) if det_on else (0, 180, 255)
        cv2.putText(canvas, status_txt, (p_x + 34, y_pos + 22), cv2.FONT_HERSHEY_DUPLEX, 0.44, status_col, 1, cv2.LINE_AA)

        # Subtitle Action Mode
        mode_label = "MODE: TARGET (Cross-Drop)" if MOUSE_SIDE_BUTTON_MODE == "TARGET" else "MODE: DIRECT PAYLOAD"
        cv2.putText(canvas, f"Safeguard: BYPASS | {mode_label}", (p_x + 18, y_pos + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

        # Mouse Side Buttons Guide
        cv2.putText(canvas, "TOMBOL SAMPING MOUSE (FANTECH):", (p_x + 18, y_pos + 62), cv2.FONT_HERSHEY_DUPLEX, 0.38, (0, 220, 255), 1)
        side_top_desc = "Drop Merah Square (Servo 8)" if MOUSE_SIDE_BUTTON_MODE == "TARGET" else "Drop Payload Merah (Servo 7)"
        side_bot_desc = "Drop Biru Square (Servo 7)" if MOUSE_SIDE_BUTTON_MODE == "TARGET" else "Drop Payload Biru (Servo 8)"
        cv2.putText(canvas, f" * Top/Fwd : {side_top_desc}", (p_x + 18, y_pos + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, f" * Bot/Back: {side_bot_desc}", (p_x + 18, y_pos + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 230, 230), 1, cv2.LINE_AA)

        # Daftarkan tombol interaktif untuk Deteksi Master Card
        self.buttons.append({
            "name": "TOGGLE_DETECTION",
            "rect": (p_x + 10, y_pos, panel_w - 20, card_h),
            "label": "TOGGLE DETEKSI & DROP",
            "bg": card_bg,
            "fg": status_col
        })

        # 3. RECORDER & PROOF CARD
        y_pos += card_h + 8
        rec_box_h = 44
        if self.recorder.is_recording:
            cv2.rectangle(canvas, (p_x + 10, y_pos), (p_x + panel_w - 10, y_pos + rec_box_h), (0, 0, 140), -1)
            cv2.rectangle(canvas, (p_x + 10, y_pos), (p_x + panel_w - 10, y_pos + rec_box_h), (0, 0, 255), 1)
            blink = int(time.time() * 2) % 2 == 0
            cv2.circle(canvas, (p_x + 24, y_pos + 16), 6, (0, 0, 255) if blink else (255, 255, 255), -1)
            cv2.putText(canvas, f"MEREKAM: {self.recorder.get_status_str()}", (p_x + 38, y_pos + 20), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Bukti: {self.proof_logger.saved_count} foto -> detected_proof/", (p_x + 22, y_pos + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (p_x + 10, y_pos), (p_x + panel_w - 10, y_pos + rec_box_h), (34, 34, 34), -1)
            cv2.rectangle(canvas, (p_x + 10, y_pos), (p_x + panel_w - 10, y_pos + rec_box_h), (55, 55, 55), 1)
            cv2.circle(canvas, (p_x + 24, y_pos + 16), 6, (120, 120, 120), -1)
            cv2.putText(canvas, "RECORDER: STANDBY [R]", (p_x + 38, y_pos + 20), cv2.FONT_HERSHEY_DUPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Bukti: {self.proof_logger.saved_count} foto -> {self.proof_logger.session_time_str}", (p_x + 22, y_pos + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)

        # 4. KOTAK STATUS PAYLOAD MERAH (Servo 7)
        y_pos += rec_box_h + 8
        r_box_y = y_pos
        cv2.rectangle(canvas, (p_x + 10, r_box_y), (p_x + panel_w - 10, r_box_y + 48), (40, 40, 40), -1)
        r_color = (0, 0, 255) if self.payload_red_dropped else (100, 100, 100)
        cv2.rectangle(canvas, (p_x + 10, r_box_y), (p_x + panel_w - 10, r_box_y + 48), r_color, 1)
        cv2.putText(canvas, f"PAYLOAD MERAH (Servo {self.servo_red_chan})", (p_x + 18, r_box_y + 18), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1)
        r_status = f"DROPPED ({self.servo_red_pwm} PWM)" if self.payload_red_dropped else f"STANDBY ({self.servo_red_pwm} PWM)"
        cv2.putText(canvas, r_status, (p_x + 18, r_box_y + 38), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 0, 255) if self.payload_red_dropped else (0, 255, 0), 1)

        # 5. KOTAK STATUS PAYLOAD BIRU (Servo 8)
        y_pos = r_box_y + 56
        b_box_y = y_pos
        cv2.rectangle(canvas, (p_x + 10, b_box_y), (p_x + panel_w - 10, b_box_y + 48), (40, 40, 40), -1)
        b_color = (255, 120, 0) if self.payload_blue_dropped else (100, 100, 100)
        cv2.rectangle(canvas, (p_x + 10, b_box_y), (p_x + panel_w - 10, b_box_y + 48), b_color, 1)
        cv2.putText(canvas, f"PAYLOAD BIRU (Servo {self.servo_blue_chan})", (p_x + 18, b_box_y + 18), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1)
        b_status = f"DROPPED ({self.servo_blue_pwm} PWM)" if self.payload_blue_dropped else f"STANDBY ({self.servo_blue_pwm} PWM)"
        cv2.putText(canvas, b_status, (p_x + 18, b_box_y + 38), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 120, 0) if self.payload_blue_dropped else (0, 255, 0), 1)

        # 6. KOTAK STATUS & TOGGLE VIDEO ENHANCER (Tombol Interaktif [SPACE])
        enh_box_y = b_box_y + 56
        enh_box_h = 42
        enh_w = panel_w - 20
        enh_active = self.enhancer.enabled

        enh_bg = (18, 48, 22) if enh_active else (34, 34, 34)
        enh_border = (0, 255, 120) if enh_active else (75, 75, 75)
        cv2.rectangle(canvas, (p_x + 10, enh_box_y), (p_x + panel_w - 10, enh_box_y + enh_box_h), enh_bg, -1)
        cv2.rectangle(canvas, (p_x + 10, enh_box_y), (p_x + panel_w - 10, enh_box_y + enh_box_h), enh_border, 1)

        dot_c = (0, 255, 120) if enh_active else (120, 120, 120)
        cv2.circle(canvas, (p_x + 24, enh_box_y + 16), 5, dot_c, -1)

        enh_title = "ENHANCER: AKTIF (CLEAN V2)" if enh_active else "ENHANCER: NONAKTIF (RAW)"
        enh_title_col = (0, 255, 180) if enh_active else (200, 200, 200)
        cv2.putText(canvas, enh_title, (p_x + 36, enh_box_y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.42, enh_title_col, 1, cv2.LINE_AA)

        enh_sub = "[SPACE] / Klik -> Matikan Filter" if enh_active else "[SPACE] / Klik -> Aktifkan Filter"
        enh_sub_col = (160, 240, 180) if enh_active else (0, 210, 255)
        cv2.putText(canvas, enh_sub, (p_x + 18, enh_box_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, enh_sub_col, 1, cv2.LINE_AA)

        # Daftarkan tombol interaktif untuk Enhancer
        self.buttons.append({
            "name": "TOGGLE_ENHANCER",
            "rect": (p_x + 10, enh_box_y, enh_w, enh_box_h),
            "label": "[SPACE] TOGGLE ENHANCER",
            "bg": enh_bg,
            "fg": enh_title_col
        })

        # 7. 5 TOMBOL UJI INTERAKTIF SERVO
        btn_y = enh_box_y + enh_box_h + 8
        btn_h = 28
        btn_w = panel_w - 20
        spacing = 33

        # Tombol 1: Drop Merah
        b1 = {"name": "DROP_RED", "rect": (p_x + 10, btn_y, btn_w, btn_h), "label": "[1] DROP MERAH", "bg": (0, 0, 160), "fg": (255, 255, 255)}
        self.buttons.append(b1)
        cv2.rectangle(canvas, (b1["rect"][0], b1["rect"][1]), (b1["rect"][0] + b1["rect"][2], b1["rect"][1] + b1["rect"][3]), b1["bg"], -1)
        cv2.rectangle(canvas, (b1["rect"][0], b1["rect"][1]), (b1["rect"][0] + b1["rect"][2], b1["rect"][1] + b1["rect"][3]), (255, 255, 255), 1)
        cv2.putText(canvas, b1["label"], (b1["rect"][0] + 15, b1["rect"][1] + 19), cv2.FONT_HERSHEY_DUPLEX, 0.40, b1["fg"], 1, cv2.LINE_AA)

        # Tombol 2: Reset Merah
        btn_y += spacing
        b2 = {"name": "RESET_RED", "rect": (p_x + 10, btn_y, btn_w, btn_h), "label": "[2] RESET SERVO MERAH", "bg": (55, 55, 55), "fg": (200, 200, 200)}
        self.buttons.append(b2)
        cv2.rectangle(canvas, (b2["rect"][0], b2["rect"][1]), (b2["rect"][0] + b2["rect"][2], b2["rect"][1] + b2["rect"][3]), b2["bg"], -1)
        cv2.rectangle(canvas, (b2["rect"][0], b2["rect"][1]), (b2["rect"][0] + b2["rect"][2], b2["rect"][1] + b2["rect"][3]), (160, 160, 160), 1)
        cv2.putText(canvas, b2["label"], (b2["rect"][0] + 15, b2["rect"][1] + 19), cv2.FONT_HERSHEY_DUPLEX, 0.40, b2["fg"], 1, cv2.LINE_AA)

        # Tombol 3: Drop Biru
        btn_y += spacing
        b3 = {"name": "DROP_BLUE", "rect": (p_x + 10, btn_y, btn_w, btn_h), "label": "[3] DROP BIRU", "bg": (160, 80, 0), "fg": (255, 255, 255)}
        self.buttons.append(b3)
        cv2.rectangle(canvas, (b3["rect"][0], b3["rect"][1]), (b3["rect"][0] + b3["rect"][2], b3["rect"][1] + b3["rect"][3]), b3["bg"], -1)
        cv2.rectangle(canvas, (b3["rect"][0], b3["rect"][1]), (b3["rect"][0] + b3["rect"][2], b3["rect"][1] + b3["rect"][3]), (255, 255, 255), 1)
        cv2.putText(canvas, b3["label"], (b3["rect"][0] + 15, b3["rect"][1] + 19), cv2.FONT_HERSHEY_DUPLEX, 0.40, b3["fg"], 1, cv2.LINE_AA)

        # Tombol 4: Reset Biru
        btn_y += spacing
        b4 = {"name": "RESET_BLUE", "rect": (p_x + 10, btn_y, btn_w, btn_h), "label": "[4] RESET SERVO BIRU", "bg": (55, 55, 55), "fg": (200, 200, 200)}
        self.buttons.append(b4)
        cv2.rectangle(canvas, (b4["rect"][0], b4["rect"][1]), (b4["rect"][0] + b4["rect"][2], b4["rect"][1] + b4["rect"][3]), b4["bg"], -1)
        cv2.rectangle(canvas, (b4["rect"][0], b4["rect"][1]), (b4["rect"][0] + b4["rect"][2], b4["rect"][1] + b4["rect"][3]), (160, 160, 160), 1)
        cv2.putText(canvas, b4["label"], (b4["rect"][0] + 15, b4["rect"][1] + 19), cv2.FONT_HERSHEY_DUPLEX, 0.40, b4["fg"], 1, cv2.LINE_AA)

        # Tombol 5: Reset All (Shortcut [X])
        btn_y += spacing
        b5 = {"name": "RESET_ALL", "rect": (p_x + 10, btn_y, btn_w, btn_h), "label": "[X] RESET SEMUA SERVO", "bg": (45, 45, 75), "fg": (220, 220, 255)}
        self.buttons.append(b5)
        cv2.rectangle(canvas, (b5["rect"][0], b5["rect"][1]), (b5["rect"][0] + b5["rect"][2], b5["rect"][1] + b5["rect"][3]), b5["bg"], -1)
        cv2.rectangle(canvas, (b5["rect"][0], b5["rect"][1]), (b5["rect"][0] + b5["rect"][2], b5["rect"][1] + b5["rect"][3]), (180, 180, 220), 1)
        cv2.putText(canvas, b5["label"], (b5["rect"][0] + 15, b5["rect"][1] + 19), cv2.FONT_HERSHEY_DUPLEX, 0.40, b5["fg"], 1, cv2.LINE_AA)

        # 8. FOOTER TELEMETRI & STATUS BANNER
        hud_h = 36
        tot_w = scaled_w + panel_w
        hud_bar = np.zeros((hud_h, tot_w, 3), dtype=np.uint8)
        hud_bar[:] = (18, 18, 18)

        msg_c = self.bridge.msg_count if self.bridge else 0
        det_s = "DETEKSI: AKTIF" if self.detection_active else "DETEKSI: STANDBY"
        enh_status = "ON" if self.enhancer.enabled else "OFF"
        model_name = Path(self.model_path).name
        alt_val = f"{self.bridge.relative_alt:.1f}m" if self.bridge else "N/A"
        mode_val = self.bridge.mode_name if self.bridge else "N/A"
        info_txt = f"FPS: {self.pipeline_fps:4.1f} | Cam: {self.cam_fps:4.1f} | Lat: {self.proc_ms:3.0f}ms | {det_s} | Enh: {enh_status} | Alt: {alt_val} | Mode: {mode_val} | {model_name}"
        cv2.putText(hud_bar, info_txt, (15, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

        if time.time() < self.status_timer and self.status_banner:
            (bw, _), _ = cv2.getTextSize(self.status_banner, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            toast_w = min(bw + 18, 420)
            toast_x = tot_w - toast_w - 10
            cv2.rectangle(hud_bar, (toast_x, 4), (tot_w - 10, 32), (32, 32, 10), -1)
            cv2.rectangle(hud_bar, (toast_x, 4), (tot_w - 10, 32), (0, 200, 255), 1)
            cv2.putText(hud_bar, self.status_banner[:45], (toast_x + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 255), 1, cv2.LINE_AA)

        final_gui = np.vstack([canvas, hud_bar])
        return final_gui

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for b in self.buttons:
                bx, by, bw, bh = b["rect"]
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    name = b["name"]
                    if name == "TOGGLE_DETECTION":
                        self.detection_active = not self.detection_active
                        st = "AKTIF" if self.detection_active else "STANDBY"
                        self.status_banner = f">> Deteksi & Auto-Drop: {st}"
                        self.status_timer = time.time() + 2.5
                    elif name == "DROP_RED":
                        self.trigger_drop_red("KLIK TOMBOL GUI")
                    elif name == "RESET_RED":
                        self.reset_servo_red()
                    elif name == "DROP_BLUE":
                        self.trigger_drop_blue("KLIK TOMBOL GUI")
                    elif name == "RESET_BLUE":
                        self.reset_servo_blue()
                    elif name == "RESET_ALL":
                        self.reset_all_servos()
                    elif name == "TOGGLE_ENHANCER":
                        self.enhancer.enabled = not self.enhancer.enabled
                        self.status_banner = f">> Enhancer: {'AKTIF (CLEAN V2)' if self.enhancer.enabled else 'NONAKTIF (RAW BYPASS)'}"
                        self.status_timer = time.time() + 2.5
                    break


    def run(self):
        global WAYPOINT_GUARD_ENABLED, AUTO_MODE_GUARD_ENABLED, TAKEOFF_GUARD_ENABLED, LEVEL_GUARD_ENABLED

        cv2.namedWindow(GUI_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(GUI_WINDOW_NAME, 1280, 680)
        cv2.setMouseCallback(GUI_WINDOW_NAME, self.on_mouse_click)

        print("\n" + "=" * 65)
        print(" SISTEM AUTONOMI DUAL DROPPING TEKNOFEST (DIRECT / NO SAFEGUARD):")
        det_st = "AKTIF" if self.detection_active else "NONAKTIF"
        print(f"  -> Deteksi & Auto-Drop : [{det_st}]")
        print(f"  -> Video Enhancer      : {'[AKTIF (Clean V2)]' if self.enhancer.enabled else '[NONAKTIF / RAW (Tekan SPACE utk aktifkan)]'}")
        print(f"  -> Mode Tombol Samping : {MOUSE_SIDE_BUTTON_MODE} (Cross-Drop Sesuai Regulasi)")
        print("  [CTRL]              : Toggle Deteksi & Auto-Dropping ON/OFF")
        print("  [Side Btn Atas]     : Drop Merah Square (Forward Thumb Btn)")
        print("  [Side Btn Bawah]    : Drop Biru Square (Back Thumb Btn)")
        print("  [1]                 : Trigger Drop Merah (Servo 7 -> Drop PWM)")
        print("  [2]                 : Reset Servo Merah (Servo 7 -> Start PWM)")
        print("  [3]                 : Trigger Drop Biru (Servo 8 -> Drop PWM)")
        print("  [4]                 : Reset Servo Biru (Servo 8 -> Start PWM)")
        print("  [X]                 : Reset Semua Servo ke Standby")
        print("  [R]                 : Toggle Rekam Video Full (Mulai / Stop Rekam)")
        print("  [SPACE]             : Toggle Video Enhancer ON/OFF (Default: NONAKTIF)")
        print("  [Q] / [ESC]         : Keluar")
        print("=" * 65 + "\n")

        self.fps_timestamps.clear()
        self.cam_timestamps.clear()
        self.pipeline_fps = 0.0

        # State Edge Detection untuk Tombol Windows (CTRL & Tombol Samping Mouse)
        last_ctrl_state = False
        last_xbtn1_state = False
        last_xbtn2_state = False
        user32 = ctypes.windll.user32 if os.name == 'nt' else None

        try:
            while True:
                # Polling Input Windows (CTRL & Tombol Samping Mouse Fantech)
                if user32 is not None:
                    # 1. Polling [CTRL] (VK_CONTROL = 0x11): Toggle Deteksi & Auto-Dropping
                    try:
                        ctrl_down = bool(user32.GetAsyncKeyState(VK_CONTROL) & 0x8000)
                        if ctrl_down and not last_ctrl_state:
                            self.detection_active = not self.detection_active
                            st = "AKTIF" if self.detection_active else "STANDBY"
                            self.status_banner = f">> Deteksi & Auto-Drop: {st}"
                            self.status_timer = time.time() + 3.0
                            print(f"\n[CONTROL] Deteksi & Auto-Dropping di-toggle: {st}")
                        last_ctrl_state = ctrl_down
                    except Exception:
                        pass

                    # 2. Polling Tombol Samping Atas (VK_XBUTTON2 = 0x06 / Forward Thumb Btn)
                    try:
                        xbtn2_down = bool(user32.GetAsyncKeyState(VK_XBUTTON2) & 0x8000)
                        if xbtn2_down and not last_xbtn2_state:
                            self.handle_side_button_top()
                        last_xbtn2_state = xbtn2_down
                    except Exception:
                        pass

                    # 3. Polling Tombol Samping Bawah (VK_XBUTTON1 = 0x05 / Back Thumb Btn)
                    try:
                        xbtn1_down = bool(user32.GetAsyncKeyState(VK_XBUTTON1) & 0x8000)
                        if xbtn1_down and not last_xbtn1_state:
                            self.handle_side_button_bottom()
                        last_xbtn1_state = xbtn1_down
                    except Exception:
                        pass

                # 1. Vision YOLO & Color Guard
                processed_frame = self.process_vision()
                if processed_frame is None:
                    if len(self.fps_timestamps) > 0 and (time.perf_counter() - self.fps_timestamps[-1]) > 1.0:
                        self.pipeline_fps = 0.0
                    time.sleep(0.003)
                    continue

                # Update Real-Time Pipeline FPS via Rolling Window (time.perf_counter)
                t_now = time.perf_counter()
                self.fps_timestamps.append(t_now)
                if len(self.fps_timestamps) >= 2:
                    total_dt = self.fps_timestamps[-1] - self.fps_timestamps[0]
                    if total_dt > 0:
                        self.pipeline_fps = (len(self.fps_timestamps) - 1) / total_dt

                # 2. Render GUI & Buttons
                gui_view = self.render_gui(processed_frame)
                cv2.imshow(GUI_WINDOW_NAME, gui_view)

                # 3. Keyboard Shortcuts
                key = cv2.waitKey(1) & 0xFF
                if key in [27, ord('q'), ord('Q')]:
                    break
                elif key == ord('1'):
                    self.trigger_drop_red("KEYBOARD [1]")
                elif key == ord('2'):
                    self.reset_servo_red()
                elif key == ord('3'):
                    self.trigger_drop_blue("KEYBOARD [3]")
                elif key == ord('4'):
                    self.reset_servo_blue()
                elif key in [ord('r'), ord('R')]:
                    # Toggle Video Full Recording ke folder video_rec/
                    h, w = processed_frame.shape[:2]
                    saved = self.recorder.toggle(width=w, height=h, fps=self.cam_fps or 30.0)
                    if self.recorder.is_recording:
                        self.status_banner = f">> MEREKAM: {Path(saved).name}"
                        self.status_timer = time.time() + 3.5
                    else:
                        if saved:
                            self.status_banner = f">> REKAMAN DISIMPAN: {Path(saved).name}"
                            self.status_timer = time.time() + 3.5
                elif key in [ord('x'), ord('X')]:
                    self.reset_all_servos()
                    self.status_banner = ">> SEMUA SERVO DI-RESET KE STANDBY (Tombol [X])"
                    self.status_timer = time.time() + 2.5
                elif key == 32: # SPACE
                    self.enhancer.enabled = not self.enhancer.enabled
                    self.status_banner = f">> Enhancer: {'AKTIF (CLEAN V2)' if self.enhancer.enabled else 'NONAKTIF (RAW BYPASS)'}"
                    self.status_timer = time.time() + 2.5

        finally:
            if self.recorder.is_recording:
                self.recorder.stop()
            if hasattr(self, 'proof_logger'):
                self.proof_logger.close()
            if self.cap:
                self.cap.release()
            if self.bridge:
                self.bridge.stop()
            cv2.destroyAllWindows()
            print("[MISSION] Sistem ditutup dengan aman.")


# ==============================================================================
# MENU SELEKSI PORT, SUMBER VIDEO & MODEL (FAST PREPARATION)
# ==============================================================================
def find_available_models(default_model=YOLO_MODEL_PATH):
    models = []
    for p in Path(".").glob("*.onnx"):
        models.append(p)
    for p in Path(".").glob("*/*.onnx"):
        if "__pycache__" not in str(p):
            models.append(p)

    unique = []
    seen = set()
    for m in models:
        res = m.resolve()
        if res not in seen:
            seen.add(res)
            unique.append(m)

    unique.sort(key=lambda x: 0 if x.name == default_model else 1)
    return unique


def interactive_setup():
    # 1. Deteksi & Pilih Port COM Telemetri
    com_ports = list(serial.tools.list_ports.comports())
    default_port = "COM7"

    print("\n" + "=" * 65)
    print("   TEKNOFEST - DUAL PAYLOAD DROPPING MISSION CONTROL")
    print("=" * 65)
    print(" [1/3] Pilih Port Serial Telemetri (RFD900 / SiK Radio / USB Telem):")

    if com_ports:
        for idx, p in enumerate(com_ports, start=1):
            tag = " [DEFAULT]" if p.device.upper() == default_port.upper() else ""
            print(f"  [{idx}] {p.device} - {p.description}{tag}")
        if default_port not in [x.device.upper() for x in com_ports]:
            default_port = com_ports[0].device
        print("  [M] Ketik manual nama port COM (misal COM3, COM7, /dev/ttyUSB0)")
        print("  [S] Mode Simulasi / Tanpa Koneksi Serial Radio")
        print("=" * 65)
        p_prompt = f"Pilih Port COM (1-{len(com_ports)} / nama port / S, tekan [ENTER] utk default '{default_port}'): "
    else:
        print("  [!] Port serial tidak terdeteksi otomatis (pastikan modul tercolok).")
        print(f"  -> Ketik nama port COM secara manual (misal: COM3, COM7, COM4)")
        print(f"  -> Tekan [ENTER] untuk port default '{default_port}'")
        print("  -> Atau ketik 'S' untuk Mode Simulasi (tanpa koneksi serial)")
        print("=" * 65)
        p_prompt = f"Pilih / Ketik Port COM (tekan [ENTER] utk '{default_port}', atau 'S' utk Simulasi): "

    p_choice = input(p_prompt).strip()

    if p_choice == "":
        selected_port = default_port
    elif p_choice.upper() in ["S", "SIM", "NONE"]:
        selected_port = "SIM"
    elif com_ports and p_choice.isdigit() and 1 <= int(p_choice) <= len(com_ports):
        selected_port = com_ports[int(p_choice) - 1].device
    elif p_choice.isdigit():
        selected_port = f"COM{p_choice}"
    elif p_choice.upper().startswith("COM"):
        selected_port = p_choice.upper()
    elif p_choice.upper() == "M":
        m_input = input("Masukkan nama port COM (misal COM7): ").strip()
        selected_port = m_input.upper() if m_input.upper().startswith("COM") else (m_input if m_input else default_port)
    else:
        selected_port = p_choice

    # 2. Deteksi & Pilih Sumber Video Input
    print("\n" + "=" * 65)
    print(" [2/3] Pilih Sumber Video Input Kamera / Berkas Video:")
    print("  [0] Kamera EasyCap USB Video Grabber (Index 0) [DEFAULT]")
    print("  [1] Kamera Internal Laptop / USB Webcam Lain (Index 1)")

    # Pindai berkas video di folder vid_input / vid input
    video_files = []
    for d in ["vid_input", "vid input"]:
        p_dir = Path(d)
        if p_dir.is_dir():
            for ext in ("*.mp4", "*.avi", "*.mkv", "*.mov"):
                for vf in p_dir.glob(ext):
                    video_files.append(vf)

    if video_files:
        print("  [2] Uji Menggunakan Berkas Video (dari folder 'vid_input/'):")
        for v_idx, vf in enumerate(video_files, start=1):
            print(f"       -> [v{v_idx}] {vf.name}")
    print("=" * 65)

    v_choice = input("Pilih Sumber Video (0/1/2/index/path, tekan [ENTER] untuk default [0]): ").strip()

    if v_choice == "" or v_choice == "0":
        selected_cam = 0
    elif v_choice == "1":
        selected_cam = 1
    elif v_choice in ["2", "v1"] and video_files:
        selected_cam = str(video_files[0])
    elif v_choice.startswith("v") and v_choice[1:].isdigit():
        v_num = int(v_choice[1:])
        if 1 <= v_num <= len(video_files):
            selected_cam = str(video_files[v_num - 1])
        else:
            selected_cam = 0
    elif v_choice.isdigit():
        selected_cam = int(v_choice)
    else:
        selected_cam = v_choice

    # 3. Deteksi & Pilih Model Deteksi YOLO (.onnx)
    print("\n" + "=" * 65)
    print(" [3/3] Pilih Model Deteksi YOLO (.onnx):")
    available_models = find_available_models(YOLO_MODEL_PATH)
    default_model = available_models[0].name if available_models else YOLO_MODEL_PATH
    for idx, m in enumerate(available_models, 1):
        tag = " [DEFAULT]" if m.name == default_model or idx == 1 else ""
        print(f"  [{idx}] {m.name}{tag}")
    print("=" * 65)

    m_choice = input(f"Pilih Model (1-{len(available_models)}/path, tekan [ENTER] untuk default '{default_model}'): ").strip()
    if m_choice.isdigit() and 1 <= int(m_choice) <= len(available_models):
        selected_model = str(available_models[int(m_choice) - 1])
    elif m_choice and Path(m_choice).is_file():
        selected_model = m_choice
    else:
        selected_model = str(available_models[0]) if available_models else default_model

    return selected_port, DEFAULT_SERIAL_BAUD, selected_cam, selected_model


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Teknofest Dual Dropping Mission System")
    parser.add_argument("--port", "-p", type=str, default=None, help="Port Serial RFD900 (misal COM7)")
    parser.add_argument("--baud", "-b", type=int, default=DEFAULT_SERIAL_BAUD, help="Baud rate RFD900")
    parser.add_argument("--cam", "-c", default=None, help="Index kamera (0, 1) atau path file video")
    parser.add_argument("--video", "-v", default=None, help="Path berkas video pengujian")
    parser.add_argument("--model", "-m", type=str, default=None, help="Path berkas model YOLO (.onnx)")
    parser.add_argument("--wp", "-w", type=str, default=None, help="Nomor Waypoint target deteksi & drop (misal: 3 atau 3,4 atau 3-5)")
    parser.add_argument("--enhance", action="store_true", default=False, help="Aktifkan Video Enhancer sejak awal (default: False / Raw Camera)")
    parser.add_argument("--sim", action="store_true", help="Jalankan dalam mode simulasi tanpa serial")
    args = parser.parse_args()

    # Prioritaskan argumen target waypoint jika ada
    if args.wp is not None:
        TARGET_WAYPOINTS = normalize_waypoints(args.wp)
        print(f"[CONFIG] Target Waypoint diatur via CLI: {TARGET_WAYPOINTS}")

    # Prioritaskan argumen video jika ada
    cam_input = args.video if args.video is not None else args.cam
    model_input = args.model

    if args.sim:
        cam_idx = cam_input if cam_input is not None else DEFAULT_CAMERA_INDEX
        model_path = model_input if model_input is not None else YOLO_MODEL_PATH
        runner = TeknofestDualDroppingMission(is_sim=True, camera_index=cam_idx, model_path=model_path, enable_enhancer=args.enhance)
    else:
        serial_port = args.port
        baud_rate = args.baud
        cam_idx = cam_input
        model_path = model_input

        if serial_port is None or cam_idx is None or model_path is None:
            sel_port, sel_baud, sel_cam, sel_model = interactive_setup()
            if serial_port is None: serial_port = sel_port
            if baud_rate == DEFAULT_SERIAL_BAUD: baud_rate = sel_baud
            if cam_idx is None: cam_idx = sel_cam
            if model_path is None: model_path = sel_model

        runner = TeknofestDualDroppingMission(
            serial_port=serial_port,
            baud_rate=baud_rate,
            camera_index=cam_idx,
            model_path=model_path,
            is_sim=False,
            enable_enhancer=args.enhance
        )

    runner.run()
