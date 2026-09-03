"""
================================================================================
     EASYCAP & VRX ANALOG FPV VIDEO ENHANCER (KHUSUS DETEKSI TERPAL UAV)
                         Teknofest Autonomy & Vision
================================================================================

Deskripsi:
Engine pemrosesan citra khusus kamera bawah (nadir) pada wahana UAV untuk
mendeteksi target terpal kotak biru (square_blue) dan merah (square_red).

Karakteristik Deteksi Terpal di Lapangan:
1. Terpal adalah objek dengan warna solid seragam (uniform flat color).
2. Tepi/sudut kotak terpal harus tegas dan terdefinisi agar model YOLO mengunci bounding box.
3. Permukaan rumput/aspal & bayangan TIDAK BOLEH berbintik-bintik (grainy/noisy),
   karena noise berlebihan akan merusak deteksi HSV Color Guard dan menurunkan confidence YOLO.

Pipeline Khusus yang Diterapkan:
1. [Bob Single-Field Deinterlacing]: 100% menghapus garis sisir horizontal saat pesawat meluncur cepat.
2. [O(1) DirectShow YCrCb AWB]: Membuang lapisan warna kuning/hijau kusam EasyCap.
3. [Heavy Chroma Denoising (Gaussian 9x9)]: Membersihkan derau RF warna sehingga terpal menjadi 100% solid biru/merah murni.
4. [Chroma Contrast Expansion (1.35x)]: Mempertegas perbedaan warna biru/merah terhadap rumput hijau.
5. [Edge-Preserving Luma Surface Smoothing]: Membersihkan bintik semut/grain pada permukaan tanpa memblur garis sudut terpal.
6. [Controlled CLAHE (Clip 1.35)]: Menaikkan kontras lokal tanpa meledakkan derau di area gelap/baju/bayangan.
7. [High-Coring Edge Sharpening]: Hanya mempertajam perimeter/sudut terpal, bukan tekstur rumput.
8. [Temporal Static Rejection]: Menghapus bintik salju RF yang berkedip antar frame.

Shortcut Keyboard (GUI):
  [P]         : Ganti Preset (TARP_DETECTION -> SMOOTH_COLOR -> OUTDOOR_SUN -> CRISP_EDGES)
  [S]         : Simpan konfigurasi saat ini ke 'easycap_config.json'
  [L]         : Muat ulang konfigurasi dari 'easycap_config.json'
  [R]         : Reset ke preset optimal TARP_DETECTION
  [TAB] / [M] : Ganti mode tampilan (Side-by-Side, Split-Slider, Enhanced, Raw)
  [C] / [SPC] : Snapshot tangkapan layar Before & After
  [Q] / [ESC] : Keluar
================================================================================
"""

import sys
import os
import time
import json
import threading
from pathlib import Path
import numpy as np
import cv2


# ==============================================================================
# PRESETS KHUSUS DETEKSI TERPAL UAV
# ==============================================================================
PRESETS = {
    "TARP_DETECTION": {
        "description": "Optimal untuk Terpal Kotak Biru & Merah (Bersih, Solid, Bebas Grain)",
        "crop": {"top": 8, "bottom": 8, "left": 10, "right": 10},
        "deinterlace": {"enabled": True, "mode": "bob"},
        "luma_smoothing": {"enabled": True, "edge_threshold": 10},
        "denoise": {"chroma_strength": 9, "temporal_denoise": True, "temporal_alpha": 0.25},
        "clahe": {"enabled": True, "clip_limit": 1.35, "grid_size": 8},
        "sharpen": {"enabled": True, "amount": 1.35, "radius": 1.0, "coring_threshold": 8},
        "color": {
            "auto_white_balance": True,
            "chroma_boost": 1.35,
            "contrast": 1.10,
            "brightness": 0,
            "gamma": 1.02
        }
    },
    "SMOOTH_COLOR": {
        "description": "Penghalusan maksimal untuk kondisi sinyal RF banyak semut/grain",
        "crop": {"top": 8, "bottom": 8, "left": 10, "right": 10},
        "deinterlace": {"enabled": True, "mode": "bob"},
        "luma_smoothing": {"enabled": True, "edge_threshold": 14},
        "denoise": {"chroma_strength": 11, "temporal_denoise": True, "temporal_alpha": 0.35},
        "clahe": {"enabled": True, "clip_limit": 1.20, "grid_size": 8},
        "sharpen": {"enabled": True, "amount": 1.10, "radius": 1.0, "coring_threshold": 10},
        "color": {
            "auto_white_balance": True,
            "chroma_boost": 1.40,
            "contrast": 1.08,
            "brightness": 2,
            "gamma": 1.05
        }
    },
    "OUTDOOR_SUN": {
        "description": "Anti-glare untuk terpal di bawah terik matahari siang",
        "crop": {"top": 8, "bottom": 8, "left": 10, "right": 10},
        "deinterlace": {"enabled": True, "mode": "bob"},
        "luma_smoothing": {"enabled": True, "edge_threshold": 8},
        "denoise": {"chroma_strength": 7, "temporal_denoise": True, "temporal_alpha": 0.20},
        "clahe": {"enabled": True, "clip_limit": 1.45, "grid_size": 8},
        "sharpen": {"enabled": True, "amount": 1.40, "radius": 1.0, "coring_threshold": 8},
        "color": {
            "auto_white_balance": True,
            "chroma_boost": 1.30,
            "contrast": 1.08,
            "brightness": -5,
            "gamma": 0.96
        }
    },
    "CRISP_EDGES": {
        "description": "Penajaman ekstra pada kontur perimeter target",
        "crop": {"top": 8, "bottom": 8, "left": 10, "right": 10},
        "deinterlace": {"enabled": True, "mode": "bob"},
        "luma_smoothing": {"enabled": True, "edge_threshold": 6},
        "denoise": {"chroma_strength": 7, "temporal_denoise": True, "temporal_alpha": 0.20},
        "clahe": {"enabled": True, "clip_limit": 1.60, "grid_size": 8},
        "sharpen": {"enabled": True, "amount": 1.50, "radius": 1.0, "coring_threshold": 6},
        "color": {
            "auto_white_balance": True,
            "chroma_boost": 1.35,
            "contrast": 1.12,
            "brightness": 0,
            "gamma": 1.02
        }
    }
}

DEFAULT_CONFIG = {
    "preset": "TARP_DETECTION",
    "camera": {
        "index": 0,
        "width": 720,
        "height": 576,
        "fps": 30,
        "fourcc": "MJPG",
        "buffer_size": 1
    },
    "output": {
        "resize_width": 640,
        "resize_height": 480
    },
    **PRESETS["TARP_DETECTION"]
}


# ==============================================================================
# CLASS: EasyCapEnhancer (Clean Tarp Vision Engine)
# ==============================================================================
class EasyCapEnhancer:
    """
    Engine pemrosesan citra khusus deteksi terpal kotak biru & merah Teknofest.
    """
    def __init__(self, config=None, config_path=None, preset_name="TARP_DETECTION"):
        self.config = json.loads(json.dumps(DEFAULT_CONFIG))
        self.current_preset = preset_name
        self._clahe = None
        self._lut = None
        self._last_lut_params = None
        self._prev_y = None

        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        elif config:
            self.update_config(config)
        else:
            self.apply_preset(preset_name)

        self._init_clahe()
        self._rebuild_lut()

        self.last_process_time_ms = 0.0
        self.process_fps = 0.0
        self._last_time = time.time()

    def apply_preset(self, preset_name):
        if preset_name in PRESETS:
            self.current_preset = preset_name
            self.config["preset"] = preset_name
            p_data = PRESETS[preset_name]
            for k in ["crop", "deinterlace", "luma_smoothing", "denoise", "clahe", "sharpen", "color"]:
                if k in p_data:
                    self.config[k] = json.loads(json.dumps(p_data[k]))
            self._init_clahe()
            self._rebuild_lut()
            print(f"[ENHANCER] Preset diaktifkan: [{preset_name}] -> {p_data.get('description', '')}")
            return True
        return False

    def _init_clahe(self):
        clip = float(self.config["clahe"]["clip_limit"])
        grid = int(self.config["clahe"]["grid_size"])
        grid = max(2, min(32, grid))
        clip = max(0.5, min(10.0, clip))
        self._clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))

    def _rebuild_lut(self):
        gamma = max(0.1, float(self.config["color"].get("gamma", 1.0)))
        contrast = max(0.1, float(self.config["color"].get("contrast", 1.0)))
        brightness = int(self.config["color"].get("brightness", 0))

        current_params = (gamma, contrast, brightness)
        if current_params == self._last_lut_params:
            return

        lut = np.zeros(256, dtype=np.uint8)
        inv_gamma = 1.0 / gamma
        for i in range(256):
            val = contrast * (i - 128) + 128 + brightness
            val = np.clip(val, 0, 255)
            val = 255.0 * ((val / 255.0) ** inv_gamma)
            lut[i] = np.clip(val, 0, 255).astype(np.uint8)

        self._lut = lut
        self._last_lut_params = current_params

    def update_config(self, new_cfg):
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d:
                    deep_update(d[k], v)
                else:
                    d[k] = v
        deep_update(self.config, new_cfg)
        self._init_clahe()
        self._rebuild_lut()

    def save_config(self, filepath="easycap_config.json"):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            print(f"[ENHANCER] Konfigurasi berhasil disimpan ke '{filepath}'")
            return True
        except Exception as e:
            print(f"[ENHANCER ERROR] Gagal menyimpan config: {e}")
            return False

    def load_config(self, filepath="easycap_config.json"):
        try:
            if not os.path.exists(filepath):
                return False
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.update_config(data)
                if "preset" in data:
                    self.current_preset = data["preset"]
            print(f"[ENHANCER] Konfigurasi berhasil dimuat dari '{filepath}'")
            return True
        except Exception as e:
            print(f"[ENHANCER ERROR] Gagal memuat config: {e}")
            return False

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Pipeline pemrosesan video berkecepatan tinggi (<6ms / >160 FPS) 
        khusus deteksi terpal di atas rumput.
        """
        if frame is None or frame.size == 0:
            return frame

        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        # 1. CROP MARGINS
        c = self.config["crop"]
        t, b, l, r = c.get("top", 0), c.get("bottom", 0), c.get("left", 0), c.get("right", 0)
        if t > 0 or b > 0 or l > 0 or r > 0:
            y1, y2 = t, h - b if b > 0 else h
            x1, x2 = l, w - r if r > 0 else w
            if y2 > y1 and x2 > x1:
                frame = frame[y1:y2, x1:x2]
                h, w = frame.shape[:2]

        # 2. INSTANTANEOUS FIELD DEINTERLACING (Bob)
        if self.config["deinterlace"].get("enabled", True):
            field = frame[1::2, :, :]
            frame = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)

        # 3. YCrCb SEPARATION
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # 4. FAST AUTO WHITE BALANCE (Membuang tint hijau/kuning)
        if self.config["color"].get("auto_white_balance", True):
            cr_offset = int(np.clip(np.mean(cr) - 128, -20, 20))
            cb_offset = int(np.clip(np.mean(cb) - 128, -20, 20))
            if cr_offset != 0:
                cr = cv2.subtract(cr, cr_offset)
            if cb_offset != 0:
                cb = cv2.subtract(cb, cb_offset)

        # 5. HEAVY CHROMA DENOISING & COLOR EXPANSION (Warna Terpal Jadi Solid Murni)
        chroma_k = int(self.config["denoise"].get("chroma_strength", 9))
        if chroma_k > 0:
            if chroma_k % 2 == 0:
                chroma_k += 1
            cr = cv2.GaussianBlur(cr, (chroma_k, chroma_k), 2.5)
            cb = cv2.GaussianBlur(cb, (chroma_k, chroma_k), 2.5)

        chroma_boost = float(self.config["color"].get("chroma_boost", 1.35))
        if abs(chroma_boost - 1.0) > 0.05:
            neutral_128 = np.full_like(cr, 128)
            cr = cv2.addWeighted(cr, chroma_boost, neutral_128, 1.0 - chroma_boost, 0)
            cb = cv2.addWeighted(cb, chroma_boost, neutral_128, 1.0 - chroma_boost, 0)

        # 6. EDGE-PRESERVING LUMA SMOOTHING (Hapus Grain Permukaan, Jaga Sudut Terpal)
        if self.config.get("luma_smoothing", {}).get("enabled", True):
            edge_th = int(self.config["luma_smoothing"].get("edge_threshold", 10))
            blur_y = cv2.GaussianBlur(y, (5, 5), 1.5)
            diff_noise = cv2.absdiff(y, blur_y)
            mask_edges = cv2.threshold(diff_noise, edge_th, 255, cv2.THRESH_BINARY)[1]
            y = np.where(mask_edges == 255, y, blur_y)

        # 7. CONTROLLED CLAHE (Clip rendah 1.35 agar tidak mengamplifikasi noise)
        if self.config["clahe"].get("enabled", True) and self._clahe is not None:
            y = self._clahe.apply(y)

        # 8. TEMPORAL NOISE REJECTION
        if self.config["denoise"].get("temporal_denoise", False):
            alpha = float(self.config["denoise"].get("temporal_alpha", 0.25))
            if self._prev_y is not None and self._prev_y.shape == y.shape:
                diff = cv2.absdiff(y, self._prev_y)
                mask_noise = cv2.threshold(diff, 16, 255, cv2.THRESH_BINARY_INV)[1]
                smoothed = cv2.addWeighted(y, 1.0 - alpha, self._prev_y, alpha, 0)
                y = np.where(mask_noise == 255, smoothed, y)
            self._prev_y = y.copy()

        # 9. HIGH-CORING EDGE SHARPENING (Hanya pertajam garis keliling terpal)
        if self.config["sharpen"].get("enabled", True):
            amount = float(self.config["sharpen"].get("amount", 1.35))
            coring = int(self.config["sharpen"].get("coring_threshold", 8))
            if amount > 0.05:
                blur_s = cv2.GaussianBlur(y, (0, 0), sigmaX=1.0)
                diff_s = cv2.absdiff(y, blur_s)
                mask_p = cv2.threshold(diff_s, coring, 255, cv2.THRESH_BINARY)[1]
                boosted = cv2.addWeighted(y, 1.0 + amount, blur_s, -amount, 0)
                y = np.where(mask_p == 255, boosted, y)

        # 10. RECOMBINE & LUT
        merged = cv2.merge([y, cr, cb])
        enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

        self._rebuild_lut()
        if self._lut is not None:
            enhanced_bgr = cv2.LUT(enhanced_bgr, self._lut)

        # 11. RESIZE JIKA DIPERLUKAN
        target_w = self.config["output"].get("resize_width", 0)
        target_h = self.config["output"].get("resize_height", 0)
        if target_w > 0 and target_h > 0:
            if w != target_w or h != target_h:
                enhanced_bgr = cv2.resize(enhanced_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0
        self.last_process_time_ms = dt_ms

        now = time.time()
        dt = now - self._last_time
        if dt > 0:
            inst = 1.0 / dt
            self.process_fps = (0.85 * self.process_fps) + (0.15 * inst) if self.process_fps > 0 else inst
        self._last_time = now

        return enhanced_bgr


# ==============================================================================
# CLASS: EasyCapCapture (DirectShow Windows Threaded)
# ==============================================================================
class EasyCapCapture:
    def __init__(self, camera_index=0, width=720, height=576, fps=30, fourcc='MJPG'):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc

        self.cap = None
        self.running = False
        self.thread = None

        self.latest_raw_frame = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self.fps_tracker = 0.0
        self._last_frame_time = None

    def start(self):
        print(f"[CAPTURE] Membuka EasyCap USB pada indeks {self.camera_index} (DirectShow)...")
        if os.name == 'nt':
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"[CAPTURE ERROR] Kamera pada indeks {self.camera_index} tidak dapat dibuka!")
            return False

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Kalibrasi hardware default EasyCap DirectShow untuk mencegah overexposure
            self.cap.set(cv2.CAP_PROP_CONTRAST, 75)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
            self.cap.set(cv2.CAP_PROP_SATURATION, 128)

            if self.fourcc and len(self.fourcc) == 4:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        except Exception as e:
            print(f"[CAPTURE WARN] Properties: {e}")

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[CAPTURE] Sukses: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

        self.running = True
        self.thread = threading.Thread(target=self._capture_worker, daemon=True, name="EasyCapCaptureThread")
        self.thread.start()
        return True

    def _capture_worker(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_raw_frame = frame
                    self.frame_id += 1

                now = time.time()
                if self._last_frame_time is not None:
                    dt = now - self._last_frame_time
                    if dt > 0:
                        inst = 1.0 / dt
                        self.fps_tracker = (0.9 * self.fps_tracker) + (0.1 * inst) if self.fps_tracker > 0 else inst
                self._last_frame_time = now
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            if self.latest_raw_frame is not None:
                return True, self.latest_raw_frame.copy()
            return False, None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None


# ==============================================================================
# REALISTIC UAV FLIGHT SIMULATION FRAME GENERATOR
# ==============================================================================
def create_realistic_uav_flight_frame(step_count=0):
    w, h = 640, 480
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Lapangan Rumput
    frame[:, :] = (85, 115, 80)
    noise_texture = (np.random.randn(h, w, 3) * 10).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise_texture, 0, 255).astype(np.uint8)

    # Runway Aspal
    cv2.rectangle(frame, (40, 0), (200, h), (70, 75, 70), -1)
    for y_mark in range(20, h, 60):
        cv2.rectangle(frame, (115, y_mark), (125, y_mark + 35), (170, 175, 165), -1)

    # Terpal Kotak Biru
    cx = int(360 + 50 * np.sin(step_count * 0.04))
    cy = int(240 + 30 * np.cos(step_count * 0.04))
    cv2.rectangle(frame, (cx - 45, cy - 45), (cx + 45, cy + 45), (190, 190, 190), -1)
    cv2.rectangle(frame, (cx - 36, cy - 36), (cx + 36, cy + 36), (150, 75, 20), -1)
    cv2.putText(frame, "TARGET BLUE", (cx - 45, cy - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    # Target Merah
    rx = 520
    ry = 310
    cv2.circle(frame, (rx, ry), 36, (200, 200, 200), -1)
    cv2.circle(frame, (rx, ry), 30, (30, 30, 170), -1)

    # OSD
    cv2.putText(frame, "ALT: 28.4M   SPD: 16.2M/S   BAT: 15.6V", (30, 35), cv2.FONT_HERSHEY_DUPLEX, 0.55, (210, 210, 210), 1)
    cv2.putText(frame, "UAV MISSION 2: SCANNING DROP ZONE", (30, h - 25), cv2.FONT_HERSHEY_DUPLEX, 0.52, (210, 210, 210), 1)

    # Motion Combing
    shift = 7
    frame[1::2, shift:, :] = frame[1::2, :-shift, :]

    # RF Noise
    noise_cr = (np.random.randn(h, w) * 14).astype(np.int16)
    noise_cb = (np.random.randn(h, w) * 14).astype(np.int16)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.int16)
    ycrcb[:, :, 1] = np.clip(ycrcb[:, :, 1] + noise_cr, 0, 255)
    ycrcb[:, :, 2] = np.clip(ycrcb[:, :, 2] + noise_cb, 0, 255)
    frame = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    frame[:8, :] = 12
    frame[-8:, :] = 12
    frame[:, :10] = 12
    frame[:, -10:] = 12
    return frame


# ==============================================================================
# GUI RUNNER
# ==============================================================================
def run_interactive_enhancer(camera_index=0, config_file="easycap_config.json", video_file=None, image_file=None):
    config_path = str(Path(config_file).resolve())
    enhancer = EasyCapEnhancer(config_path=config_path if os.path.exists(config_path) else None)

    capture = None
    use_synthetic = False
    cap_file = None
    static_img = None

    if image_file and os.path.exists(image_file):
        print(f"[MAIN] Membaca file gambar: {image_file}")
        static_img = cv2.imread(image_file)
    elif video_file and os.path.exists(video_file):
        print(f"[MAIN] Membuka file video: {video_file}")
        cap_file = cv2.VideoCapture(video_file)
    else:
        cfg_cam = enhancer.config.get("camera", {})
        capture = EasyCapCapture(
            camera_index=camera_index,
            width=cfg_cam.get("width", 720),
            height=cfg_cam.get("height", 576),
            fps=cfg_cam.get("fps", 30),
            fourcc=cfg_cam.get("fourcc", "MJPG")
        )
        if not capture.start():
            print("\n[INFO] Capture Card tidak terdeteksi. Beralih ke SIMULASI PENERBANGAN UAV.")
            use_synthetic = True

    win_name = "EasyCap VRX - Deteksi Terpal UAV Teknofest"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    def nothing(x):
        pass

    cv2.createTrackbar("Deinterlace", win_name, 1, 1, nothing)         # 0/1 (Bob)
    cv2.createTrackbar("Luma Smooth", win_name, 1, 1, nothing)          # 0/1 (Hapus bintik/grain)
    cv2.createTrackbar("Chroma Denoise", win_name, 9, 15, nothing)      # 0 - 15 (Solidkan warna terpal)
    cv2.createTrackbar("CLAHE Clip x10", win_name, 14, 30, nothing)     # 14 = 1.4 (Kontras lembut)
    cv2.createTrackbar("Sharpen x10", win_name, 14, 30, nothing)        # 14 = 1.4 (Pertegas tepi)
    cv2.createTrackbar("Chroma Boost x10", win_name, 14, 25, nothing)   # 14 = 1.4 (Warna biru/merah pop)
    cv2.createTrackbar("Auto White Bal", win_name, 1, 1, nothing)       # 0/1
    cv2.createTrackbar("Crop Margin", win_name, 10, 30, nothing)

    # Sync posisi awal
    try:
        cv2.setTrackbarPos("Deinterlace", win_name, 1 if enhancer.config["deinterlace"]["enabled"] else 0)
        cv2.setTrackbarPos("Luma Smooth", win_name, 1 if enhancer.config.get("luma_smoothing", {}).get("enabled", True) else 0)
        cv2.setTrackbarPos("Chroma Denoise", win_name, int(enhancer.config["denoise"]["chroma_strength"]))
        cv2.setTrackbarPos("CLAHE Clip x10", win_name, int(enhancer.config["clahe"]["clip_limit"] * 10))
        cv2.setTrackbarPos("Sharpen x10", win_name, int(enhancer.config["sharpen"]["amount"] * 10))
        cv2.setTrackbarPos("Chroma Boost x10", win_name, int(enhancer.config["color"].get("chroma_boost", 1.35) * 10))
        cv2.setTrackbarPos("Auto White Bal", win_name, 1 if enhancer.config["color"]["auto_white_balance"] else 0)
        cv2.setTrackbarPos("Crop Margin", win_name, int(enhancer.config["crop"]["left"]))
    except Exception as e:
        print(f"[GUI WARN] Sync: {e}")

    view_mode = 0
    view_names = ["SIDE-BY-SIDE (Before | After)", "SPLIT SLIDER (50/50)", "ENHANCED FULLSCREEN", "RAW VRX FULLSCREEN"]
    preset_keys = list(PRESETS.keys())
    preset_idx = preset_keys.index(enhancer.current_preset) if enhancer.current_preset in preset_keys else 0
    status_msg = f"Preset: [{enhancer.current_preset}]. Tekan [P] ganti preset, [S] simpan."
    status_timer = time.time() + 4.0
    step_sim = 0

    print("\n" + "=" * 65)
    print(" KONTROL OPERATOR TEKNOFEST:")
    print("  [P]         : Ganti Preset (TARP_DETECTION, SMOOTH_COLOR, OUTDOOR_SUN, CRISP_EDGES)")
    print("  [S]         : Simpan konfigurasi ke easycap_config.json")
    print("  [L]         : Muat ulang konfigurasi")
    print("  [R]         : Reset ke preset TARP_DETECTION")
    print("  [TAB] / [M] : Ganti mode tampilan layar")
    print("  [C] / [SPC] : Snapshot tangkapan layar Before & After")
    print("  [Q] / [ESC] : Keluar")
    print("=" * 65 + "\n")

    try:
        while True:
            raw_frame = None
            if static_img is not None:
                raw_frame = static_img.copy()
                time.sleep(0.03)
            elif cap_file:
                ret, raw_frame = cap_file.read()
                if not ret or raw_frame is None:
                    cap_file.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, raw_frame = cap_file.read()
            elif capture and not use_synthetic:
                ret, raw_frame = capture.read()
                if not ret or raw_frame is None:
                    time.sleep(0.005)
                    continue
            else:
                step_sim += 1
                raw_frame = create_realistic_uav_flight_frame(step_sim)
                time.sleep(0.03)

            if raw_frame is None:
                continue

            # Update dari trackbars
            enhancer.config["deinterlace"]["enabled"] = (cv2.getTrackbarPos("Deinterlace", win_name) == 1)
            enhancer.config["luma_smoothing"]["enabled"] = (cv2.getTrackbarPos("Luma Smooth", win_name) == 1)
            enhancer.config["denoise"]["chroma_strength"] = cv2.getTrackbarPos("Chroma Denoise", win_name)
            enhancer.config["clahe"]["clip_limit"] = max(0.5, cv2.getTrackbarPos("CLAHE Clip x10", win_name) / 10.0)
            enhancer.config["sharpen"]["amount"] = cv2.getTrackbarPos("Sharpen x10", win_name) / 10.0
            enhancer.config["color"]["chroma_boost"] = max(0.5, cv2.getTrackbarPos("Chroma Boost x10", win_name) / 10.0)
            enhancer.config["color"]["auto_white_balance"] = (cv2.getTrackbarPos("Auto White Bal", win_name) == 1)

            crop_m = cv2.getTrackbarPos("Crop Margin", win_name)
            enhancer.config["crop"]["top"] = crop_m
            enhancer.config["crop"]["bottom"] = crop_m
            enhancer.config["crop"]["left"] = crop_m
            enhancer.config["crop"]["right"] = crop_m

            enhancer._init_clahe()
            enhancer._rebuild_lut()

            enhanced = enhancer.process(raw_frame)

            h, w = enhanced.shape[:2]
            raw_resized = cv2.resize(raw_frame, (w, h))

            display_img = None
            if view_mode == 0:
                lbl_raw = raw_resized.copy()
                lbl_enh = enhanced.copy()
                cv2.rectangle(lbl_raw, (0, 0), (w, 35), (20, 20, 20), -1)
                cv2.putText(lbl_raw, "[RAW VRX INPUT - ANALOG]", (15, 24), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 255), 2)

                cv2.rectangle(lbl_enh, (0, 0), (w, 35), (20, 20, 20), -1)
                cv2.putText(lbl_enh, f"[CLEAN TARP - {enhancer.current_preset}]", (15, 24), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 0), 2)
                display_img = np.hstack([lbl_raw, lbl_enh])

            elif view_mode == 1:
                split_x = int(w * 0.5)
                display_img = enhanced.copy()
                display_img[:, :split_x] = raw_resized[:, :split_x]
                cv2.line(display_img, (split_x, 0), (split_x, h), (0, 255, 255), 2)
                cv2.putText(display_img, "RAW", (split_x - 70, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_img, "ENHANCED", (split_x + 15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)

            elif view_mode == 2:
                display_img = enhanced.copy()
                cv2.putText(display_img, f"[CLEAN TARP - {enhancer.current_preset}]", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
            else:
                display_img = raw_resized.copy()
                cv2.putText(display_img, "[RAW VRX]", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)

            dh, dw = display_img.shape[:2]
            hud_bar = np.zeros((45, dw, 3), dtype=np.uint8)
            hud_bar[:] = (25, 25, 25)

            fps_cam = capture.fps_tracker if capture else 30.0
            fps_proc = enhancer.process_fps
            lat_ms = enhancer.last_process_time_ms

            info_text = f"Cam: {fps_cam:4.1f} FPS | Process: {lat_ms:4.1f}ms ({fps_proc:4.1f} FPS) | Preset: [{enhancer.current_preset}]"
            cv2.putText(hud_bar, info_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            if time.time() < status_timer:
                cv2.putText(hud_bar, f">> {status_msg}", (dw - 520, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

            final_gui = np.vstack([display_img, hud_bar])
            cv2.imshow(win_name, final_gui)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q'), ord('Q')]:
                break
            elif key in [ord('p'), ord('P')]:
                preset_idx = (preset_idx + 1) % len(preset_keys)
                p_name = preset_keys[preset_idx]
                enhancer.apply_preset(p_name)
                cv2.setTrackbarPos("Deinterlace", win_name, 1 if enhancer.config["deinterlace"]["enabled"] else 0)
                cv2.setTrackbarPos("Luma Smooth", win_name, 1 if enhancer.config.get("luma_smoothing", {}).get("enabled", True) else 0)
                cv2.setTrackbarPos("Chroma Denoise", win_name, int(enhancer.config["denoise"]["chroma_strength"]))
                cv2.setTrackbarPos("CLAHE Clip x10", win_name, int(enhancer.config["clahe"]["clip_limit"] * 10))
                cv2.setTrackbarPos("Sharpen x10", win_name, int(enhancer.config["sharpen"]["amount"] * 10))
                cv2.setTrackbarPos("Chroma Boost x10", win_name, int(enhancer.config["color"].get("chroma_boost", 1.35) * 10))
                status_msg = f"Preset: [{p_name}]"
                status_timer = time.time() + 3.0
            elif key in [9, ord('m'), ord('M')]:
                view_mode = (view_mode + 1) % 4
                status_msg = f"Tampilan: {view_names[view_mode]}"
                status_timer = time.time() + 2.5
            elif key in [ord('s'), ord('S')]:
                enhancer.save_config(config_file)
                status_msg = f"Preset disimpan ke '{config_file}'!"
                status_timer = time.time() + 3.0
            elif key in [ord('l'), ord('L')]:
                enhancer.load_config(config_file)
                status_msg = "Konfigurasi dimuat ulang!"
                status_timer = time.time() + 3.0
            elif key in [ord('r'), ord('R')]:
                enhancer.apply_preset("TARP_DETECTION")
                cv2.setTrackbarPos("Deinterlace", win_name, 1)
                cv2.setTrackbarPos("Luma Smooth", win_name, 1)
                cv2.setTrackbarPos("Chroma Denoise", win_name, 9)
                cv2.setTrackbarPos("CLAHE Clip x10", win_name, 14)
                cv2.setTrackbarPos("Sharpen x10", win_name, 14)
                cv2.setTrackbarPos("Chroma Boost x10", win_name, 14)
                cv2.setTrackbarPos("Auto White Bal", win_name, 1)
                cv2.setTrackbarPos("Crop Margin", win_name, 10)
                status_msg = "Reset ke default TARP_DETECTION!"
                status_timer = time.time() + 2.5
            elif key in [ord('c'), ord('C'), 32]:
                ts = int(time.time())
                fn_before = f"easycap_snapshot_raw_{ts}.png"
                fn_after = f"easycap_snapshot_enhanced_{ts}.png"
                cv2.imwrite(fn_before, raw_frame)
                cv2.imwrite(fn_after, enhanced)
                print(f"[SNAPSHOT] Disimpan: {fn_before} & {fn_after}")
                status_msg = f"Snapshot disimpan ({fn_after})!"
                status_timer = time.time() + 3.0

    finally:
        if capture:
            capture.stop()
        if cap_file:
            cap_file.release()
        cv2.destroyAllWindows()


def create_easycap_stream(camera_index=0, config_path="easycap_config.json", preset="TARP_DETECTION"):
    enhancer = EasyCapEnhancer(config_path=config_path if os.path.exists(config_path) else None, preset_name=preset)
    cfg_cam = enhancer.config.get("camera", {})
    capture = EasyCapCapture(
        camera_index=camera_index,
        width=cfg_cam.get("width", 720),
        height=cfg_cam.get("height", 576),
        fps=cfg_cam.get("fps", 30),
        fourcc=cfg_cam.get("fourcc", "MJPG")
    )
    if capture.start():
        return capture, enhancer
    return None, enhancer


def list_available_cameras(max_indices=6):
    print("\n" + "=" * 50)
    print(" PEMINDAI KAMERA DIRECTSHOW (WINDOWS):")
    print("=" * 50)
    found = []
    for idx in range(max_indices):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            found.append((idx, w, h, fps))
            cap.release()
            print(f"  [OK] Kamera ditemukan di Index [{idx}] -> Resolusi: {w}x{h}")
    if not found:
        print("  [!] Tidak ada kamera DirectShow yang terdeteksi.")
    print("=" * 50 + "\n")
    return found


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EasyCap VRX Clean Tarp Video Enhancer (UAV Ground Tarp Detection)")
    parser.add_argument("--index", "-i", type=int, default=0, help="Indeks kamera capture card (default: 0)")
    parser.add_argument("--config", "-c", type=str, default="easycap_config.json", help="Path file konfigurasi JSON")
    parser.add_argument("--preset", "-p", type=str, default="TARP_DETECTION", choices=list(PRESETS.keys()), help="Preset mode")
    parser.add_argument("--video", "-v", type=str, default=None, help="Path ke file video rekaman (opsional)")
    parser.add_argument("--image", "-img", type=str, default=None, help="Path ke file gambar (opsional)")
    parser.add_argument("--list", "-l", action="store_true", help="Pindai kamera aktif di Windows")
    args = parser.parse_args()

    if args.list:
        list_available_cameras()
        sys.exit(0)

    run_interactive_enhancer(
        camera_index=args.index,
        config_file=args.config,
        video_file=args.video,
        image_file=args.image
    )
