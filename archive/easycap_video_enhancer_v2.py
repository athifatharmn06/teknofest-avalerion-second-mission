"""
================================================================================
     EASYCAP & VRX ANALOG FPV VIDEO ENHANCER (V2 - ULTRA EDITION)
                         Teknofest Autonomy & Vision
================================================================================

Deskripsi V2:
Versi evolusi lanjutan (V2) dengan algoritma Multi-Band Signal Decomposition
khusus untuk kamera bawah (nadir) drone pendeteksi terpal kotak biru & merah.

Penyempurnaan Utama di V2:
1. [Anti-Moire & RF Diagonal Carrier Filter]:
   - Menghapus pola gelombang diagonal 45 derajat (herringbone noise) akibat 
     interferensi subcarrier analog 5.8 GHz & jitter ADC EasyCap.
2. [Selective Target Color Isolation (Blue & Red Boost)]:
   - Menambah saturasi secara khusus pada warna Biru (Hue ~105-125) dan Merah (Hue 0-10, 170-180),
     sehingga terpal sangat kontras terhadap rumput/aspal tanpa membuat latar belakang over-saturated.
3. [Dual-Band Edge-Preserving Luma Reconstruction]:
   - Memisahkan layer iluminasi dasar dan layer detail tepi. Permukaan bidang dibuat
     100% mulus dan bebas bintik semut, sedangkan sudut kotak terpal terkunci sangat tajam.
4. [Zoom Loupe Inspector (GUI Feature)]:
   - Tekan tombol [Z] untuk menampilkan kaca pembesar (2x Zoom ROI) di tengah layar
     guna memverifikasi ketajaman garis tepi terpal secara real-time.
5. [Default Index = 1 (Windows DirectShow)]:
   - Dikonfigurasi otomatis ke index 1 (EasyCap VRX), menghindari bentrok dengan Webcam internal (Index 0).

Shortcut Keyboard (GUI V2):
  [P]         : Ganti Preset V2 (ULTRA_TARP_V2 -> MAX_COLOR_POP -> ANTI_MOIRE -> LOW_LIGHT_HDR)
  [Z]         : Toggle Zoom Loupe 2x (Kaca Pembesar ROI Target)
  [S]         : Simpan konfigurasi ke 'easycap_config_v2.json'
  [L]         : Muat ulang konfigurasi dari file JSON
  [R]         : Reset ke preset optimal ULTRA_TARP_V2
  [TAB] / [M] : Ganti mode tampilan (Side-by-Side, Split-Slider, Enhanced, Raw)
  [C] / [SPC] : Simpan gambar snapshot komparasi Before & After
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
# PRESETS KHUSUS V2 (ULTRA TARGET VISIBILITY)
# ==============================================================================
PRESETS_V2 = {
    "ULTRA_TARP_V2": {
        "description": "Keseimbangan sempurna: Bebas garis diagonal, terpal solid, tepi tajam",
        "crop": {"top": 8, "bottom": 8, "left": 10, "right": 10},
        "deinterlace": {"enabled": True, "mode": "bob"},
        "anti_moire": {"enabled": True, "filter_size": 5, "coring_threshold": 6},
        "luma_smoothing": {"enabled": True, "edge_threshold": 10},
        "denoise": {"chroma_strength": 9, "temporal_denoise": True, "temporal_alpha": 0.25},
        "clahe": {"enabled": True, "clip_limit": 1.50, "grid_size": 8},
        "sharpen": {"enabled": True, "amount": 1.45, "radius": 1.0, "coring_threshold": 7},
        "color": {
            "auto_white_balance": True,
            "chroma_boost": 1.40,
            "selective_target_boost": True,
            "contrast": 1.12,
            "brightness": 1,
            "gamma": 1.04
        }
    },
    "MAX_COLOR_POP": {
        "description": "Saturasi ekstra agresif untuk target terpal yang sangat jauh / pudar",
        "crop": {"top": 8, "bottom": 8, "left": 10, "right": 10},
        "deinterlace": {"enabled": True, "mode": "bob"},
        "anti_moire": {"enabled": True, "filter_size": 5, "coring_threshold": 5},
        "luma_smoothing": {"enabled": True, "edge_threshold": 12},
        "denoise": {"chroma_strength": 11, "temporal_denoise": True, "temporal_alpha": 0.30},
        "clahe": {"enabled": True, "clip_limit": 1.65, "grid_size": 8},
        "sharpen": {"enabled": True, "amount": 1.50, "radius": 1.0, "coring_threshold": 6},
        "color": {
            "auto_white_balance": True,
            "chroma_boost": 1.60,
            "selective_target_boost": True,
            "contrast": 1.15,
            "brightness": 2,
            "gamma": 1.06
        }
    },
    "ANTI_MOIRE": {
        "description": "Peredaman maksimal untuk gangguan garis-garis gelombang diagonal RF",
        "crop": {"top": 8, "bottom": 8, "left": 10, "right": 10},
        "deinterlace": {"enabled": True, "mode": "bob"},
        "anti_moire": {"enabled": True, "filter_size": 7, "coring_threshold": 8},
        "luma_smoothing": {"enabled": True, "edge_threshold": 14},
        "denoise": {"chroma_strength": 11, "temporal_denoise": True, "temporal_alpha": 0.35},
        "clahe": {"enabled": True, "clip_limit": 1.30, "grid_size": 8},
        "sharpen": {"enabled": True, "amount": 1.30, "radius": 1.0, "coring_threshold": 9},
        "color": {
            "auto_white_balance": True,
            "chroma_boost": 1.35,
            "selective_target_boost": True,
            "contrast": 1.08,
            "brightness": 0,
            "gamma": 1.02
        }
    },
    "LOW_LIGHT_HDR": {
        "description": "HDR Shadow Recovery untuk kondisi sore hari atau mendung",
        "crop": {"top": 8, "bottom": 8, "left": 10, "right": 10},
        "deinterlace": {"enabled": True, "mode": "bob"},
        "anti_moire": {"enabled": True, "filter_size": 5, "coring_threshold": 6},
        "luma_smoothing": {"enabled": True, "edge_threshold": 10},
        "denoise": {"chroma_strength": 9, "temporal_denoise": True, "temporal_alpha": 0.25},
        "clahe": {"enabled": True, "clip_limit": 2.20, "grid_size": 8},
        "sharpen": {"enabled": True, "amount": 1.55, "radius": 1.1, "coring_threshold": 6},
        "color": {
            "auto_white_balance": True,
            "chroma_boost": 1.45,
            "selective_target_boost": True,
            "contrast": 1.18,
            "brightness": 8,
            "gamma": 1.15
        }
    }
}

DEFAULT_CONFIG_V2 = {
    "preset": "ULTRA_TARP_V2",
    "camera": {
        "index": 1,
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
    **PRESETS_V2["ULTRA_TARP_V2"]
}


# ==============================================================================
# CLASS: EasyCapEnhancerV2 (Ultra High-Performance Engine)
# ==============================================================================
class EasyCapEnhancerV2:
    """
    Engine V2: Pemrosesan citra tingkat lanjut khusus sinyal EasyCap VRX 
    untuk deteksi objek terpal kotak biru & merah di wahana UAV Teknofest.
    """
    def __init__(self, config=None, config_path=None, preset_name="ULTRA_TARP_V2"):
        self.config = json.loads(json.dumps(DEFAULT_CONFIG_V2))
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
        if preset_name in PRESETS_V2:
            self.current_preset = preset_name
            self.config["preset"] = preset_name
            p_data = PRESETS_V2[preset_name]
            for k in ["crop", "deinterlace", "anti_moire", "luma_smoothing", "denoise", "clahe", "sharpen", "color"]:
                if k in p_data:
                    self.config[k] = json.loads(json.dumps(p_data[k]))
            self._init_clahe()
            self._rebuild_lut()
            print(f"[ENHANCER V2] Preset diaktifkan: [{preset_name}] -> {p_data.get('description', '')}")
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

    def save_config(self, filepath="easycap_config_v2.json"):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            print(f"[ENHANCER V2] Konfigurasi disimpan ke '{filepath}'")
            return True
        except Exception as e:
            print(f"[ENHANCER V2 ERROR] Gagal menyimpan: {e}")
            return False

    def load_config(self, filepath="easycap_config_v2.json"):
        try:
            if not os.path.exists(filepath):
                return False
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.update_config(data)
                if "preset" in data:
                    self.current_preset = data["preset"]
            print(f"[ENHANCER V2] Konfigurasi dimuat dari '{filepath}'")
            return True
        except Exception as e:
            print(f"[ENHANCER V2 ERROR] Gagal memuat: {e}")
            return False

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Pipeline V2: Anti-Moire, Selective Color Isolation, Edge-Preserving Luma & Sharpening.
        """
        if frame is None or frame.size == 0:
            return frame

        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        # 1. CROP MARGINS (Border analog kotor)
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

        # 4. FAST AUTO WHITE BALANCE (Hapus Tint Hijau/Kuning)
        if self.config["color"].get("auto_white_balance", True):
            cr_offset = int(np.clip(np.mean(cr) - 128, -25, 25))
            cb_offset = int(np.clip(np.mean(cb) - 128, -25, 25))
            if cr_offset != 0:
                cr = cv2.subtract(cr, cr_offset)
            if cb_offset != 0:
                cb = cv2.subtract(cb, cb_offset)

        # 5. ANTI-MOIRE & RF DIAGONAL CARRIER LINE FILTER (V2 Inovasi)
        if self.config.get("anti_moire", {}).get("enabled", True):
            ksize = int(self.config["anti_moire"].get("filter_size", 5))
            if ksize % 2 == 0: ksize += 1
            coring_m = int(self.config["anti_moire"].get("coring_threshold", 6))
            
            y_base = cv2.GaussianBlur(y, (ksize, ksize), 1.6)
            diff_moire = cv2.absdiff(y, y_base)
            # Mask batas objek kuat vs gelombang ripple halus
            mask_strong = cv2.threshold(diff_moire, coring_m, 255, cv2.THRESH_BINARY)[1]
            y_detail = cv2.subtract(y, y_base)
            y_clean_detail = np.where(mask_strong == 255, y_detail, 0)
            y = cv2.add(y_base, y_clean_detail)

        # 6. LUMA SURFACE SMOOTHING (Bersihkan Grain pada Permukaan)
        if self.config.get("luma_smoothing", {}).get("enabled", True):
            edge_th = int(self.config["luma_smoothing"].get("edge_threshold", 10))
            blur_y = cv2.GaussianBlur(y, (5, 5), 1.5)
            diff_noise = cv2.absdiff(y, blur_y)
            mask_edges = cv2.threshold(diff_noise, edge_th, 255, cv2.THRESH_BINARY)[1]
            y = np.where(mask_edges == 255, y, blur_y)

        # 7. DUAL-BAND ADAPTIVE CLAHE (Dynamic Range & Shadow Recovery)
        if self.config["clahe"].get("enabled", True) and self._clahe is not None:
            y = self._clahe.apply(y)

        # 8. TEMPORAL NOISE ACCUMULATOR (Live Stream RF Snow Filter)
        if self.config["denoise"].get("temporal_denoise", False):
            alpha = float(self.config["denoise"].get("temporal_alpha", 0.25))
            if self._prev_y is not None and self._prev_y.shape == y.shape:
                diff_temp = cv2.absdiff(y, self._prev_y)
                mask_static = cv2.threshold(diff_temp, 16, 255, cv2.THRESH_BINARY_INV)[1]
                smoothed = cv2.addWeighted(y, 1.0 - alpha, self._prev_y, alpha, 0)
                y = np.where(mask_static == 255, smoothed, y)
            self._prev_y = y.copy()

        # 9. HIGH-CORING EDGE SHARPENING (Pertajam Garis Keliling Terpal)
        if self.config["sharpen"].get("enabled", True):
            amount = float(self.config["sharpen"].get("amount", 1.45))
            coring = int(self.config["sharpen"].get("coring_threshold", 7))
            if amount > 0.05:
                blur_s = cv2.GaussianBlur(y, (0, 0), sigmaX=0.9)
                diff_s = cv2.absdiff(y, blur_s)
                mask_p = cv2.threshold(diff_s, coring, 255, cv2.THRESH_BINARY)[1]
                boosted = cv2.addWeighted(y, 1.0 + amount, blur_s, -amount, 0)
                y = np.where(mask_p == 255, boosted, y)

        # 10. CHROMA PURGE & SELECTIVE TARGET COLOR BOOST (V2 Inovasi)
        chroma_k = int(self.config["denoise"].get("chroma_strength", 9))
        if chroma_k > 0:
            if chroma_k % 2 == 0: chroma_k += 1
            cr = cv2.GaussianBlur(cr, (chroma_k, chroma_k), 2.2)
            cb = cv2.GaussianBlur(cb, (chroma_k, chroma_k), 2.2)

        chroma_boost = float(self.config["color"].get("chroma_boost", 1.40))
        neutral_128 = np.full_like(cr, 128)
        cr = cv2.addWeighted(cr, chroma_boost, neutral_128, 1.0 - chroma_boost, 0)
        cb = cv2.addWeighted(cb, chroma_boost, neutral_128, 1.0 - chroma_boost, 0)

        # 11. RECOMBINE & COLOR SPACE CONVERSION
        merged = cv2.merge([y, cr, cb])
        enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

        # 12. SELECTIVE TARGET PURITY BOOST (Blue & Red Pop)
        if self.config["color"].get("selective_target_boost", True):
            hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
            h_chan, s_chan, v_chan = cv2.split(hsv)
            
            # Mask warna target Biru (Hue 95-135) dan Merah (Hue 0-15 & 165-180)
            mask_blue = cv2.inRange(h_chan, 95, 135)
            mask_red1 = cv2.inRange(h_chan, 0, 15)
            mask_red2 = cv2.inRange(h_chan, 165, 180)
            mask_targets = cv2.bitwise_or(mask_blue, cv2.bitwise_or(mask_red1, mask_red2))
            
            # Tambah saturasi 1.25x khusus pada target terpal
            s_boosted = np.clip(cv2.multiply(s_chan.astype(np.float32), 1.25), 0, 255).astype(np.uint8)
            s_final = np.where(mask_targets == 255, s_boosted, s_chan)
            
            enhanced_bgr = cv2.cvtColor(cv2.merge([h_chan, s_final, v_chan]), cv2.COLOR_HSV2BGR)

        # 13. LUT CONTRAST & GAMMA
        self._rebuild_lut()
        if self._lut is not None:
            enhanced_bgr = cv2.LUT(enhanced_bgr, self._lut)

        # 14. RESIZE
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
    def __init__(self, camera_index=1, width=720, height=576, fps=30, fourcc='MJPG'):
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
        print(f"[CAPTURE V2] Membuka EasyCap pada indeks {self.camera_index} (DirectShow Windows)...")
        if os.name == 'nt':
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"[CAPTURE V2 ERROR] Kamera pada indeks {self.camera_index} tidak dapat dibuka!")
            return False

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Kalibrasi hardware default EasyCap DirectShow untuk mencegah overexposure (layar putih)
            self.cap.set(cv2.CAP_PROP_CONTRAST, 75)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
            self.cap.set(cv2.CAP_PROP_SATURATION, 128)

            if self.fourcc and len(self.fourcc) == 4:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        except Exception as e:
            print(f"[CAPTURE V2 WARN] Properties: {e}")

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[CAPTURE V2] Sukses: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

        self.running = True
        self.thread = threading.Thread(target=self._capture_worker, daemon=True, name="EasyCapCaptureV2Thread")
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
# REALISTIC UAV FLIGHT SIMULATION FRAME GENERATOR (With RF Moire Wave)
# ==============================================================================
def create_realistic_uav_flight_frame_v2(step_count=0):
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

    # RF Diagonal Ripple (Herringbone pattern 5.8 GHz)
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    diagonal_ripple = (np.sin((xx + yy) * 0.4) * 6).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + diagonal_ripple[:, :, None], 0, 255).astype(np.uint8)

    # Chroma Noise
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
# GUI RUNNER V2
# ==============================================================================
def run_interactive_enhancer_v2(camera_index=1, config_file="easycap_config_v2.json", video_file=None, image_file=None):
    config_path = str(Path(config_file).resolve())
    enhancer = EasyCapEnhancerV2(config_path=config_path if os.path.exists(config_path) else None)

    capture = None
    use_synthetic = False
    cap_file = None
    static_img = None

    if image_file and os.path.exists(image_file):
        print(f"[MAIN V2] Membaca file gambar: {image_file}")
        static_img = cv2.imread(image_file)
    elif video_file and os.path.exists(video_file):
        print(f"[MAIN V2] Membuka file video: {video_file}")
        cap_file = cv2.VideoCapture(video_file)
    else:
        cfg_cam = enhancer.config.get("camera", {})
        cam_idx = camera_index if camera_index is not None else cfg_cam.get("index", 1)
        capture = EasyCapCapture(
            camera_index=cam_idx,
            width=cfg_cam.get("width", 720),
            height=cfg_cam.get("height", 576),
            fps=cfg_cam.get("fps", 30),
            fourcc=cfg_cam.get("fourcc", "MJPG")
        )
        if not capture.start():
            print(f"\n[INFO] Capture Card (Index {cam_idx}) tidak terdeteksi. Beralih ke SIMULASI PENERBANGAN UAV V2.")
            use_synthetic = True

    win_name = "EasyCap VRX - Ultra Enhancer V2 (Teknofest UAV)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    def nothing(x):
        pass

    cv2.createTrackbar("Deinterlace", win_name, 1, 1, nothing)         # 0/1 (Bob)
    cv2.createTrackbar("Anti-Moire", win_name, 1, 1, nothing)          # 0/1 (Hapus garis diagonal RF)
    cv2.createTrackbar("Luma Smooth", win_name, 1, 1, nothing)          # 0/1 (Hapus bintik/grain)
    cv2.createTrackbar("Chroma Denoise", win_name, 9, 15, nothing)      # 0 - 15 (Solidkan warna)
    cv2.createTrackbar("CLAHE Clip x10", win_name, 15, 30, nothing)     # 15 = 1.5
    cv2.createTrackbar("Sharpen x10", win_name, 15, 30, nothing)        # 15 = 1.5 (Tepi tajam)
    cv2.createTrackbar("Chroma Boost x10", win_name, 14, 25, nothing)   # 14 = 1.4
    cv2.createTrackbar("Target Pop", win_name, 1, 1, nothing)           # 0/1 (Selective Blue & Red boost)
    cv2.createTrackbar("Auto White Bal", win_name, 1, 1, nothing)       # 0/1
    cv2.createTrackbar("Crop Margin", win_name, 10, 30, nothing)

    try:
        cv2.setTrackbarPos("Deinterlace", win_name, 1 if enhancer.config["deinterlace"]["enabled"] else 0)
        cv2.setTrackbarPos("Anti-Moire", win_name, 1 if enhancer.config.get("anti_moire", {}).get("enabled", True) else 0)
        cv2.setTrackbarPos("Luma Smooth", win_name, 1 if enhancer.config.get("luma_smoothing", {}).get("enabled", True) else 0)
        cv2.setTrackbarPos("Chroma Denoise", win_name, int(enhancer.config["denoise"]["chroma_strength"]))
        cv2.setTrackbarPos("CLAHE Clip x10", win_name, int(enhancer.config["clahe"]["clip_limit"] * 10))
        cv2.setTrackbarPos("Sharpen x10", win_name, int(enhancer.config["sharpen"]["amount"] * 10))
        cv2.setTrackbarPos("Chroma Boost x10", win_name, int(enhancer.config["color"].get("chroma_boost", 1.40) * 10))
        cv2.setTrackbarPos("Target Pop", win_name, 1 if enhancer.config["color"].get("selective_target_boost", True) else 0)
        cv2.setTrackbarPos("Auto White Bal", win_name, 1 if enhancer.config["color"]["auto_white_balance"] else 0)
        cv2.setTrackbarPos("Crop Margin", win_name, int(enhancer.config["crop"]["left"]))
    except Exception as e:
        print(f"[GUI V2 WARN] Sync: {e}")

    view_mode = 0
    show_zoom_loupe = False
    view_names = ["SIDE-BY-SIDE (Before | After)", "SPLIT SLIDER (50/50)", "ENHANCED FULLSCREEN", "RAW VRX FULLSCREEN"]
    preset_keys = list(PRESETS_V2.keys())
    preset_idx = preset_keys.index(enhancer.current_preset) if enhancer.current_preset in preset_keys else 0
    status_msg = f"Preset: [{enhancer.current_preset}]. Tekan [P] ganti preset, [Z] Zoom Loupe 2x."
    status_timer = time.time() + 4.0
    step_sim = 0

    print("\n" + "=" * 65)
    print(" KONTROL OPERATOR TEKNOFEST V2 (ULTRA):")
    print("  [P]         : Ganti Preset (ULTRA_TARP_V2, MAX_COLOR_POP, ANTI_MOIRE, LOW_LIGHT_HDR)")
    print("  [Z]         : Toggle Zoom Loupe 2x (Kaca Pembesar ROI Target)")
    print("  [S]         : Simpan konfigurasi ke easycap_config_v2.json")
    print("  [L]         : Muat ulang konfigurasi")
    print("  [R]         : Reset ke preset ULTRA_TARP_V2")
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
                raw_frame = create_realistic_uav_flight_frame_v2(step_sim)
                time.sleep(0.03)

            if raw_frame is None:
                continue

            # Update dari trackbars
            enhancer.config["deinterlace"]["enabled"] = (cv2.getTrackbarPos("Deinterlace", win_name) == 1)
            enhancer.config["anti_moire"]["enabled"] = (cv2.getTrackbarPos("Anti-Moire", win_name) == 1)
            enhancer.config["luma_smoothing"]["enabled"] = (cv2.getTrackbarPos("Luma Smooth", win_name) == 1)
            enhancer.config["denoise"]["chroma_strength"] = cv2.getTrackbarPos("Chroma Denoise", win_name)
            enhancer.config["clahe"]["clip_limit"] = max(0.5, cv2.getTrackbarPos("CLAHE Clip x10", win_name) / 10.0)
            enhancer.config["sharpen"]["amount"] = cv2.getTrackbarPos("Sharpen x10", win_name) / 10.0
            enhancer.config["color"]["chroma_boost"] = max(0.5, cv2.getTrackbarPos("Chroma Boost x10", win_name) / 10.0)
            enhancer.config["color"]["selective_target_boost"] = (cv2.getTrackbarPos("Target Pop", win_name) == 1)
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

            # Zoom Loupe Overlay jika aktif
            if show_zoom_loupe:
                # Ambil ROI tengah 160x120 dan perbesar 2x
                rw, rh = 160, 120
                rx1, ry1 = (w - rw) // 2, (h - rh) // 2
                rx2, ry2 = rx1 + rw, ry1 + rh
                
                roi_raw = cv2.resize(raw_resized[ry1:ry2, rx1:rx2], (rw * 2, rh * 2), interpolation=cv2.INTER_NEAREST)
                roi_enh = cv2.resize(enhanced[ry1:ry2, rx1:rx2], (rw * 2, rh * 2), interpolation=cv2.INTER_NEAREST)
                
                # Tempelkan loupe di pojok kanan bawah frame
                lw, lh = rw * 2, rh * 2
                raw_resized[h - lh - 10:h - 10, w - lw - 10:w - 10] = roi_raw
                cv2.rectangle(raw_resized, (w - lw - 10, h - lh - 10), (w - 10, h - 10), (0, 0, 255), 2)
                cv2.putText(raw_resized, "ZOOM 2X (RAW)", (w - lw - 5, h - lh + 18), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 0, 255), 1)

                enhanced[h - lh - 10:h - 10, w - lw - 10:w - 10] = roi_enh
                cv2.rectangle(enhanced, (w - lw - 10, h - lh - 10), (w - 10, h - 10), (0, 255, 0), 2)
                cv2.putText(enhanced, "ZOOM 2X (ENHANCED)", (w - lw - 5, h - lh + 18), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 255, 0), 1)

            display_img = None
            if view_mode == 0:
                lbl_raw = raw_resized.copy()
                lbl_enh = enhanced.copy()
                cv2.rectangle(lbl_raw, (0, 0), (w, 35), (20, 20, 20), -1)
                cv2.putText(lbl_raw, "[RAW VRX INPUT - ANALOG]", (15, 24), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 255), 2)

                cv2.rectangle(lbl_enh, (0, 0), (w, 35), (20, 20, 20), -1)
                cv2.putText(lbl_enh, f"[V2 ULTRA - {enhancer.current_preset}]", (15, 24), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 0), 2)
                display_img = np.hstack([lbl_raw, lbl_enh])

            elif view_mode == 1:
                split_x = int(w * 0.5)
                display_img = enhanced.copy()
                display_img[:, :split_x] = raw_resized[:, :split_x]
                cv2.line(display_img, (split_x, 0), (split_x, h), (0, 255, 255), 2)
                cv2.putText(display_img, "RAW", (split_x - 70, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_img, "V2 ULTRA", (split_x + 15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)

            elif view_mode == 2:
                display_img = enhanced.copy()
                cv2.putText(display_img, f"[V2 ULTRA - {enhancer.current_preset}]", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
            else:
                display_img = raw_resized.copy()
                cv2.putText(display_img, "[RAW VRX]", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)

            dh, dw = display_img.shape[:2]
            hud_bar = np.zeros((45, dw, 3), dtype=np.uint8)
            hud_bar[:] = (25, 25, 25)

            fps_cam = capture.fps_tracker if capture else 30.0
            fps_proc = enhancer.process_fps
            lat_ms = enhancer.last_process_time_ms

            info_text = f"Cam (Idx {camera_index}): {fps_cam:4.1f} FPS | V2 Proc: {lat_ms:4.1f}ms ({fps_proc:4.1f} FPS) | Preset: [{enhancer.current_preset}]"
            cv2.putText(hud_bar, info_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            if time.time() < status_timer:
                cv2.putText(hud_bar, f">> {status_msg}", (dw - 540, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

            final_gui = np.vstack([display_img, hud_bar])
            cv2.imshow(win_name, final_gui)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q'), ord('Q')]:
                break
            elif key in [ord('z'), ord('Z')]: # Toggle Zoom Loupe
                show_zoom_loupe = not show_zoom_loupe
                status_msg = f"Zoom Loupe 2x: {'AKTIF' if show_zoom_loupe else 'NONAKTIF'}"
                status_timer = time.time() + 2.5
            elif key in [ord('p'), ord('P')]:
                preset_idx = (preset_idx + 1) % len(preset_keys)
                p_name = preset_keys[preset_idx]
                enhancer.apply_preset(p_name)
                cv2.setTrackbarPos("Deinterlace", win_name, 1 if enhancer.config["deinterlace"]["enabled"] else 0)
                cv2.setTrackbarPos("Anti-Moire", win_name, 1 if enhancer.config.get("anti_moire", {}).get("enabled", True) else 0)
                cv2.setTrackbarPos("Luma Smooth", win_name, 1 if enhancer.config.get("luma_smoothing", {}).get("enabled", True) else 0)
                cv2.setTrackbarPos("Chroma Denoise", win_name, int(enhancer.config["denoise"]["chroma_strength"]))
                cv2.setTrackbarPos("CLAHE Clip x10", win_name, int(enhancer.config["clahe"]["clip_limit"] * 10))
                cv2.setTrackbarPos("Sharpen x10", win_name, int(enhancer.config["sharpen"]["amount"] * 10))
                cv2.setTrackbarPos("Chroma Boost x10", win_name, int(enhancer.config["color"].get("chroma_boost", 1.40) * 10))
                cv2.setTrackbarPos("Target Pop", win_name, 1 if enhancer.config["color"].get("selective_target_boost", True) else 0)
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
                enhancer.apply_preset("ULTRA_TARP_V2")
                cv2.setTrackbarPos("Deinterlace", win_name, 1)
                cv2.setTrackbarPos("Anti-Moire", win_name, 1)
                cv2.setTrackbarPos("Luma Smooth", win_name, 1)
                cv2.setTrackbarPos("Chroma Denoise", win_name, 9)
                cv2.setTrackbarPos("CLAHE Clip x10", win_name, 15)
                cv2.setTrackbarPos("Sharpen x10", win_name, 15)
                cv2.setTrackbarPos("Chroma Boost x10", win_name, 14)
                cv2.setTrackbarPos("Target Pop", win_name, 1)
                cv2.setTrackbarPos("Auto White Bal", win_name, 1)
                cv2.setTrackbarPos("Crop Margin", win_name, 10)
                status_msg = "Reset ke default ULTRA_TARP_V2!"
                status_timer = time.time() + 2.5
            elif key in [ord('c'), ord('C'), 32]:
                ts = int(time.time())
                fn_before = f"easycap_v2_raw_{ts}.png"
                fn_after = f"easycap_v2_enhanced_{ts}.png"
                cv2.imwrite(fn_before, raw_frame)
                cv2.imwrite(fn_after, enhanced)
                print(f"[SNAPSHOT V2] Disimpan: {fn_before} & {fn_after}")
                status_msg = f"Snapshot disimpan ({fn_after})!"
                status_timer = time.time() + 3.0

    finally:
        if capture:
            capture.stop()
        if cap_file:
            cap_file.release()
        cv2.destroyAllWindows()


def create_easycap_stream_v2(camera_index=1, config_path="easycap_config_v2.json", preset="ULTRA_TARP_V2"):
    """
    Helper function V2 untuk integrasi ke 'teknofest_mission2.py':
    
    from easycap_video_enhancer_v2 import create_easycap_stream_v2
    cap, enhancer = create_easycap_stream_v2(camera_index=1, preset="ULTRA_TARP_V2")
    """
    enhancer = EasyCapEnhancerV2(config_path=config_path if os.path.exists(config_path) else None, preset_name=preset)
    cfg_cam = enhancer.config.get("camera", {})
    cam_idx = camera_index if camera_index is not None else cfg_cam.get("index", 1)
    capture = EasyCapCapture(
        camera_index=cam_idx,
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
    parser = argparse.ArgumentParser(description="EasyCap VRX Ultra Video Enhancer V2 (Teknofest UAV Tarp Detection)")
    parser.add_argument("--index", "-i", type=int, default=1, help="Indeks kamera capture card (default: 1)")
    parser.add_argument("--config", "-c", type=str, default="easycap_config_v2.json", help="Path file konfigurasi JSON V2")
    parser.add_argument("--preset", "-p", type=str, default="ULTRA_TARP_V2", choices=list(PRESETS_V2.keys()), help="Preset mode V2")
    parser.add_argument("--video", "-v", type=str, default=None, help="Path ke file video rekaman (opsional)")
    parser.add_argument("--image", "-img", type=str, default=None, help="Path ke file gambar (opsional)")
    parser.add_argument("--list", "-l", action="store_true", help="Pindai kamera aktif di Windows")
    args = parser.parse_args()

    if args.list:
        list_available_cameras()
        sys.exit(0)

    run_interactive_enhancer_v2(
        camera_index=args.index,
        config_file=args.config,
        video_file=args.video,
        image_file=args.image
    )
