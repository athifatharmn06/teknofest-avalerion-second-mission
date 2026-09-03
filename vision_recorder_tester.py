"""
================================================================================
     TEKNOFEST VISION - ENHANCER V2 VIDEO RECORDER & TESTING SUITE
        (Live EasyCap VRX + Enhancer V2 + YOLO Detection + Video Recording)
================================================================================

Fitur Utama:
1. Video Input: Kamera EasyCap VRX (DirectShow Windows) ATAU Video File (.mp4/.avi).
2. Image Processing: Enhancer V2 (Anti-Moire, Deinterlace, AWB, Luma Smoothing, CLAHE, Selective Saturation).
3. Object Detection: YOLO ONNX (v1_gazbmodel_exp.onnx) mendeteksi square_blue & square_red.
4. Video Recording [R]: Rekam video real-time langsung ke folder 'video_rec/' (rec_YYYYMMDD_HHMMSS.mp4).
5. Dua Mode Operasi:
   - MODE RECORDING (Live Camera): Tampilan live, deteksi target, rekam dengan [R].
   - MODE TESTING (Video File / Camera): Evaluasi model, play/pause [SPACE], step frame [D], 
     rewind [W], atur confidence [+]/[-], toggle Color Guard [G], toggle Enhancer [E].
================================================================================
"""

import sys
import os
import time
import datetime
import subprocess
import importlib.util
from pathlib import Path

# ==============================================================================
# AUTO-DEPENDENCY CHECKER & INSTALLER
# ==============================================================================
REQUIRED_PACKAGES = {
    "numpy": "numpy>=1.23.0",
    "cv2": "opencv-python>=4.8.0",
    "onnxruntime": "onnxruntime>=1.16.0"
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

# ==============================================================================
# KONFIGURASI GLOBAL
# ==============================================================================
YOLO_MODEL_PATH = "v1_gazbmodel_exp.onnx"
YOLO_INPUT_SIZE = 640
DEFAULT_CONF_THRESHOLD = 0.80     # Default 80%

# Folder Penyimpanan:
INPUT_VID_DIRS = ["vid_input", "vid input"] # Folder sumber video pengujian
RECORD_DIR = "video_rec"                    # Folder penyimpanan hasil rekaman kamera

DEFAULT_CAM_INDEX = 0             # Index 0 untuk EasyCap VRX di Windows

# HSV Color Guard
COLOR_GUARD_BLUE_LOWER = np.array([75, 30, 30], dtype=np.uint8)
COLOR_GUARD_BLUE_UPPER = np.array([145, 255, 255], dtype=np.uint8)

COLOR_GUARD_RED_LOWER1 = np.array([0, 30, 30], dtype=np.uint8)
COLOR_GUARD_RED_UPPER1 = np.array([18, 255, 255], dtype=np.uint8)
COLOR_GUARD_RED_LOWER2 = np.array([155, 30, 30], dtype=np.uint8)
COLOR_GUARD_RED_UPPER2 = np.array([180, 255, 255], dtype=np.uint8)

MIN_COLOR_GUARD_PIXELS = 15
MIN_ROI_COLOR_PIXELS = 5


# ==============================================================================
# CLASS: CleanEnhancerV2 (Self-Contained Ultra Enhancer Engine)
# ==============================================================================
class CleanEnhancerV2:
    """
    Engine Enhancer V2:
    - Bob Deinterlacing (Hapus garis sisir 100%)
    - Auto White Balance (Hapus tint warna)
    - Anti-Moire & RF Diagonal Suppressor (Hapus gelombang sinyal analog 5.8GHz)
    - Luma Surface Smoothing (Bersihkan noise bintik semut pada rumput/tanah)
    - Controlled CLAHE (Kontras lokal tajam)
    - High-Coring Edge Sharpening (Tajamkan garis batas terpal)
    - Selective Target Purity Boost (Warna terpal Biru & Merah dibuat pop)
    """
    def __init__(self):
        self.enabled = True
        self.clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8))

    def process(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled or frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]

        # 1. Border Crop (Hapus pinggiran kotor analog 8px)
        orig_h, orig_w = frame.shape[:2]
        crop_m = 8
        if h > 2 * crop_m and w > 2 * crop_m:
            frame = frame[crop_m:-crop_m, crop_m:-crop_m]
            h, w = frame.shape[:2]

        # 2. Bob Deinterlacing
        field = frame[1::2, :, :]
        deint = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)

        # 3. YCrCb Separation
        ycrcb = cv2.cvtColor(deint, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # 4. Auto White Balance
        cr_off = int(np.clip(np.mean(cr) - 128, -25, 25))
        cb_off = int(np.clip(np.mean(cb) - 128, -25, 25))
        if cr_off != 0: cr = cv2.subtract(cr, cr_off)
        if cb_off != 0: cb = cv2.subtract(cb, cb_off)

        # 5. Anti-Moire (Peredam Gelombang Diagonal RF 5.8GHz)
        y_base = cv2.GaussianBlur(y, (5, 5), 1.6)
        diff_m = cv2.absdiff(y, y_base)
        mask_strong = cv2.threshold(diff_m, 6, 255, cv2.THRESH_BINARY)[1]
        y_detail = cv2.subtract(y, y_base)
        y_clean_detail = np.where(mask_strong == 255, y_detail, 0)
        y = cv2.add(y_base, y_clean_detail)

        # 6. Luma Surface Smoothing (Hilangkan Derau Tanah/Rumput)
        blur_y = cv2.GaussianBlur(y, (5, 5), 1.5)
        diff_noise = cv2.absdiff(y, blur_y)
        mask_edges = cv2.threshold(diff_noise, 10, 255, cv2.THRESH_BINARY)[1]
        y = np.where(mask_edges == 255, y, blur_y)

        # 7. Controlled CLAHE
        y = self.clahe.apply(y)

        # 8. High-Coring Sharpening (Pertajam Outline Terpal)
        blur_s = cv2.GaussianBlur(y, (0, 0), sigmaX=0.9)
        diff_s = cv2.absdiff(y, blur_s)
        mask_p = cv2.threshold(diff_s, 7, 255, cv2.THRESH_BINARY)[1]
        boosted = cv2.addWeighted(y, 1.40, blur_s, -0.40, 0)
        y = np.where(mask_p == 255, boosted, y)

        # 9. Chroma Denoising
        cr = cv2.GaussianBlur(cr, (9, 9), 2.0)
        cb = cv2.GaussianBlur(cb, (9, 9), 2.0)

        # 10. Recombine
        merged = cv2.merge([y, cr, cb])
        bgr = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

        # 11. Selective Target Purity Boost (Biru Hue 95-135, Merah Hue 0-15 & 165-180)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = cv2.split(hsv)

        mask_blue = cv2.inRange(h_chan, 95, 135)
        mask_red1 = cv2.inRange(h_chan, 0, 15)
        mask_red2 = cv2.inRange(h_chan, 165, 180)
        mask_targets = cv2.bitwise_or(mask_blue, cv2.bitwise_or(mask_red1, mask_red2))

        s_boosted = np.clip(cv2.multiply(s_chan.astype(np.float32), 1.30), 0, 255).astype(np.uint8)
        s_final = np.where(mask_targets == 255, s_boosted, s_chan)

        enhanced = cv2.cvtColor(cv2.merge([h_chan, s_final, v_chan]), cv2.COLOR_HSV2BGR)

        # Kembalikan ke resolusi awal
        if enhanced.shape[0] != orig_h or enhanced.shape[1] != orig_w:
            enhanced = cv2.resize(enhanced, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        return enhanced


# ==============================================================================
# CLASS: YoloDetector (YOLO ONNX Inference + Color Guard)
# ==============================================================================
class YoloDetector:
    def __init__(self, model_path=YOLO_MODEL_PATH, conf_threshold=DEFAULT_CONF_THRESHOLD):
        self.conf_threshold = conf_threshold
        self.color_guard_enabled = True
        self.session = None
        self.input_name = None
        self.classes = {0: 'triangle_red', 1: 'hexagon_blue', 2: 'square_red', 3: 'square_blue'}
        self.load_model(model_path)

    def load_model(self, model_path):
        p = Path(model_path)
        if not p.is_file():
            p = Path(__file__).resolve().parent / model_path

        print(f"[YOLO] Memuat model ONNX: {p}...")
        try:
            self.session = ort.InferenceSession(str(p), providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            meta = self.session.get_modelmeta().custom_metadata_map
            if "names" in meta:
                import ast
                self.classes = ast.literal_eval(meta["names"])
            print(f"[YOLO] Model Berhasil Dimuat. Kelas: {self.classes}")
        except Exception as e:
            print(f"[YOLO ERROR] Gagal memuat ONNX: {e}")
            sys.exit(1)

    def _letterbox(self, frame, size=YOLO_INPUT_SIZE):
        h, w = frame.shape[:2]
        scale = min(size / h, size / w)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), (114, 114, 114), dtype=np.uint8)
        pad_x = (size - nw) // 2
        pad_y = (size - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        return canvas, scale, pad_x, pad_y

    def _check_roi_color(self, hsv, box, label):
        x1, y1, x2, y2 = box
        roi = hsv[y1:y2 + 1, x1:x2 + 1]
        if roi.size == 0: return False

        if "blue" in label:
            mask = cv2.inRange(roi, COLOR_GUARD_BLUE_LOWER, COLOR_GUARD_BLUE_UPPER)
        elif "red" in label:
            m1 = cv2.inRange(roi, COLOR_GUARD_RED_LOWER1, COLOR_GUARD_RED_UPPER1)
            m2 = cv2.inRange(roi, COLOR_GUARD_RED_LOWER2, COLOR_GUARD_RED_UPPER2)
            mask = cv2.bitwise_or(m1, m2)
        else:
            return True

        return cv2.countNonZero(mask) >= MIN_ROI_COLOR_PIXELS

    def detect(self, frame):
        """
        Menjalankan inferensi YOLO dan mengembalikan list bounding box deteksi.
        Format dict: {'label': str, 'conf': float, 'box': (x1,y1,x2,y2), 'center': (cx,cy)}
        """
        if frame is None or frame.size == 0 or self.session is None:
            return []

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Pre-check Color Guard
        if self.color_guard_enabled:
            mask_b = cv2.inRange(hsv, COLOR_GUARD_BLUE_LOWER, COLOR_GUARD_BLUE_UPPER)
            mask_r1 = cv2.inRange(hsv, COLOR_GUARD_RED_LOWER1, COLOR_GUARD_RED_UPPER1)
            mask_r2 = cv2.inRange(hsv, COLOR_GUARD_RED_LOWER2, COLOR_GUARD_RED_UPPER2)
            mask_r = cv2.bitwise_or(mask_r1, mask_r2)
            if cv2.countNonZero(mask_b) < MIN_COLOR_GUARD_PIXELS and cv2.countNonZero(mask_r) < MIN_COLOR_GUARD_PIXELS:
                return []

        # Letterbox Preprocess
        canvas, scale, pad_x, pad_y = self._letterbox(frame)
        tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = tensor[None]

        # ONNX Run
        preds = self.session.run(None, {self.input_name: tensor})[0][0]

        detections = []
        for row in preds:
            conf = float(row[4])
            if conf < self.conf_threshold:
                continue

            class_id = int(row[5])
            label = self.classes.get(class_id, str(class_id)).lower()

            x1, y1, x2, y2 = [float(v) for v in row[:4]]
            x1 = max(0, min(w - 1, int(round((x1 - pad_x) / scale))))
            x2 = max(0, min(w - 1, int(round((x2 - pad_x) / scale))))
            y1 = max(0, min(h - 1, int(round((y1 - pad_y) / scale))))
            y2 = max(0, min(h - 1, int(round((y2 - pad_y) / scale))))

            if x2 <= x1 or y2 <= y1:
                continue

            # ROI Color Guard Check
            if self.color_guard_enabled:
                if not self._check_roi_color(hsv, (x1, y1, x2, y2), label):
                    continue

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            detections.append({
                'label': label,
                'conf': conf,
                'box': (x1, y1, x2, y2),
                'center': center
            })

        return detections


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
        
        # Fallback ke AVI XVID jika mp4v gagal
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
        dur = int(time.time() - self.start_time)
        m, s = divmod(dur, 60)
        return f"{m:02d}:{s:02d} ({self.frame_count}f)"


def draw_fps_badge(image, fps, latency_ms=None, top_right_x=None, top_y=14):
    """
    Menggambar badge FPS Counter HUD modern semi-transparan:
    - Hijau neon jika FPS >= 24 (Sangat lancar)
    - Kuning jika FPS 15-23 (Cukup)
    - Merah jika FPS < 15 (Perhatian/Drop frame)
    """
    fps_val = max(0.0, fps)
    fps_text = f"{fps_val:4.1f} FPS"
    lat_text = f"{latency_ms:4.1f} ms" if latency_ms is not None and latency_ms > 0 else ""

    if fps_val >= 24.0:
        badge_c = (0, 255, 120)   # Neon Green
    elif fps_val >= 15.0:
        badge_c = (0, 220, 255)   # Amber / Yellow
    else:
        badge_c = (0, 60, 255)    # Bright Red

    bw, bh = (118, 42) if lat_text else (98, 28)
    h_img, w_img = image.shape[:2]
    rx = (w_img - bw - 14) if top_right_x is None else (top_right_x - bw)
    ry = top_y
    if rx < 0: rx = 10

    overlay = image.copy()
    cv2.rectangle(overlay, (rx, ry), (rx + bw, ry + bh), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)
    cv2.rectangle(image, (rx, ry), (rx + bw, ry + bh), badge_c, 1)

    cv2.putText(image, fps_text, (rx + 8, ry + 20), cv2.FONT_HERSHEY_DUPLEX, 0.52, badge_c, 1, cv2.LINE_AA)
    if lat_text:
        cv2.putText(image, lat_text, (rx + 8, ry + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 210, 210), 1, cv2.LINE_AA)


# ==============================================================================
# MAIN APPLICATION: VisionRecorderTesterApp
# ==============================================================================
class VisionRecorderTesterApp:
    def __init__(self, mode="live", source=DEFAULT_CAM_INDEX, model_path=YOLO_MODEL_PATH):
        self.mode = mode  # "live" atau "test"
        self.source = source
        self.model_path = model_path
        self.cap = None

        # Components
        self.enhancer = CleanEnhancerV2()
        self.detector = YoloDetector(model_path=self.model_path)
        self.recorder = VideoRecorder(output_dir=RECORD_DIR)

        # Display Mode (0: Full Enhanced+YOLO, 1: Side-by-Side, 2: Clean Enhanced, 3: Raw)
        self.view_mode = 0
        self.view_names = ["ENHANCED + YOLO", "SIDE-BY-SIDE [RAW | ENHANCED+YOLO]", "ENHANCED CLEAN", "RAW INPUT"]

        # Testing Mode Video Playback State & Smart Path Resolver
        if isinstance(self.source, str) and not str(self.source).isdigit():
            p = Path(self.source)
            if not p.is_file():
                for d in INPUT_VID_DIRS:
                    cand = Path(d) / p.name
                    if cand.is_file():
                        self.source = str(cand)
                        break
            self.is_video_file = Path(self.source).is_file()
        else:
            self.is_video_file = False

        self.is_paused = False
        self.loop_video = True
        self.total_frames = 0
        self.current_frame_pos = 0
        self.video_fps = 30.0
        self.updating_trackbar = False
        self.seek_requested_frame = None

        # Performance & Stats
        self.fps = 0.0
        self.proc_ms = 0.0
        self.status_msg = "Siap. Gunakan Slider Timeline untuk pilih frame, [SPACE] Pause/Play."
        self.status_timer = time.time() + 4.0

        # Setup Video Source
        self.init_source()

    def init_source(self):
        if self.is_video_file:
            print(f"[SOURCE] Membuka Video File: '{self.source}'...")
            self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
                print(f"[SOURCE] Video Dimuat: {w}x{h} @ {self.video_fps:.1f} FPS, Total: {self.total_frames} frames.")
        else:
            cam_idx = int(self.source) if str(self.source).isdigit() else 0
            print(f"[SOURCE] Membuka Kamera Index {cam_idx} (DirectShow Windows)...")
            if os.name == 'nt':
                self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(cam_idx)

            if not self.cap.isOpened():
                print(f"[SOURCE WARN] DirectShow gagal, mencoba default backend...")
                self.cap = cv2.VideoCapture(cam_idx)

        if not self.cap.isOpened():
            print(f"[SOURCE ERROR] Tidak dapat membuka sumber video: {self.source}")
            sys.exit(1)

    def draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            conf = det['conf']
            center = det['center']

            # Warna: Biru untuk target blue, Merah untuk target red
            if "blue" in label:
                color = (255, 140, 0)
            elif "red" in label:
                color = (0, 60, 255)
            else:
                color = (0, 255, 255)

            # Bounding Box Utama
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Corner Brackets
            d = 12
            cv2.line(frame, (x1, y1), (x1 + d, y1), (255, 255, 255), 2)
            cv2.line(frame, (x1, y1), (x1, y1 + d), (255, 255, 255), 2)
            cv2.line(frame, (x2, y2), (x2 - d, y2), (255, 255, 255), 2)
            cv2.line(frame, (x2, y2), (x2, y2 - d), (255, 255, 255), 2)

            # Center Point & Crosshair
            cv2.circle(frame, center, 4, (0, 255, 255), -1)
            cv2.line(frame, (center[0] - 8, center[1]), (center[0] + 8, center[1]), (0, 255, 255), 1)
            cv2.line(frame, (center[0], center[1] - 8), (center[0], center[1] + 8), (0, 255, 255), 1)

            # Label Tag Badge
            tag = f"{label.upper()} {conf * 100:.0f}%"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.50, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 5, max(16, y1 - 6)), cv2.FONT_HERSHEY_DUPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    def render_studio_dashboard(self, display_frame, detections):
        """
        Merender tampilan Studio Dashboard: Video di sisi kiri dan
        Panel Kontrol & Telemetri di sisi kanan tanpa teks bertumpukan.
        """
        h, w = display_frame.shape[:2]
        panel_w = 380
        canvas_h = max(h, 560)

        # Skala video agar tinggi video pas dengan tinggi canvas
        scaled_w = int(round(w * (canvas_h / h)))
        scaled_video = cv2.resize(display_frame, (scaled_w, canvas_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((canvas_h, scaled_w + panel_w, 3), dtype=np.uint8)
        canvas[:, :scaled_w] = scaled_video
        canvas[:, scaled_w:] = (22, 22, 22)  # Background sidebar abu-abu gelap

        # Indikator watermark merekam di pojok kiri atas video jika sedang merekam
        if self.recorder.is_recording:
            blink = int(time.time() * 2) % 2 == 0
            rec_c = (0, 0, 255) if blink else (0, 0, 160)
            cv2.circle(canvas, (24, 24), 8, rec_c, -1)
            cv2.putText(canvas, f"REC {self.recorder.get_status_str()}", (38, 30), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)

        # Badge FPS Counter Modern di Pojok Kanan Atas Area Video
        draw_fps_badge(canvas, fps=self.fps, latency_ms=self.proc_ms, top_right_x=scaled_w - 14, top_y=14)

        # Titik awal komponen sidebar
        p_x = scaled_w + 10
        card_w = panel_w - 20

        # 1. HEADER SIDEBAR
        cv2.rectangle(canvas, (scaled_w, 0), (scaled_w + panel_w, 44), (32, 32, 32), -1)
        cv2.line(canvas, (scaled_w, 44), (scaled_w + panel_w, 44), (55, 55, 55), 1)
        cv2.putText(canvas, "TEKNOFEST VISION V2", (p_x + 5, 29), cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 220, 255), 1, cv2.LINE_AA)

        # 2. SOURCE & PLAYBACK CARD
        y = 54
        card_h = 76 if self.is_video_file else 52
        cv2.rectangle(canvas, (p_x, y), (p_x + card_w, y + card_h), (32, 32, 32), -1)
        cv2.rectangle(canvas, (p_x, y), (p_x + card_w, y + card_h), (50, 50, 50), 1)

        src_name = Path(self.source).name if self.is_video_file else f"Live VRX (Index {self.source})"
        mode_label = "[TESTING: VIDEO]" if self.is_video_file else "[LIVE CAMERA]"
        cv2.putText(canvas, f"{mode_label} {src_name[:20]}", (p_x + 10, y + 18), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        if self.is_video_file and self.total_frames > 0:
            cur_sec = int(self.current_frame_pos / max(1.0, self.video_fps))
            tot_sec = int(self.total_frames / max(1.0, self.video_fps))
            pct = int((self.current_frame_pos / max(1, self.total_frames)) * 100)
            status_tag = "PAUSED" if self.is_paused else "PLAYING"
            tag_color = (0, 200, 255) if self.is_paused else (0, 255, 0)
            time_str = f"{cur_sec//60:02d}:{cur_sec%60:02d} / {tot_sec//60:02d}:{tot_sec%60:02d}"
            cv2.putText(canvas, f"{time_str} | F:{self.current_frame_pos}/{self.total_frames} ({pct}%) [{status_tag}]", (p_x + 10, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, tag_color, 1, cv2.LINE_AA)

            # Timeline Progress Bar
            bar_x, bar_y, bar_w, bar_h = p_x + 10, y + 46, card_w - 20, 6
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (55, 55, 55), -1)
            fill_w = int(bar_w * (pct / 100.0))
            if fill_w > 0:
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 220, 255), -1)

            cv2.putText(canvas, ">> Geser Slider 'Timeline' di atas jendela", (p_x + 10, y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 240, 220), 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "Live Video Stream: 30 FPS", (p_x + 10, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)

        # 3. RECORDING STATUS CARD
        y += card_h + 8
        rec_box_h = 42
        if self.recorder.is_recording:
            cv2.rectangle(canvas, (p_x, y), (p_x + card_w, y + rec_box_h), (0, 0, 160), -1)
            cv2.rectangle(canvas, (p_x, y), (p_x + card_w, y + rec_box_h), (0, 0, 255), 1)
            blink = int(time.time() * 2) % 2 == 0
            cv2.circle(canvas, (p_x + 18, y + 21), 6, (0, 0, 255) if blink else (255, 255, 255), -1)
            cv2.putText(canvas, f"MEREKAM: {self.recorder.get_status_str()}", (p_x + 32, y + 27), cv2.FONT_HERSHEY_DUPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (p_x, y), (p_x + card_w, y + rec_box_h), (35, 35, 35), -1)
            cv2.rectangle(canvas, (p_x, y), (p_x + card_w, y + rec_box_h), (55, 55, 55), 1)
            cv2.circle(canvas, (p_x + 18, y + 21), 6, (120, 120, 120), -1)
            cv2.putText(canvas, "RECORDER: STANDBY [Tekan R]", (p_x + 32, y + 27), cv2.FONT_HERSHEY_DUPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

        # 4. METRICS CARD (FPS, Latency, Conf, Enhancer, View)
        y += rec_box_h + 8
        met_h = 72
        cv2.rectangle(canvas, (p_x, y), (p_x + card_w, y + met_h), (30, 30, 30), -1)
        cv2.rectangle(canvas, (p_x, y), (p_x + card_w, y + met_h), (50, 50, 50), 1)
        model_name = Path(self.model_path).name
        cv2.putText(canvas, f"FPS: {self.fps:4.1f} | Latency: {self.proc_ms:4.1f}ms | {model_name}", (p_x + 10, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Conf: {self.detector.conf_threshold*100:.0f}% (+/-) | Guard: {'ON' if self.detector.color_guard_enabled else 'OFF'} (G)", (p_x + 10, y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Enhancer: {'ON' if self.enhancer.enabled else 'OFF'} (E) | View: {self.view_names[self.view_mode][:9]} (TAB)", (p_x + 10, y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 255, 180), 1, cv2.LINE_AA)

        # 5. DETECTIONS CARD
        y += met_h + 8
        det_h = 78
        cv2.rectangle(canvas, (p_x, y), (p_x + card_w, y + det_h), (30, 30, 30), -1)
        cv2.rectangle(canvas, (p_x, y), (p_x + card_w, y + det_h), (50, 50, 50), 1)
        cv2.putText(canvas, f"TARGET TERDETEKSI ({len(detections)}):", (p_x + 10, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)

        if detections:
            dy = y + 25
            for det in detections[:2]:
                lbl = det['label'].upper()
                cnf = int(det['conf'] * 100)
                cx, cy = det['center']
                box_bg = (80, 40, 0) if "BLUE" in lbl else (0, 25, 90)
                cv2.rectangle(canvas, (p_x + 8, dy), (p_x + card_w - 8, dy + 22), box_bg, -1)
                cv2.putText(canvas, f"{lbl} {cnf}% @ ({cx}, {cy})", (p_x + 14, dy + 16), cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                dy += 24
        else:
            cv2.putText(canvas, "Tidak ada target di dalam frame", (p_x + 10, y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)

        # 6. KEYBOARD SHORTCUTS CHEATSHEET
        y += det_h + 8
        sc_h = 108
        cv2.rectangle(canvas, (p_x, y), (p_x + card_w, y + sc_h), (28, 28, 28), -1)
        cv2.rectangle(canvas, (p_x, y), (p_x + card_w, y + sc_h), (45, 45, 45), 1)
        cv2.putText(canvas, "KONTROL KEYBOARD:", (p_x + 10, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)

        shortcuts = [
            ("[R] Rekam Video", "[SPACE] Pause/Play"),
            ("[D] Step 1 Frame", "[W] Restart Video"),
            ("[TAB] Ganti View", "[C] Snapshot Foto"),
            ("[+] / [-] Conf 5%", "[G] HSV Guard"),
            ("[E] Enhancer V2", "[Q] Keluar"),
        ]
        sy = y + 32
        for col1, col2 in shortcuts:
            cv2.putText(canvas, col1, (p_x + 10, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(canvas, col2, (p_x + 185, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
            sy += 15

        # 7. STATUS TOAST BANNER (Bawah)
        y += sc_h + 8
        if y + 36 <= canvas_h:
            cv2.rectangle(canvas, (p_x, y), (p_x + card_w, y + 34), (20, 36, 36), -1)
            cv2.rectangle(canvas, (p_x, y), (p_x + card_w, y + 34), (0, 160, 160), 1)
            toast_text = f">> {self.status_msg}"[:42]
            cv2.putText(canvas, toast_text, (p_x + 10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 240, 220), 1, cv2.LINE_AA)

        return canvas

    def run(self):
        win_name = "Teknofest Vision V2 - Video Recorder & Detector Suite"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720)

        # Trackbar Slider untuk Memilih Frame / Waktu Video
        trackbar_name = "Timeline"
        if self.is_video_file and self.total_frames > 0:
            def on_timeline_slide(val):
                if not self.updating_trackbar:
                    self.seek_requested_frame = val

            cv2.createTrackbar(trackbar_name, win_name, 0, max(1, self.total_frames - 1), on_timeline_slide)

        print("\n" + "=" * 65)
        print(f" SUITE VISION V2 BERJALAN [Mode: {self.mode.upper()}]:")
        if self.is_video_file:
            print("  [SLIDER]    : Geser Slider 'Timeline' di atas jendela untuk pilih frame")
            print("  [SPACE]     : Pause / Resume Video (Mode Testing)")
            print("  [D] / [A]   : Step 1 Frame Maju / Mundur (Saat Pause)")
            print("  [W]         : Rewind / Restart Video dari awal")
        print("  [R]         : MULAI / STOP REKAM VIDEO -> folder 'video_rec/'")
        print("  [TAB]       : Ganti Tampilan (Fullscreen / Side-by-Side [RAW|V2])")
        print("  [+] / [-]   : Naikkan / Turunkan Confidence YOLO (+/- 5%)")
        print("  [G]         : Toggle HSV Color Guard ON/OFF")
        print("  [E]         : Toggle Enhancer V2 Filter ON/OFF")
        print("  [C]         : Simpan Snapshot Foto")
        print("  [Q] / [ESC] : Keluar")
        print("=" * 65 + "\n")

        fps_timer = time.time()
        last_raw = None

        try:
            while True:
                # 1. Cek Apakah Ada Permintaan Seek dari Slider Trackbar
                if self.seek_requested_frame is not None:
                    target_f = max(0, min(self.total_frames - 1, self.seek_requested_frame))
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
                    self.current_frame_pos = target_f
                    self.seek_requested_frame = None
                    ret, raw_frame = self.cap.read()
                    if ret and raw_frame is not None:
                        last_raw = raw_frame
                        self.current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                # 2. Baca Frame Video / Kamera
                elif not self.is_paused or last_raw is None:
                    ret, raw_frame = self.cap.read()
                    if not ret or raw_frame is None:
                        if self.is_video_file and self.loop_video:
                            # Loop video file jika selesai
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, raw_frame = self.cap.read()
                        else:
                            time.sleep(0.01)
                            continue

                    if self.is_video_file:
                        self.current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                        # Update posisi slider trackbar secara sinkron
                        self.updating_trackbar = True
                        cv2.setTrackbarPos(trackbar_name, win_name, self.current_frame_pos)
                        self.updating_trackbar = False
                    last_raw = raw_frame
                else:
                    raw_frame = last_raw.copy()
                    time.sleep(0.02)

                # 3. Pipeline Enhancer V2
                t0 = time.perf_counter()
                enhanced_frame = self.enhancer.process(raw_frame)
                
                # 4. YOLO Object Detection
                detections = self.detector.detect(enhanced_frame)
                self.proc_ms = (time.perf_counter() - t0) * 1000.0

                # Hitung FPS
                now = time.time()
                dt = now - fps_timer
                if dt > 0:
                    inst_fps = 1.0 / dt
                    self.fps = (0.85 * self.fps) + (0.15 * inst_fps) if self.fps > 0 else inst_fps
                fps_timer = now

                # Buat Frame Visual dengan Bounding Box
                annotated_frame = enhanced_frame.copy()
                self.draw_detections(annotated_frame, detections)

                # 5. Rekam Frame Jika Tombol [R] Aktif
                if self.recorder.is_recording:
                    # Simpan frame dengan anotasi deteksi
                    self.recorder.write(annotated_frame)

                # 6. Render Sesuai Mode Tampilan (View Mode)
                out_h, out_w = annotated_frame.shape[:2]
                raw_matched = cv2.resize(raw_frame, (out_w, out_h))

                display_img = None
                if self.view_mode == 0:
                    # Enhanced + YOLO Fullscreen
                    display_img = annotated_frame
                elif self.view_mode == 1:
                    # Side-by-Side: [RAW ANALOG] | [ENHANCED + YOLO]
                    lbl_raw = raw_matched.copy()
                    cv2.putText(lbl_raw, "[RAW INPUT]", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 255), 2)
                    lbl_enh = annotated_frame.copy()
                    cv2.putText(lbl_enh, "[V2 ENHANCED + YOLO]", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 0), 2)
                    display_img = np.hstack([lbl_raw, lbl_enh])
                elif self.view_mode == 2:
                    # Enhanced Clean (Tanpa bounding box)
                    display_img = enhanced_frame
                else:
                    # Raw Input
                    display_img = raw_matched

                # Render Studio Dashboard (Video + Sidebar tanpa font bertumpuk)
                final_view = self.render_studio_dashboard(display_img, detections)
                cv2.imshow(win_name, final_view)

                # 7. Keyboard Shortcuts
                key = cv2.waitKey(1) & 0xFF
                if key in [27, ord('q'), ord('Q')]: # Keluar
                    break
                elif key in [ord('r'), ord('R')]: # Toggle Recording
                    saved = self.recorder.toggle(width=out_w, height=out_h, fps=30.0)
                    if self.recorder.is_recording:
                        self.status_msg = f"MEREKAM KE: {Path(saved).name}"
                    else:
                        self.status_msg = f"REKAMAN DISIMPAN: {Path(saved).name}"
                    self.status_timer = time.time() + 4.0
                elif key == 32: # SPACE: Pause / Play (Testing Mode)
                    self.is_paused = not self.is_paused
                    self.status_msg = "Video DI-PAUSE" if self.is_paused else "Video DILANJUTKAN"
                    self.status_timer = time.time() + 2.0
                elif key in [ord('d'), ord('D')]: # Step 1 Frame Maju
                    if self.is_video_file and self.total_frames > 0:
                        new_f = min(self.total_frames - 1, self.current_frame_pos + 1)
                        self.seek_requested_frame = new_f
                        self.updating_trackbar = True
                        cv2.setTrackbarPos(trackbar_name, win_name, new_f)
                        self.updating_trackbar = False
                        self.status_msg = f"Step Frame Maju: {new_f}"
                        self.status_timer = time.time() + 1.5
                elif key in [ord('a'), ord('A')]: # Step 1 Frame Mundur
                    if self.is_video_file and self.total_frames > 0:
                        new_f = max(0, self.current_frame_pos - 2)
                        self.seek_requested_frame = new_f
                        self.updating_trackbar = True
                        cv2.setTrackbarPos(trackbar_name, win_name, new_f)
                        self.updating_trackbar = False
                        self.status_msg = f"Step Frame Mundur: {new_f}"
                        self.status_timer = time.time() + 1.5
                elif key in [ord('w'), ord('W')]: # Rewind / Restart ke frame 0
                    if self.is_video_file:
                        self.seek_requested_frame = 0
                        self.updating_trackbar = True
                        cv2.setTrackbarPos(trackbar_name, win_name, 0)
                        self.updating_trackbar = False
                        self.status_msg = "Video Di-restart dari Awal (Frame 0)"
                        self.status_timer = time.time() + 2.0
                elif key in [9, ord('t'), ord('T')]: # TAB: Ganti View Mode
                    self.view_mode = (self.view_mode + 1) % len(self.view_names)
                    self.status_msg = f"Tampilan: {self.view_names[self.view_mode]}"
                    self.status_timer = time.time() + 2.0
                elif key in [ord('+'), ord('=')]: # Confidence UP
                    self.detector.conf_threshold = min(0.95, self.detector.conf_threshold + 0.05)
                    self.status_msg = f"Confidence Naik: {self.detector.conf_threshold*100:.0f}%"
                    self.status_timer = time.time() + 2.0
                elif key in [ord('-'), ord('_')]: # Confidence DOWN
                    self.detector.conf_threshold = max(0.20, self.detector.conf_threshold - 0.05)
                    self.status_msg = f"Confidence Turun: {self.detector.conf_threshold*100:.0f}%"
                    self.status_timer = time.time() + 2.0
                elif key in [ord('g'), ord('G')]: # Toggle Color Guard
                    self.detector.color_guard_enabled = not self.detector.color_guard_enabled
                    self.status_msg = f"HSV Color Guard: {'AKTIF' if self.detector.color_guard_enabled else 'NONAKTIF'}"
                    self.status_timer = time.time() + 2.0
                elif key in [ord('e'), ord('E')]: # Toggle Enhancer
                    self.enhancer.enabled = not self.enhancer.enabled
                    self.status_msg = f"Enhancer V2: {'AKTIF' if self.enhancer.enabled else 'BYPASS (OFF)'}"
                    self.status_timer = time.time() + 2.0
                elif key in [ord('c'), ord('C')]: # Snapshot
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    snap_fn = str(Path(RECORD_DIR) / f"snap_{ts}.png")
                    cv2.imwrite(snap_fn, annotated_frame)
                    print(f"[SNAPSHOT] Foto disimpan ke: {snap_fn}")
                    self.status_msg = f"Foto disimpan ke {Path(snap_fn).name}!"
                    self.status_timer = time.time() + 3.0

        finally:
            if self.recorder.is_recording:
                self.recorder.stop()
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("[SUITE] Selesai & Ditutup dengan aman.")


# ==============================================================================
# MENU SELEKSI & ENTRY POINT
# ==============================================================================
def find_available_videos():
    candidates = []
    # 1. Prioritas Utama: Cari di folder sumber video input ('vid_input' atau 'vid input')
    for d in INPUT_VID_DIRS:
        p_in = Path(d)
        if p_in.exists():
            for ext in ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.ts"]:
                candidates.extend(list(p_in.glob(ext)))

    # 2. Prioritas Kedua: Cari di folder rekaman ('video_rec') jika ingin menguji rekaman sebelumnya
    p_rec = Path(RECORD_DIR)
    if p_rec.exists():
        for ext in ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.ts"]:
            candidates.extend(list(p_rec.glob(ext)))

    # 3. Prioritas Ketiga: Current Directory
    for ext in ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.ts"]:
        candidates.extend(list(Path(".").glob(ext)))

    # Hilangkan duplikat dan urutkan
    unique_candidates = []
    seen = set()
    for c in candidates:
        abs_p = c.resolve()
        if abs_p not in seen:
            seen.add(abs_p)
            unique_candidates.append(c)

    return unique_candidates


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


def select_model_interactive(default_model=YOLO_MODEL_PATH):
    models = find_available_models(default_model)
    if not models:
        return default_model

    print("\n" + "=" * 65)
    print(" Pilih Model Deteksi YOLO (.onnx):")
    for idx, m in enumerate(models, 1):
        tag = " [DEFAULT]" if m.name == default_model or idx == 1 else ""
        print(f"   [{idx}] {m.name}{tag}")
    print("=" * 65)

    m_choice = input(f"Pilih Model (1-{len(models)}/path, tekan [ENTER] untuk default [{models[0].name}]): ").strip()
    if m_choice.isdigit() and 1 <= int(m_choice) <= len(models):
        return str(models[int(m_choice) - 1])
    elif m_choice and Path(m_choice).is_file():
        return m_choice
    else:
        return str(models[0])


def main_menu():
    print("\n" + "=" * 65)
    print("   TEKNOFEST VISION - ENHANCER V2 RECORDER & TESTING SUITE")
    print("=" * 65)
    print(" Pilih Mode:")
    print("   [1] LIVE RECORDING & DETECTION (Kamera EasyCap VRX)")
    print("       -> Rekam video live ke folder 'video_rec/' dengan tombol [R]")
    print("   [2] TESTING MODE: Menggunakan Video File dari folder 'vid_input/'")
    print("       -> Uji deteksi model YOLO & filter pada rekaman video")
    print("   [3] TESTING MODE: Menggunakan Kamera Live (Evaluasi Parameter)")
    print("=" * 65)

    choice = input("Pilihan (1/2/3, default [1]): ").strip()
    if choice == "2":
        videos = find_available_videos()
        if videos:
            print("\nVideo Ditemukan:")
            for idx, v in enumerate(videos, 1):
                folder_tag = f"[{v.parent.name}]"
                print(f"  [{idx}] {folder_tag:15s} {v.name}")
            v_choice = input(f"Pilih nomor video (1-{len(videos)}, default [1]): ").strip()
            if v_choice.isdigit() and 1 <= int(v_choice) <= len(videos):
                video_file = str(videos[int(v_choice) - 1])
            elif v_choice and os.path.exists(v_choice):
                video_file = v_choice
            else:
                video_file = str(videos[0])
        else:
            print("\n[INFO] Folder 'vid_input/' belum memiliki file video (.mp4/.avi).")
            print("Silakan salin video pengujian ke dalam folder 'vid_input/'.")
            video_file = input("Atau ketik path file video sekarang (tekan ENTER untuk batal): ").strip()
            if not video_file or not os.path.exists(video_file):
                print("[INFO] Beralih ke Kamera Live karena file video tidak ditemukan.\n")
                video_file = DEFAULT_CAM_INDEX

        mode, source = "test", video_file
    elif choice == "3":
        c_idx = input(f"Masukkan Index Kamera (default [{DEFAULT_CAM_INDEX}]): ").strip() or str(DEFAULT_CAM_INDEX)
        mode, source = "test", int(c_idx)
    else:
        c_idx = input(f"Masukkan Index Kamera EasyCap VRX (default [{DEFAULT_CAM_INDEX}]): ").strip() or str(DEFAULT_CAM_INDEX)
        mode, source = "live", int(c_idx)

    # Pilih Model Deteksi YOLO (.onnx)
    selected_model = select_model_interactive()
    return mode, source, selected_model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Teknofest Enhancer V2 Video Recorder & Testing Suite")
    parser.add_argument("--mode", "-m", type=str, default=None, choices=["live", "test"], help="Mode operasi")
    parser.add_argument("--cam", "-c", type=int, default=DEFAULT_CAM_INDEX, help="Index kamera EasyCap VRX")
    parser.add_argument("--video", "-v", type=str, default=None, help="Path berkas video untuk testing")
    parser.add_argument("--model", "-md", type=str, default=None, help="Path berkas model YOLO (.onnx)")
    args = parser.parse_args()

    mode = args.mode
    source = args.cam
    model_path = args.model

    if args.video is not None:
        mode = "test"
        source = args.video

    if mode is None or model_path is None:
        menu_mode, menu_source, menu_model = main_menu()
        if mode is None: mode = menu_mode
        if args.video is None and args.cam == DEFAULT_CAM_INDEX: source = menu_source
        if model_path is None: model_path = menu_model

    if model_path is None:
        model_path = YOLO_MODEL_PATH

    app = VisionRecorderTesterApp(mode=mode, source=source, model_path=model_path)
    app.run()
