"""
================================================================================
             TEKNOFEST - TEST SERVO 7 & DETEKSI DROPPING (BLUE SQUARE)
================================================================================

Deskripsi:
- Menghubungkan Telemetri (RFD900 / Serial / SITL) & Kamera (Foxeer USB / Gazebo).
- Menyediakan MAVLink Forwarder bi-directional agar Mission Planner bisa connect
  secara simultan via UDP 14550 & LAN Broadcast.
- Mengambil & menyimpan data default PWM Servo Channel 7 dari Flight Controller.
- Menjalankan deteksi vision YOLO (v1_gazbmodel_exp.onnx).
- Jika mendeteksi target "square_blue" (Biru Kotak), otomatis mentrigger
  Servo Channel 7 ke PWM 2200.
- Kontrol Keyboard di GUI:
    [T] : Manual Test Trigger Servo 7 -> 2200 PWM
    [R] : Reset Servo 7 -> Default PWM
    [Q] : Keluar (Quit)

================================================================================
"""

import sys
import time
import math
import socket
import threading
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort
from pymavlink import mavutil

# --- Gazebo Transport (Optional untuk mode simulasi) ---
HAVE_GZ = False
try:
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image
    HAVE_GZ = True
except ImportError:
    try:
        from gz.transport import Node
        from gz.msgs.image_pb2 import Image
        HAVE_GZ = True
    except ImportError:
        HAVE_GZ = False


# ==============================================================================
# KONFIGURASI PENGGUNA
# ==============================================================================

# Target Servo Channel & PWM
TARGET_SERVO_CHANNEL = 7
TRIGGER_DROP_PWM = 2200       # Nilai PWM saat target biru kotak terdeteksi
FALLBACK_DEFAULT_PWM = 1000   # Nilai PWM default jika parameter FC tidak terbaca

# MAVLink Networking & Forwarding Ports
DEFAULT_SIM_CONNECTION_STRING = 'udp:localhost:14551'
DEFAULT_MP_FORWARD_PORT = 14550
DEFAULT_SCRIPT_PORT = 14551
DEFAULT_SERIAL_BAUD_RATE = 57600

# YOLO Configuration
YOLO_MODEL_PATH = "v1_gazbmodel_exp.onnx"
YOLO_INPUT_SIZE = 640
YOLO_CONF_THRESHOLD = 0.85  # Minimal confidence 85% (0.85)
TARGET_CLASS_NAME = "square_blue"

# --- COLOR GUARD DETECTION FILTER (OpenCV HSV) ---
# Wide HSV range for Blue targets (OpenCV H: 0-179, S: 0-255, V: 0-255)
# Rentang lebar agar mendeteksi variasi warna biru (cyan, navy, sky blue) di kondisi pencahayaan outdoor
COLOR_GUARD_BLUE_LOWER = np.array([75, 30, 30], dtype=np.uint8)
COLOR_GUARD_BLUE_UPPER = np.array([145, 255, 255], dtype=np.uint8)

# Wide HSV range for Red targets (wraps around hue 0 and 180)
# Rentang lebar untuk variasi warna merah (oranye-merah, merah terang, merah tua/marun)
COLOR_GUARD_RED_LOWER1 = np.array([0, 30, 30], dtype=np.uint8)
COLOR_GUARD_RED_UPPER1 = np.array([18, 255, 255], dtype=np.uint8)
COLOR_GUARD_RED_LOWER2 = np.array([155, 30, 30], dtype=np.uint8)
COLOR_GUARD_RED_UPPER2 = np.array([180, 255, 255], dtype=np.uint8)

# Parameter Filter Warna
COLOR_GUARD_ENABLED = True
MIN_COLOR_GUARD_PIXELS = 15   # Minimal pixel warna di frame sebelum inferensi
MIN_ROI_COLOR_PIXELS = 5      # Minimal pixel warna di dalam ROI bounding box

# Kamera Resolution
IMG_W = 640
IMG_H = 480
DEBUG_WINDOW_NAME = "Teknofest - Test Servo 7 & Dropping Detection"


# ==============================================================================
# Helper MAVLink Serial Multiplexer & Port Forwarding
# ==============================================================================
def start_mavlink_forwarder(serial_port, baud_rate, mp_port=DEFAULT_MP_FORWARD_PORT, script_port=DEFAULT_SCRIPT_PORT):
    """
    Membuka port serial RFD900 dan membuat bi-directional UDP bridge transparan ke:
    - UDP 14550 (Mission Planner & LAN Broadcast)
    - UDP 14551 (Python Script ini)
    """
    import serial
    print(f"\n[MAVLINK FORWARDER] Membuka serial port {serial_port} @ {baud_rate} baud...")
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=0.005)
    except Exception as e:
        print(f"[MAVLINK FORWARDER ERROR] Gagal membuka {serial_port}: {e}")
        return False, None

    bridge_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    bridge_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    bridge_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    bridge_sock.setblocking(False)
    bridge_sock.bind(('127.0.0.1', 0))

    known_clients = {
        ('127.0.0.1', mp_port),
        ('127.0.0.1', script_port),
        ('255.255.255.255', mp_port),
    }
    clients_lock = threading.Lock()
    running = True

    def serial_to_udp():
        while running:
            try:
                waiting = ser.in_waiting
                if waiting > 0:
                    raw = ser.read(waiting)
                    if raw:
                        with clients_lock:
                            targets = list(known_clients)
                        for target in targets:
                            try:
                                bridge_sock.sendto(raw, target)
                            except Exception:
                                pass
                else:
                    time.sleep(0.001)
            except Exception:
                time.sleep(0.002)

    def udp_to_serial():
        while running:
            try:
                data, addr = bridge_sock.recvfrom(4096)
                if data:
                    with clients_lock:
                        if addr not in known_clients:
                            known_clients.add(addr)
                    ser.write(data)
            except (BlockingIOError, socket.error):
                time.sleep(0.001)
            except Exception:
                time.sleep(0.002)

    t1 = threading.Thread(target=serial_to_udp, daemon=True, name="Forwarder-Ser2UDP")
    t2 = threading.Thread(target=udp_to_serial, daemon=True, name="Forwarder-UDP2Ser")
    t1.start()
    t2.start()

    print(
        f"[MAVLINK FORWARDER] AKTIF: Aliran data dua arah antara {serial_port} dan:\n"
        f"  -> Mission Planner: UDP {mp_port} (Localhost & LAN Broadcast)\n"
        f"  -> Script Testing : UDP {script_port}\n"
    )
    return True, f"udpin:127.0.0.1:{script_port}"


# ==============================================================================
# Helper YOLO Image Letterbox & Preprocess
# ==============================================================================
def letterbox(frame, size=YOLO_INPUT_SIZE, color=(114, 114, 114)):
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    new_width, new_height = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    pad_x = (size - new_width) // 2
    pad_y = (size - new_height) // 2
    canvas[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
    return canvas, scale, pad_x, pad_y


def preprocess_yolo(frame):
    image, scale, pad_x, pad_y = letterbox(frame)
    tensor = image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return tensor[None], scale, pad_x, pad_y


# ==============================================================================
# Menu Pemilihan Mode & Port
# ==============================================================================
def select_operating_mode():
    print("\n" + "=" * 65)
    print("      TEKNOFEST - TEST SERVO 7 & DETEKSI DROPPING (BLUE SQUARE)")
    print("=" * 65)
    print(" Pilih Mode Operasi:")
    print("   [1] REAL HARDWARE MODE (RFD900 USB Serial + Foxeer USB Camera)")
    print("   [2] SIMULATION MODE (Gazebo / SITL UDP:14551)")
    print("=" * 65)

    choice = input("Masukkan pilihan (1 atau 2, default [1]): ").strip()
    if choice == "2":
        return "simulation"
    return "real"


def configure_real_hardware():
    print("\n--- Konfigurasi Hardware Nyata ---")
    com_ports = []
    try:
        import serial.tools.list_ports
        com_ports = list(serial.tools.list_ports.comports())
    except ImportError:
        pass

    if com_ports:
        print("Daftar Port COM Terdeteksi:")
        for idx, p in enumerate(com_ports, start=1):
            print(f"  [{idx}] {p.device} - {p.description}")
        selected_port = com_ports[0].device
        p_choice = input(f"Pilih Port COM (1-{len(com_ports)}, default [1: {selected_port}]): ").strip()
        if p_choice.isdigit() and 1 <= int(p_choice) <= len(com_ports):
            selected_port = com_ports[int(p_choice) - 1].device
    else:
        selected_port = input("Masukkan Port COM (misal COM7, default [COM7]): ").strip() or "COM7"

    baud_input = input(f"Masukkan RFD900 baud rate (default [{DEFAULT_SERIAL_BAUD_RATE}]): ").strip() or str(DEFAULT_SERIAL_BAUD_RATE)
    baud_rate = int(baud_input)

    cam_input = input("Masukkan Index USB Video Capture Foxeer camera (default [0]): ").strip() or "0"
    cam_index = int(cam_input)

    return selected_port, baud_rate, cam_index


# ==============================================================================
# Main ServoTestRunner Class
# ==============================================================================
class ServoTestRunner:
    def __init__(self, mode="real", connection_string=DEFAULT_SIM_CONNECTION_STRING, baud_rate=DEFAULT_SERIAL_BAUD_RATE, camera_index=0):
        self.mode = mode
        self.connection_string = connection_string
        self.baud_rate = baud_rate
        self.camera_index = camera_index

        # Status Servo 7
        self.servo_channel = TARGET_SERVO_CHANNEL
        self.servo_default_pwm = FALLBACK_DEFAULT_PWM
        self.servo_current_pwm = FALLBACK_DEFAULT_PWM
        self.servo_triggered = False
        self.last_trigger_time = 0
        self.trigger_count = 0

        # Status Telemetri
        self.current_alt = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.debug_fps = 0.0
        self.debug_last_frame_time = None
        self.gui_enabled = True

        # Frame & Deteksi
        self.latest_img = None
        self.new_frame = False
        self.last_detection_info = None
        self.blue_detected = False
        self.consecutive_blue_frames = 0

        # ----------------------------------------------------------------------
        # 1. Koneksi MAVLink
        # ----------------------------------------------------------------------
        print(f"\nMenghubungkan ke Flight Controller di {self.connection_string}...")
        if self.connection_string.upper().startswith("COM") or "/dev/" in self.connection_string:
            self.master = mavutil.mavlink_connection(self.connection_string, baud=self.baud_rate, source_system=1, source_component=255)
        else:
            self.master = mavutil.mavlink_connection(self.connection_string, source_system=1, source_component=255)

        print("Menunggu MAVLink Heartbeat...")
        self.master.wait_heartbeat()
        print(f"Flight Controller Terhubung! (SysID: {self.master.target_system}, CompID: {self.master.target_component})")

        # ----------------------------------------------------------------------
        # 2. Ambil Parameter Default Servo Channel 7 dari FC
        # ----------------------------------------------------------------------
        self.fetch_servo7_parameters()

        # ----------------------------------------------------------------------
        # 3. Muat Model YOLO ONNX
        # ----------------------------------------------------------------------
        self.load_yolo_model()

        # ----------------------------------------------------------------------
        # 4. Inisialisasi Kamera
        # ----------------------------------------------------------------------
        if self.mode == "real":
            self.start_usb_camera()
        else:
            self.init_simulation_camera()

    def fetch_servo7_parameters(self):
        """Membaca parameter SERVO7_TRIM, SERVO7_MIN, SERVO7_MAX dari Flight Controller"""
        print("\n[SERVO 7] Mengambil konfigurasi parameter Servo Channel 7 dari FC...")
        try:
            target_sys = getattr(self.master, 'target_system', 1)
            target_comp = getattr(self.master, 'target_component', 0)

            # Request parameter
            if hasattr(self.master, 'param_fetch_all'):
                self.master.param_fetch_all()
            else:
                try:
                    self.master.param_request_list_send()
                except TypeError:
                    self.master.param_request_list_send(target_sys, target_comp)

            # Tunggu beberapa detik untuk membaca parameter spesifik SERVO7
            tstart = time.time()
            found_params = {}
            while time.time() - tstart < 3.0:
                msg = self.master.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.3)
                if msg:
                    p_id = msg.param_id
                    if isinstance(p_id, bytes):
                        p_id = p_id.decode('utf-8', errors='ignore')
                    p_id = p_id.strip('\x00').upper()
                    if p_id.startswith("SERVO7_"):
                        found_params[p_id] = msg.param_value

            trim_val = found_params.get("SERVO7_TRIM")
            min_val = found_params.get("SERVO7_MIN")
            func_val = found_params.get("SERVO7_FUNCTION")

            if trim_val is not None and float(trim_val) > 500:
                self.servo_default_pwm = int(float(trim_val))
            elif min_val is not None and float(min_val) > 500:
                self.servo_default_pwm = int(float(min_val))
            else:
                self.servo_default_pwm = FALLBACK_DEFAULT_PWM

            self.servo_current_pwm = self.servo_default_pwm
            print(
                f"[SERVO 7] Parameter Terbaca:\n"
                f"  -> SERVO7_FUNCTION: {func_val}\n"
                f"  -> SERVO7_TRIM    : {trim_val}\n"
                f"  -> SERVO7_MIN     : {min_val}\n"
                f"  => Default Rest PWM Disimpan: {self.servo_default_pwm} PWM"
            )
        except Exception as e:
            print(f"[SERVO 7 WARNING] Gagal mengambil parameter: {e}. Menggunakan default: {FALLBACK_DEFAULT_PWM} PWM")
            self.servo_default_pwm = FALLBACK_DEFAULT_PWM
            self.servo_current_pwm = self.servo_default_pwm

    def load_yolo_model(self):
        """Memuat model ONNX YOLO"""
        model_path = Path(YOLO_MODEL_PATH)
        if not model_path.is_file():
            # Coba cari di direktori script
            script_dir = Path(__file__).resolve().parent
            model_path = script_dir / YOLO_MODEL_PATH

        print(f"\n[YOLO] Memuat model: {model_path}...")
        try:
            self.yolo_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            self.yolo_input_name = self.yolo_session.get_inputs()[0].name
            meta = self.yolo_session.get_modelmeta().custom_metadata_map
            if "names" in meta:
                import ast
                self.yolo_names = ast.literal_eval(meta["names"])
            else:
                self.yolo_names = {0: "triangle_red", 1: "hexagon_blue", 2: "square_red", 3: "square_blue"}
            print(f"[YOLO] Model berhasil dimuat. Daftar Kelas: {self.yolo_names}")
        except Exception as e:
            print(f"[YOLO ERROR] Gagal memuat model YOLO: {e}")
            sys.exit(1)

    def start_usb_camera(self):
        """Membuka camera capture dari USB Video Capture Card"""
        print(f"\n[CAMERA] Membuka USB Video Capture pada index {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"[CAMERA ERROR] Tidak dapat membuka kamera pada index {self.camera_index}")
            return False

        def usb_camera_loop():
            print(f"[CAMERA] Foxeer USB video stream aktif (index {self.camera_index}).")
            while True:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.latest_img = frame.copy()
                    now = time.time()
                    if self.debug_last_frame_time is not None:
                        dt = now - self.debug_last_frame_time
                        if dt > 0:
                            fps = 1.0 / dt
                            self.debug_fps = (0.85 * self.debug_fps) + (0.15 * fps) if self.debug_fps > 0 else fps
                    self.debug_last_frame_time = now
                    self.new_frame = True
                else:
                    time.sleep(0.01)

        t = threading.Thread(target=usb_camera_loop, daemon=True, name="CameraThread")
        t.start()
        return True

    def init_simulation_camera(self):
        """Kamera callback untuk mode simulasi Gazebo / WebCam fallback"""
        if HAVE_GZ:
            try:
                self.node = Node()
                self.node.subscribe(Image, "/camera/image_raw", self.gazebo_cam_cb)
                print("[CAMERA] Berlangganan topik Gazebo: /camera/image_raw")
            except Exception as e:
                print(f"[CAMERA] Gazebo transport tidak dapat diinisialisasi: {e}")
        else:
            self.start_usb_camera()

    def gazebo_cam_cb(self, msg):
        try:
            img_arr = np.frombuffer(msg.data, dtype=np.uint8)
            img_decoded = img_arr.reshape((msg.height, msg.width, 3))
            self.latest_img = cv2.cvtColor(img_decoded, cv2.COLOR_RGB2BGR).copy()
            now = time.time()
            if self.debug_last_frame_time is not None:
                dt = now - self.debug_last_frame_time
                if dt > 0:
                    fps = 1.0 / dt
                    self.debug_fps = (0.85 * self.debug_fps) + (0.15 * fps) if self.debug_fps > 0 else fps
            self.debug_last_frame_time = now
            self.new_frame = True
        except Exception as e:
            print(f"[CAMERA] Gagal decode frame Gazebo: {e}")

    # ==========================================================================
    # Servo Command Functions
    # ==========================================================================
    def set_servo_pwm(self, pwm_value, reason=""):
        """Mengirim MAVLink DO_SET_SERVO command ke Flight Controller"""
        self.servo_current_pwm = pwm_value
        print("\n" + "=" * 60)
        print(f" [SERVO ACTION] MENGIRIM SERVO CHANNEL {self.servo_channel} -> {pwm_value} PWM")
        if reason:
            print(f" [SERVO ACTION] Alasan: {reason}")
        print("=" * 60)

        # Kirim MAVLink Command Long DO_SET_SERVO
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0,
            self.servo_channel,
            pwm_value,
            0, 0, 0, 0, 0
        )

    def trigger_drop(self, det_info=None):
        """Trigger Servo 7 ke 2200 PWM saat target biru kotak terdeteksi"""
        if not self.servo_triggered:
            self.servo_triggered = True
            self.trigger_count += 1
            self.last_trigger_time = time.time()
            conf_str = f"(Conf: {det_info['conf']:.2f})" if det_info else ""
            self.set_servo_pwm(TRIGGER_DROP_PWM, reason=f"TARGET BLUE SQUARE (BIRU KOTAK) TERDETEKSI! {conf_str}")

    def reset_servo(self):
        """Reset Servo 7 kembali ke default resting PWM"""
        self.servo_triggered = False
        self.set_servo_pwm(self.servo_default_pwm, reason="MANUAL RESET KE DEFAULT PWM")

    # ==========================================================================
    # Telemetry & Vision Processing
    # ==========================================================================
    def update_telemetry(self):
        """Membaca pesan telemetri MAVLink dari Flight Controller"""
        # Attitude
        msg_att = self.master.recv_match(type='ATTITUDE', blocking=False)
        if msg_att:
            self.roll = msg_att.roll
            self.pitch = msg_att.pitch
            self.yaw = msg_att.yaw

        # Global Position / Altitude
        msg_pos = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if msg_pos:
            self.current_alt = msg_pos.relative_alt / 1000.0

        # Servo Output Raw (untuk memantau live servo output)
        msg_servo = self.master.recv_match(type='SERVO_OUTPUT_RAW', blocking=False)
        if msg_servo:
            raw_pwm = getattr(msg_servo, f'servo{self.servo_channel}_raw', None)
            if raw_pwm is not None and raw_pwm > 0:
                self.servo_current_pwm = raw_pwm

    def check_color_guard_in_frame(self, frame):
        """
        Pre-detection color guard: Memeriksa apakah frame memiliki warna biru atau merah
        menggunakan rentang HSV lebar sebelum menjalankan inferensi model.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, COLOR_GUARD_BLUE_LOWER, COLOR_GUARD_BLUE_UPPER)
        mask_red1 = cv2.inRange(hsv, COLOR_GUARD_RED_LOWER1, COLOR_GUARD_RED_UPPER1)
        mask_red2 = cv2.inRange(hsv, COLOR_GUARD_RED_LOWER2, COLOR_GUARD_RED_UPPER2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        return hsv, mask_blue, mask_red

    def check_roi_color_guard(self, hsv, box, color_type="blue"):
        """
        Guard deteksi: Memvalidasi ROI bounding box apakah benar-benar mengandung warna yang sesuai
        (square_blue -> warna biru, square_red -> warna merah) menggunakan rentang HSV OpenCV.
        """
        x1, y1, x2, y2 = box
        roi_hsv = hsv[y1:y2 + 1, x1:x2 + 1]
        if roi_hsv.size == 0:
            return False

        if color_type == "blue":
            mask = cv2.inRange(roi_hsv, COLOR_GUARD_BLUE_LOWER, COLOR_GUARD_BLUE_UPPER)
        elif color_type == "red":
            m1 = cv2.inRange(roi_hsv, COLOR_GUARD_RED_LOWER1, COLOR_GUARD_RED_UPPER1)
            m2 = cv2.inRange(roi_hsv, COLOR_GUARD_RED_LOWER2, COLOR_GUARD_RED_UPPER2)
            mask = cv2.bitwise_or(m1, m2)
        else:
            return True

        return cv2.countNonZero(mask) >= MIN_ROI_COLOR_PIXELS

    def process_vision(self):
        """Menjalankan inferensi YOLO pada frame terbaru dengan Guard Warna OpenCV"""
        if self.latest_img is None:
            return

        frame = self.latest_img.copy()

        # Guard Pertama: Deteksi Warna OpenCV di seluruh frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if COLOR_GUARD_ENABLED:
            mask_blue = cv2.inRange(hsv, COLOR_GUARD_BLUE_LOWER, COLOR_GUARD_BLUE_UPPER)
            mask_red1 = cv2.inRange(hsv, COLOR_GUARD_RED_LOWER1, COLOR_GUARD_RED_UPPER1)
            mask_red2 = cv2.inRange(hsv, COLOR_GUARD_RED_LOWER2, COLOR_GUARD_RED_UPPER2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            blue_px = cv2.countNonZero(mask_blue)
            red_px = cv2.countNonZero(mask_red)
            # Jika tidak ada pixel warna target sama sekali di frame, skip inferensi model untuk hemat CPU & cegah false positive
            if blue_px < MIN_COLOR_GUARD_PIXELS and red_px < MIN_COLOR_GUARD_PIXELS:
                self.consecutive_blue_frames = 0
                self.blue_detected = False
                self.render_hud(frame)
                return

        tensor, scale, pad_x, pad_y = preprocess_yolo(frame)
        raw = self.yolo_session.run(None, {self.yolo_input_name: tensor})[0][0]

        frame_h, frame_w = frame.shape[:2]
        detected_blue = None

        for row in raw:
            conf = float(row[4])
            if conf < YOLO_CONF_THRESHOLD:
                continue

            class_id = int(row[5])
            label = self.yolo_names.get(class_id, str(class_id))

            x1, y1, x2, y2 = [float(v) for v in row[:4]]
            x1 = max(0, min(frame_w - 1, int(round((x1 - pad_x) / scale))))
            x2 = max(0, min(frame_w - 1, int(round((x2 - pad_x) / scale))))
            y1 = max(0, min(frame_h - 1, int(round((y1 - pad_y) / scale))))
            y2 = max(0, min(frame_h - 1, int(round((y2 - pad_y) / scale))))

            if x2 <= x1 or y2 <= y1:
                continue

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            is_target_blue = (label == TARGET_CLASS_NAME or label == "square_blue")
            is_target_red = (label == "square_red")

            # Guard Warna Bounding Box (ROI Color Guard):
            # Memastikan ROI square_blue benar-benar berwarna biru, atau square_red berwarna merah
            if COLOR_GUARD_ENABLED:
                if is_target_blue:
                    if not self.check_roi_color_guard(hsv, (x1, y1, x2, y2), "blue"):
                        continue
                elif is_target_red:
                    if not self.check_roi_color_guard(hsv, (x1, y1, x2, y2), "red"):
                        continue

            # Gambar Bounding Box
            color = (255, 120, 0) if is_target_blue else ((0, 80, 255) if is_target_red else (0, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, center, 4, color, -1)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                1,
                cv2.LINE_AA
            )

            if is_target_blue:
                if detected_blue is None or conf > detected_blue["conf"]:
                    detected_blue = {
                        "label": label,
                        "conf": conf,
                        "box": (x1, y1, x2, y2),
                        "center": center
                    }

        # Logika Trigger Servo saat Target Biru Kotak Terdeteksi
        if detected_blue is not None:
            self.consecutive_blue_frames += 1
            self.last_detection_info = detected_blue
            self.blue_detected = True

            # Trigger jika terdeteksi stabil (>= 1 frame dengan confidence valid)
            if not self.servo_triggered:
                self.trigger_drop(detected_blue)
        else:
            self.consecutive_blue_frames = 0
            self.blue_detected = False

        # Tampilkan HUD
        self.render_hud(frame)

    # ==========================================================================
    # HUD Rendering & Visual Interface
    # ==========================================================================
    def render_hud(self, frame):
        if not self.gui_enabled:
            return

        status_text = "TRIGGERED (2200 PWM)" if self.servo_triggered else "IDLE (WAITING FOR BLUE SQUARE)"
        status_color = (0, 255, 0) if self.servo_triggered else (0, 255, 255)

        lines = [
            (f"MODE: {self.mode.upper()} | SERVO CH{self.servo_channel} TEST & VISION", (0, 255, 255)),
            (f"Debug FPS: {self.debug_fps:.1f} | Alt: {self.current_alt:.1f}m | Roll: {math.degrees(self.roll):.1f}deg", (200, 200, 200)),
            (f"Servo 7 Default PWM: {self.servo_default_pwm} | Trigger Target: {TRIGGER_DROP_PWM}", (255, 255, 255)),
            (f"Servo 7 Current PWM: {self.servo_current_pwm} | Triggers: {self.trigger_count}", (0, 255, 100) if self.servo_triggered else (255, 255, 255)),
            (f"Target: {TARGET_CLASS_NAME.upper()} | Conf Gate: {int(YOLO_CONF_THRESHOLD*100)}% | Color Guard: {'ON' if COLOR_GUARD_ENABLED else 'OFF'}", (255, 200, 0) if self.blue_detected else (200, 200, 200)),
            (f"Shortcut: [T] Test 2200 PWM | [R] Reset Default | [Q] Quit", (180, 220, 255)),
        ]

        # Top Dark Card Panel
        overlay = frame.copy()
        box_w = min(520, IMG_W - 16)
        box_h = len(lines) * 20 + 14
        cv2.rectangle(overlay, (8, 8), (box_w, box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (8, 8), (box_w, box_h), (60, 60, 60), 1)

        y = 26
        for text, color in lines:
            cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
            y += 20

        # Bottom Status Banner
        overlay_bot = frame.copy()
        cv2.rectangle(overlay_bot, (8, IMG_H - 36), (IMG_W - 8, IMG_H - 8), (20, 20, 20), -1)
        cv2.addWeighted(overlay_bot, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (8, IMG_H - 36), (IMG_W - 8, IMG_H - 8), (60, 60, 60), 1)

        banner_msg = f"STATUS: {status_text}"
        if self.blue_detected and self.last_detection_info:
            banner_msg += f" | CONF: {self.last_detection_info['conf']:.2f}"
        cv2.putText(frame, banner_msg, (14, IMG_H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 1, cv2.LINE_AA)

        # Tampilkan ke Jendela OpenCV
        try:
            cv2.imshow(DEBUG_WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n[USER] Tombol Q ditekan, menghentikan program.")
                self.gui_enabled = False
                cv2.destroyAllWindows()
                sys.exit(0)
            elif key == ord('t') or key == ord('T'):
                print("\n[USER] Tombol T ditekan -> Manual Trigger Servo 7 ke 2200 PWM!")
                self.trigger_drop()
            elif key == ord('r') or key == ord('R'):
                print("\n[USER] Tombol R ditekan -> Reset Servo 7 ke default PWM!")
                self.reset_servo()
        except cv2.error as e:
            print(f"[DEBUG] OpenCV GUI tidak tersedia (running headless): {e}")
            self.gui_enabled = False

    # ==========================================================================
    # Main Execution Loop
    # ==========================================================================
    def run(self):
        print("\n" + "=" * 65)
        print(" [SYSTEM READY] MONITORING VISION & SERVO 7 CONTROL RUNNING")
        print(" - Buka Mission Planner kapan saja -> Connect UDP 14550")
        print(f" - Menunggu deteksi target '{TARGET_CLASS_NAME}' untuk trigger Servo 7 -> 2200 PWM")
        print(" - Tekan [T] untuk test trigger, [R] untuk reset, [Q] untuk keluar.")
        print("=" * 65 + "\n")

        try:
            while True:
                self.update_telemetry()

                if self.new_frame:
                    self.process_vision()
                    self.new_frame = False

                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n[EXIT] Program dihentikan oleh pengguna (Ctrl+C).")
        finally:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()


# ==============================================================================
# Entry Point
# ==============================================================================
if __name__ == "__main__":
    mode = select_operating_mode()

    if mode == "real":
        com_port, baud_rate, cam_index = configure_real_hardware()
        success, connection_uri = start_mavlink_forwarder(com_port, baud_rate)
        if not success:
            print("[ERROR] Gagal memulai MAVLink forwarder. Keluar.")
            sys.exit(1)
        runner = ServoTestRunner(mode="real", connection_string=connection_uri, baud_rate=baud_rate, camera_index=cam_index)
    else:
        runner = ServoTestRunner(mode="simulation", connection_string=DEFAULT_SIM_CONNECTION_STRING)

    runner.run()
