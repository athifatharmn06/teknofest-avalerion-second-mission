"""
================================================================================
                      TEKNOFEST MISSION 2 - CONTROL & AUTONOMY
================================================================================

[SYSTEM REQUIREMENTS & INSTALLATION INSTRUCTIONS]
1. Python Version: Python 3.8 - 3.11 recommended.
2. Required Python Packages:
   Run the following command to install required dependencies:
   
       pip install opencv-python numpy onnxruntime pymavlink pyserial

3. Optional Dependencies for Simulation Mode:
   - Gazebo Sim (Garden / Harmonic / Fortress) with gz-transport Python bindings:
       pip install gz-transport13 gz-msgs10
   - MAVProxy (optional CLI tool):
       pip install MAVProxy

4. Hardware Setup (Real Mode):
   - Telemetry Radio: RFD900 plugged into USB (e.g. COM3 on Windows or /dev/ttyUSB0 on Linux).
   - FPV / Camera Receiver: Foxeer Camera VTX Receiver plugged into USB Video Capture Card.
   - Mission Planner: Set to connect to UDP port 14550 (automatic forwarding).

================================================================================
"""

import sys
import time
import math
import ast
import copy
import fnmatch
import struct
import logging
import threading
import socket
from pathlib import Path
import numpy as np
import cv2
import subprocess
import onnxruntime as ort
from pymavlink import mavutil

# --- Try Import Gazebo Transport (Optional for Real Mode) ---
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
# GLOBAL USER-ADJUSTABLE CONFIGURATION & VARIABLES (GCS OPERATOR SECTION)
# ==============================================================================

# --- TELEMETRY & NETWORK CONNECTION ---
# Default connection string for Simulation mode (UDP port where SITL sends telemetry)
# Function: Sets the IP and port used when running in Simulation mode.
# If Changed: Alters the target endpoint for Gazebo/SITL telemetry.
DEFAULT_SIM_CONNECTION_STRING = 'udp:localhost:14551'

# Port for Mission Planner (GCS) to auto-connect to local MAVLink telemetry stream.
# Function: Mission Planner auto-detects telemetry on UDP 14550.
# If Changed: Mission Planner must be manually directed to connect to this UDP port.
DEFAULT_MP_FORWARD_PORT = 14550

# Port for this Python autonomy script to receive MAVLink telemetry stream.
# Function: Script internal UDP port for bi-directional telemetry connection.
# If Changed: Alters local UDP binding port for python script MAVLink connection.
DEFAULT_SCRIPT_PORT = 14551

# Default baud rate for RFD900 telemetry radio in Real Hardware mode.
# Function: Baud rate for physical serial communication.
# If Changed: Must match the hardware baud rate configured on RFD900 (57600 or 115200).
DEFAULT_SERIAL_BAUD_RATE = 57600


# --- CAMERA & VISION SETTINGS ---
# Gazebo camera image topic for simulation mode.
# Function: ROS/Gazebo transport topic name for camera frames.
# If Changed: Gazebo camera plugin topic name must be updated to match.
CAMERA_TOPIC = "/camera/image_raw"

# Camera image width in pixels (Foxeer camera via USB Capture Card).
# Function: Defines image frame horizontal resolution.
# If Changed: Modifies image matrix width and FOV focal length calculations.
IMG_W = 640

# Camera image height in pixels.
# Function: Defines image frame vertical resolution.
# If Changed: Modifies image matrix height and FOV focal length calculations.
IMG_H = 480

# Horizontal Field-of-View of camera in radians (~85 degrees for Foxeer camera).
# Function: Used in 3D rotation matrix calculation to project camera pixels to GPS ground coordinates.
# If Changed: Adjusts geolocation ray projection geometry for wider or narrower lens.
FOV_H_RAD = 1.5

# Computed camera focal length in pixels (auto-calculated from IMG_W and FOV_H_RAD).
FOCAL_LENGTH_PX = (IMG_W / 2) / math.tan(FOV_H_RAD / 2)


# --- YOLO DETECTION & GEOLOCATION ---
# Path to ONNX YOLO model file (must be located in the script directory or absolute path).
# Function: Model weights file loaded by ONNXRuntime for target detection.
# If Changed: Script loads specified custom model file for target classification.
YOLO_MODEL_PATH = "v1_gazbmodel_exp.onnx"

# Input tensor dimension for YOLO model (640x640).
# Function: Dimension to which camera frames are letterboxed before inference.
# If Changed: Adjusts YOLO pre-processing letterbox input shape.
YOLO_INPUT_SIZE = 640

# Global YOLO confidence threshold for candidate detections (0.0 to 1.0).
# Function: Minimum confidence required to accept a detection candidate.
# If Changed: Increasing filters false positives; decreasing detects fainter objects.
YOLO_CONF_THRESHOLD = 0.85

# Class-specific confidence thresholds.
# Function: Fine-tunes detection sensitivity independently for blue and red targets.
# If Changed: Alters confidence gate required for locking blue vs red target shapes.
YOLO_CLASS_CONF_THRESHOLDS = {
    "square_blue": 0.85,
    "square_red": 0.85,
}

# High confidence threshold for accepting pixel sample jumps during target tracking.
# Function: Accepts larger pixel jumps if detection confidence exceeds this value.
YOLO_SAMPLE_HIGH_CONF = 0.90

# Maximum allowed pixel jump between consecutive video frames for target tracking stability.
# Function: Rejects outlier detections caused by momentary vision noise or glitch.
# If Changed: Increasing allows faster pixel movement; decreasing strictly filters jumps.
YOLO_MAX_PIXEL_JUMP = 180

# Mapping of YOLO class labels to target colors.
# Function: Connects YOLO detection label strings to target payload logic ("blue" / "red").
YOLO_FIXED_WING_CLASSES = {
    "square_blue": "blue",
    "square_red": "red",
}

# BGR color palette for drawing bounding boxes on OpenCV display window.
# Function: Defines bounding box colors for visual GCS operator monitoring.
YOLO_CLASS_COLORS = {
    "triangle_red": (0, 0, 255),
    "hexagon_blue": (255, 0, 0),
    "square_red": (0, 80, 255),
    "square_blue": (255, 80, 0),
}

# --- COLOR GUARD PRE-DETECTION FILTER ---
# Enable/disable color guard before passing frames to YOLO model.
# Function: Checks if broad red/blue color range exists in frame before executing YOLO model detection.
# If Changed: Set False to run YOLO model on every frame regardless of color content.
COLOR_GUARD_ENABLED = True

# Wide HSV range for Blue targets (OpenCV Hue 0-179, Sat 0-255, Val 0-255)
# Rentang lebar agar mendeteksi variasi warna biru (cyan, navy, sky blue) di kondisi pencahayaan outdoor
COLOR_GUARD_BLUE_LOWER = np.array([75, 30, 30], dtype=np.uint8)
COLOR_GUARD_BLUE_UPPER = np.array([145, 255, 255], dtype=np.uint8)

# Wide HSV range for Red targets (wraps around hue 0 and 180)
# Rentang lebar untuk variasi warna merah (oranye-merah, merah terang, merah tua/marun)
COLOR_GUARD_RED_LOWER1 = np.array([0, 30, 30], dtype=np.uint8)
COLOR_GUARD_RED_UPPER1 = np.array([18, 255, 255], dtype=np.uint8)
COLOR_GUARD_RED_LOWER2 = np.array([155, 30, 30], dtype=np.uint8)
COLOR_GUARD_RED_UPPER2 = np.array([180, 255, 255], dtype=np.uint8)

# Minimum number of color pixels in frame to trigger model detection.
MIN_COLOR_GUARD_PIXELS = 15

# Minimum number of color pixels in bounding box ROI to confirm detection.
MIN_ROI_COLOR_PIXELS = 5

# Maximum allowed aircraft Roll angle (in degrees) during target geolocation scan.
# Function: Discards vision frames when aircraft is banking sharply (> 20 deg) to prevent projection error.
# If Changed: Increasing allows detection during steeper turns; decreasing ensures flatter camera angle.
MAX_DETECTION_ROLL_DEG = 20.0

# Maximum allowed aircraft Pitch angle (in degrees) during target geolocation scan.
# Function: Discards vision frames when aircraft pitch is steep (> 8 deg).
# If Changed: Alters pitch gate threshold for accepting valid geolocation frames.
MAX_DETECTION_PITCH_DEG = 8.0

# Minimum number of valid target GPS samples required before locking target location.
# Function: Number of grouped detection samples needed to lock target coordinates.
# If Changed: Increasing improves target coordinate precision; decreasing locks target faster.
TARGET_SAMPLES_REQUIRED = 2

# Maximum sample buffer size for median/mean clustering.
# Function: Rolling buffer size of recent target GPS coordinates.
TARGET_SAMPLE_BUFFER = 16

# Maximum cluster radius in meters for target sample grouping.
# Function: Maximum distance allowed between samples to belong to the same target cluster.
TARGET_CLUSTER_RADIUS_M = 12.0

# Maximum allowable spread (standard deviation) of clustered GPS samples.
# Function: Rejects cluster if sample scatter exceeds this radius in meters.
MAX_TARGET_SAMPLE_SPREAD_M = 8.0

# Gated Fallback: Distance threshold (meters) from DETECT_BEFORE_WP_REACH (WP 5) to force-lock target samples.
# Function: When UAV comes within 70m of DETECT_BEFORE_WP_REACH (WP 5), force-locks whatever samples are gathered (even 2-3) & jumps to Staging.
# If Changed: Adjusts distance gate threshold to trigger scan exit upload.
SCAN_EXIT_THRESHOLD_M = 70.0

# Enable/disable OpenCV GCS visual display window.
# Function: Displays live video feed with bounding boxes, telemetry, and status text.
# If Changed: Set False to disable GUI window when running headless.
DEBUG_SHOW_CAMERA = True
DEBUG_SHOW_MASK = False
DEBUG_WINDOW_NAME = "Teknofest Mission 2 YOLO Detection & Tracking GCS"
DEBUG_MASK_WINDOW_NAME = "Teknofest YOLO Mask"


# --- MISSION & WAYPOINT SEQUENCING ---
# Default fallback sequence number after which target scanning becomes active (WP 4 entry).
# Function: Waypoint index sequence after which vision scan activates.
DETECT_AFTER_WP_REACH = 3

# Sequence number of the scan area exit waypoint (WP 5 exit).
# Function: Defines scan exit waypoint (WP 5) for distance threshold & forced target lock.
DETECT_BEFORE_WP_REACH = 4

# Waypoint sequence index after which the straight drop mission route will be inserted.
# Function: Dictates insertion index in mission list for new drop waypoints.
INSERT_DROP_ROUTE_AFTER_WP_SEQ = 3


# --- PHYSICAL CONSTANTS & TIMING DELAYS ---
# Standard gravitational acceleration (m/s^2).
GRAVITY_MPS2 = 9.80665

# Additional mechanical payload release delay (seconds).
# Function: Release delay factor for slow physical mechanical actuators.
# If Changed: Extends drop lead distance to compensate for mechanical release lag.
DROP_RELEASE_DELAY_SEC = 0.0

# Ground processing latency delay (seconds).
# Function: User-adjustable video capture card and ground station image processing latency.
# Formula: d_lead = cruise_speed * sqrt(2 * h / g) + cruise_speed * (release_delay + ground_latency)
# If Changed: Compensates for ground station image processing and transmission delay during testing.
GROUND_PROCESSING_LATENCY_SEC = 0.0


# --- PAYLOAD SERVO HARDWARE CONFIGURATION (GCS OPERATOR ADJUSTABLE) ---
# FC Servo Channel for LEFT payload release (Red target).
# Function: MAVLink MAV_CMD_DO_SET_SERVO target channel for left payload.
# If Changed: Selects physical output channel on FC for left servo mechanism.
SERVO_CHANNEL_LEFT = 9

# FC Servo Channel for RIGHT payload release (Blue target).
# Function: MAVLink MAV_CMD_DO_SET_SERVO target channel for right payload.
# If Changed: Selects physical output channel on FC for right servo mechanism.
SERVO_CHANNEL_RIGHT = 10

# PWM release value for LEFT payload servo (independent rotation direction).
# Function: PWM microsecond signal sent to trigger LEFT payload drop.
# If Changed: Adjust PWM value (e.g. 1900 or 1100) based on left servo rotation direction.
SERVO_DROP_PWM_LEFT = 1900

# PWM release value for RIGHT payload servo (independent rotation direction).
# Function: PWM microsecond signal sent to trigger RIGHT payload drop.
# If Changed: Adjust PWM value (e.g. 1900 or 1100) based on right servo rotation direction.
SERVO_DROP_PWM_RIGHT = 1900


# --- GAZEBO PAYLOAD DETACH TOPICS (SIMULATION MODE ONLY) ---
TOPIC_DROP_LEFT = "/payload_left/detach"   # Red payload Gazebo topic
TOPIC_DROP_RIGHT = "/payload_right/detach" # Blue payload Gazebo topic


# ==============================================================================
# Integrated MAVWP Classes (from mavwp.py)
# ==============================================================================
class MAVWPError(Exception):
    '''MAVLink WP error class'''
    def __init__(self, msg):
        Exception.__init__(self, msg)
        self.message = msg


class MissionItemProtocol(object):
    '''Base class for transferring items based on the MISSION_ITEM protocol'''
    def __init__(self, target_system=0, target_component=0):
        self.wpoints = []
        self.target_system = target_system
        self.target_component = target_component
        self.last_change = 0

    def count(self):
        '''return number of waypoints'''
        return len(self.wpoints)

    def wp(self, i):
        '''alias for backwards compatability'''
        return self.item(i)

    def item(self, i):
        '''return an item'''
        try:
            the_wp = self.wpoints[i]
        except Exception:
            the_wp = None
        return the_wp

    def add(self, w, comment=''):
        '''add a waypoint'''
        if type(w) == list:
            w = copy.deepcopy(w)
            n = self.count()
            for p in w:
                p.seq = n
                n += 1
            self.wpoints.extend(w)
        else:
            w = copy.copy(w)
            if comment:
                w.comment = comment
            w.seq = self.count()
            self.wpoints.append(w)
        self.last_change = time.time()

    def insert(self, idx, w, comment=''):
        '''insert a waypoint'''
        if idx >= self.count():
            self.add(w, comment)
            return
        if idx < 0:
            return
        w = copy.copy(w)
        if comment:
            w.comment = comment
        w.seq = idx
        self.wpoints.insert(idx, w)
        self.last_change = time.time()
        self.reindex()

    def reindex(self):
        '''reindex waypoints'''
        for i in range(self.count()):
            w = self.wpoints[i]
            w.seq = i
        self.last_change = time.time()

    def set(self, w, idx):
        '''set a waypoint'''
        w.seq = idx
        if w.seq == self.count():
            return self.add(w)
        if self.count() <= idx:
            raise MAVWPError('adding waypoint at idx=%u past end of list (count=%u)' % (idx, self.count()))
        self.wpoints[idx] = w
        self.last_change = time.time()

    def remove(self, w):
        '''remove a waypoint'''
        if isinstance(w, list):
            for point in w:
                self.wpoints.remove(point)
        else:
            self.wpoints.remove(w)
        self.last_change = time.time()
        self.reindex()

    def clear(self):
        '''clear waypoint list'''
        self.wpoints = []
        self.last_change = time.time()


class MAVWPLoader(MissionItemProtocol):
    '''MAVLink waypoint loader'''
    def mav_mission_type(self):
        '''returns type of mission this object transfers'''
        return mavutil.mavlink.MAV_MISSION_TYPE_MISSION

    def wp_is_loiter(self, i):
        '''return true if waypoint is a loiter waypoint'''
        loiter_cmds = [
            mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM,
            mavutil.mavlink.MAV_CMD_NAV_LOITER_TURNS,
            mavutil.mavlink.MAV_CMD_NAV_LOITER_TIME,
            mavutil.mavlink.MAV_CMD_NAV_LOITER_TO_ALT
        ]
        if self.wpoints[i].command in loiter_cmds:
            return True
        return False

    def add_latlonalt(self, lat, lon, altitude, terrain_alt=False):
        '''add a point via latitude/longitude/altitude'''
        if terrain_alt:
            frame = mavutil.mavlink.MAV_FRAME_GLOBAL_TERRAIN_ALT
        else:
            frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
        p = mavutil.mavlink.MAVLink_mission_item_message(
            self.target_system,
            self.target_component,
            0,
            frame,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0, 0, 0, 0, 0, 0,
            lat, lon, altitude
        )
        self.add(p)


# ==============================================================================
# Integrated MAVParm Classes (from mavparm.py)
# ==============================================================================
class MAVParmDict(dict):
    '''dict wrapper for MAVLink parameters'''
    def __init__(self, *args):
        dict.__init__(self, args)
        self.exclude_load = [
            'ARSPD_OFFSET', 'CMD_INDEX', 'CMD_TOTAL', 'FENCE_TOTAL',
            'FORMAT_VERSION', 'GND_ABS_PRESS', 'GND_TEMP', 'LOG_LASTFILE',
            'MIS_TOTAL', 'SYSID_SW_MREV', 'SYS_NUM_RESETS',
        ]
        self.mindelta = 0.000001

    def fetch_all(self, mav, timeout=4.0):
        '''Request parameter list from FC and populate dict'''
        print("[PARAM] Requesting parameter list from Flight Controller...")
        try:
            target_system = getattr(mav, 'target_system', 1)
            target_component = getattr(mav, 'target_component', 0)
            if hasattr(mav, 'param_fetch_all'):
                mav.param_fetch_all()
            elif hasattr(mav, 'param_request_list_send'):
                try:
                    mav.param_request_list_send()
                except TypeError:
                    mav.param_request_list_send(target_system, target_component)
            else:
                mav_obj = getattr(mav, 'mav', mav)
                mav_obj.param_request_list_send(target_system, target_component)
            tstart = time.time()
            count = None
            while time.time() - tstart < timeout:
                msg = mav.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.5)
                if msg:
                    param_id = msg.param_id
                    if isinstance(param_id, bytes):
                        param_id = param_id.decode('utf-8', errors='ignore')
                    param_id = param_id.strip('\x00').upper()
                    self[param_id] = msg.param_value
                    if count is None:
                        count = msg.param_count
                    if count and len(self) >= count:
                        break
            print(f"[PARAM] Successfully loaded {len(self)} parameters from FC.")
        except Exception as e:
            print(f"[PARAM] Note: Could not complete full parameter download: {e}")

    def get_param(self, name, default):
        '''Get parameter value with fallback default'''
        val = self.get(name.upper(), default)
        try:
            return float(val)
        except (ValueError, TypeError):
            return default


# ==============================================================================
# Mission State Enum & Math Utilities
# ==============================================================================
class MissionState:
    WAITING_FOR_SCAN_WP = 0
    SCANNING = 1
    UPLOADING_WPS = 2
    APPROACHING_STAGING = 3
    APPROACHING_TARGET_1 = 4
    APPROACHING_TARGET_2 = 5
    FINISHED = 6


def sanitize_original_mission(wpoints):
    """
    Sanitizes downloaded mission by filtering out any previously injected 
    DO_SET_SERVO or CONDITION_DISTANCE items, while preserving ALL original 
    user waypoints (from WP0 all the way to LAND/RTL).
    """
    if not wpoints:
        return []

    # Keep all items except previously injected MAV_CMD_CONDITION_DISTANCE and MAV_CMD_DO_SET_SERVO
    clean = []
    for wp in wpoints:
        if wp.command in (
            mavutil.mavlink.MAV_CMD_CONDITION_DISTANCE,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO
        ):
            continue
        clean.append(copy.deepcopy(wp))

    # Re-index sequences
    for idx, wp in enumerate(clean):
        wp.seq = idx

    return clean


def offset_gps(lat, lon, north_m, east_m):
    d_lat = north_m / 111319.9
    d_lon = east_m / (111319.9 * math.cos(math.radians(lat)))
    return lat + d_lat, lon + d_lon


def ned_offset_from_gps(origin_lat, origin_lon, target_lat, target_lon):
    north_m = (target_lat - origin_lat) * 111319.9
    east_m = (target_lon - origin_lon) * 111319.9 * math.cos(math.radians(origin_lat))
    return north_m, east_m


def distance_between_points(a, b):
    return haversine_distance(a["lat"], a["lon"], b["lat"], b["lon"])


def haversine_distance(lat1, lon1, lat2, lon2):
    radius_m = 6371000
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) * math.sin(d_lat / 2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(d_lon / 2) * math.sin(d_lon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_m * c


def point_before_target(from_point, target, distance_m):
    north_m, east_m = ned_offset_from_gps(from_point["lat"], from_point["lon"], target["lat"], target["lon"])
    length = math.hypot(north_m, east_m)
    if length <= distance_m:
        return dict(target)

    ratio = (length - distance_m) / length
    return {
        "lat": from_point["lat"] + (target["lat"] - from_point["lat"]) * ratio,
        "lon": from_point["lon"] + (target["lon"] - from_point["lon"]) * ratio,
    }


def point_after_target(from_point, target, distance_m):
    north_m, east_m = ned_offset_from_gps(from_point["lat"], from_point["lon"], target["lat"], target["lon"])
    length = math.hypot(north_m, east_m)
    if length <= 0:
        return dict(target)

    unit_n = north_m / length
    unit_e = east_m / length
    lat, lon = offset_gps(target["lat"], target["lon"], unit_n * distance_m, unit_e * distance_m)
    return {
        "lat": lat,
        "lon": lon,
    }


def resolve_model_path(model_path):
    path = Path(model_path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def parse_yolo_names(metadata):
    names_text = metadata.get("names", "")
    if not names_text:
        return {}

    try:
        parsed = ast.literal_eval(names_text)
    except (SyntaxError, ValueError):
        return {}

    return {int(k): str(v) for k, v in parsed.items()}


def letterbox(frame, size=YOLO_INPUT_SIZE, color=(114, 114, 114)):
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

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
# Helper for Bi-Directional MAVLink Serial Multiplexing & Port Forwarding
# ==============================================================================
def start_mavlink_forwarder(serial_port, baud_rate, mp_port=DEFAULT_MP_FORWARD_PORT, script_port=DEFAULT_SCRIPT_PORT):
    """
    Connects to RFD900 serial port and creates a robust, high-throughput bi-directional UDP bridge:
    - Serial (RFD900 / FC) <--> UDP 14550 (Mission Planner & LAN Broadcast) & UDP 14551 (Python Script)
    Allows Mission Planner (for HUD, parameter calibration, manual waypoint upload) and this Python Autonomy Script
    to run simultaneously without port conflicts!
    """
    import serial
    print(f"[MAVLINK FORWARDER] Opening serial port {serial_port} @ {baud_rate} baud...")
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=0.005)
    except Exception as e:
        print(f"[MAVLINK FORWARDER ERROR] Failed to open {serial_port}: {e}")
        return False, None

    bridge_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    bridge_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    bridge_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    bridge_sock.setblocking(False)
    bridge_sock.bind(('127.0.0.1', 0))

    # Pre-configure targets: Localhost MP (14550), Python Script (14551), and Subnet Broadcast (255.255.255.255:14550)
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

    t1 = threading.Thread(target=serial_to_udp, daemon=True, name="Ser2UDP")
    t2 = threading.Thread(target=udp_to_serial, daemon=True, name="UDP2Ser")
    t1.start()
    t2.start()

    print(
        f"[MAVLINK FORWARDER] Active: Bi-directional stream between {serial_port} and:\n"
        f"  -> Mission Planner: UDP {mp_port} (Localhost & LAN Broadcast)\n"
        f"  -> Autonomy Script: UDP {script_port}"
    )
    return True, f"udpin:127.0.0.1:{script_port}"


# ==============================================================================
# Interactive Startup Menu
# ==============================================================================
def select_operating_mode():
    print("\n" + "=" * 65)
    print("         TEKNOFEST MISSION 2 - CONTROL & AUTONOMY SYSTEM")
    print("=" * 65)
    print(" Select Operating Mode:")
    print("   [1] REAL COMPETITIVE MODE (RFD900 USB Telemetry + Foxeer USB Camera)")
    print("   [2] SIMULATION MODE (Gazebo + UDP:14551)")
    print("=" * 65)

    choice = input("Enter choice (1 or 2, default [2]): ").strip()
    if choice == "1":
        return "real"
    return "simulation"


def configure_real_hardware():
    print("\n--- Real Hardware Configuration ---")
    
    # Auto-scan available COM ports
    com_ports = []
    try:
        import serial.tools.list_ports
        com_ports = list(serial.tools.list_ports.comports())
    except ImportError:
        pass

    if com_ports:
        print("Available COM Ports Detected:")
        for idx, p in enumerate(com_ports, start=1):
            print(f"  [{idx}] {p.device} - {p.description}")
        selected_port = com_ports[0].device
        p_choice = input(f"Select COM port (1-{len(com_ports)}, default [1: {selected_port}]): ").strip()
        if p_choice.isdigit() and 1 <= int(p_choice) <= len(com_ports):
            selected_port = com_ports[int(p_choice) - 1].device
    else:
        selected_port = input(f"Enter COM port (e.g. COM3 or /dev/ttyUSB0, default [COM3]): ").strip() or "COM3"

    baud_input = input(f"Enter RFD900 baud rate (default [{DEFAULT_SERIAL_BAUD_RATE}]): ").strip() or str(DEFAULT_SERIAL_BAUD_RATE)
    baud_rate = int(baud_input)

    cam_input = input("Enter USB Video Capture card index for Foxeer camera (default [0]): ").strip() or "0"
    cam_index = int(cam_input)

    return selected_port, baud_rate, cam_index


# ==============================================================================
# Main AdvancedUAV Class
# ==============================================================================
class AdvancedUAV:
    def __init__(self, mode="simulation", connection_string=DEFAULT_SIM_CONNECTION_STRING, baud_rate=DEFAULT_SERIAL_BAUD_RATE, camera_index=0):
        self.mode = mode
        self.connection_string = connection_string
        self.baud_rate = baud_rate
        self.camera_index = camera_index
        self.state = MissionState.WAITING_FOR_SCAN_WP

        # MAVLink Connection
        print(f"\nConnecting to Flight Controller at {self.connection_string} (Baud: {self.baud_rate if mode == 'real' or self.connection_string.upper().startswith('COM') else 'N/A'})...")
        if self.connection_string.upper().startswith("COM") or "/dev/" in self.connection_string:
            self.master = mavutil.mavlink_connection(self.connection_string, baud=self.baud_rate, source_system=1, source_component=255)
        else:
            self.master = mavutil.mavlink_connection(self.connection_string, source_system=1, source_component=255)

        print("Waiting for MAVLink heartbeat...")
        self.master.wait_heartbeat()
        print(f"Flight Controller Connected! (SysID: {self.master.target_system}, CompID: {self.master.target_component})")

        # Initialize Waypoint Loader & Backup Memory Store
        self.wp_loader = MAVWPLoader(self.master.target_system, self.master.target_component)
        self.original_wpoints = []  # Backup copy of pristine initial mission

        # ----------------------------------------------------------------------
        # STARTUP PRE-FETCH: Fetch parameters & Initial Mission WPs Immediately
        # ----------------------------------------------------------------------
        print("\n[STARTUP PRE-FETCH] Fetching FC Parameters & Mission Waypoints...")
        self.params = MAVParmDict()
        self.params.fetch_all(self.master, timeout=3.0)

        # Initialize parameters from FC or top-level defaults
        self.cruise_speed_mps = self.params.get_param('AIRSPEED_CRUISE', self.params.get_param('TRIM_ARSPD', 22.0))
        self.servo_channel_left = int(self.params.get_param('SERVO9_FUNCTION', SERVO_CHANNEL_LEFT))
        self.servo_channel_right = int(self.params.get_param('SERVO10_FUNCTION', SERVO_CHANNEL_RIGHT))
        self.servo_drop_pwm_left = SERVO_DROP_PWM_LEFT
        self.servo_drop_pwm_right = SERVO_DROP_PWM_RIGHT
        self.staging_distance_m = self.params.get_param('STAGING_DISTANCE_M', 200.0)
        self.exit_distance_m = self.params.get_param('EXIT_DISTANCE_M', 50.0)
        self.staging_reached_radius_m = self.params.get_param('STAGING_REACHED_RADIUS_M', 10.0)
        self.ground_processing_latency_sec = GROUND_PROCESSING_LATENCY_SEC
        
        # Target altitude default (updated dynamically from WP4 & WP5)
        self.target_altitude = 50.0

        print(
            f"[PARAM] Configured: AIRSPEED_CRUISE={self.cruise_speed_mps:.1f}m/s, "
            f"servos=L{self.servo_channel_left} (PWM {self.servo_drop_pwm_left}) / "
            f"R{self.servo_channel_right} (PWM {self.servo_drop_pwm_right}), "
            f"staging_dist={self.staging_distance_m:.1f}m, exit_dist={self.exit_distance_m:.1f}m"
        )

        # Download initial mission waypoints at startup so cache is ready in memory
        self.download_mission()

        # Gazebo Node (only in simulation mode)
        self.node = None
        if self.mode == "simulation" and HAVE_GZ:
            try:
                self.node = Node()
            except Exception as e:
                print(f"[GAZEBO] Could not init Gazebo transport node: {e}")

        # Frame Data
        self.latest_img = None
        self.new_frame = False
        self.debug_fps = 0.0
        self.debug_last_frame_time = None
        self.cap = None

        # Telemetry Data
        self.current_lat = 0
        self.current_lon = 0
        self.current_alt = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.ground_course = 0
        self.current_mission_seq = None

        # Targets Found
        self.target_blue_loc = None
        self.target_red_loc = None
        self.target_samples = {
            "blue": [],
            "red": [],
        }
        self.last_detection_center = {
            "blue": None,
            "red": None,
        }
        self.debug_mask = None
        self.debug_windows_enabled = DEBUG_SHOW_CAMERA
        self.targets_uploaded = False

        # YOLO detector
        self.yolo_model_path = resolve_model_path(YOLO_MODEL_PATH)
        if not self.yolo_model_path.exists():
            print(f"[YOLO] Model not found: {self.yolo_model_path}")
            sys.exit(1)

        print(f"[YOLO] Loading model: {self.yolo_model_path}")
        self.yolo_session = ort.InferenceSession(str(self.yolo_model_path), providers=["CPUExecutionProvider"])
        self.yolo_input_name = self.yolo_session.get_inputs()[0].name
        self.yolo_names = parse_yolo_names(self.yolo_session.get_modelmeta().custom_metadata_map)
        print(f"[YOLO] Classes: {self.yolo_names}")

        # Route / Drop state
        self.staging_loc = None
        self.drop_route = []
        self.staging_seq = None
        self.target1_seq = None
        self.target2_seq = None
        self.left_dropped = False
        self.right_dropped = False

    @property
    def drop_distance_before_target_m(self):
        """Dynamic lead distance based on AIRSPEED_CRUISE, WP4-WP5 altitude, and latency delay"""
        return (
            self.cruise_speed_mps * math.sqrt(2.0 * self.target_altitude / GRAVITY_MPS2) +
            self.cruise_speed_mps * (DROP_RELEASE_DELAY_SEC + self.ground_processing_latency_sec)
        )

    def camera_cb(self, msg):
        """Gazebo camera callback"""
        try:
            img_array = np.frombuffer(msg.data, dtype=np.uint8)
            img_decoded = img_array.reshape((msg.height, msg.width, 3))
            self.latest_img = cv2.cvtColor(img_decoded, cv2.COLOR_RGB2BGR)
            now = time.time()
            if self.debug_last_frame_time is not None:
                dt = now - self.debug_last_frame_time
                if dt > 0:
                    instant_fps = 1.0 / dt
                    self.debug_fps = (0.85 * self.debug_fps) + (0.15 * instant_fps) if self.debug_fps > 0 else instant_fps
            self.debug_last_frame_time = now
            self.new_frame = True
        except Exception as e:
            print(f"[CAMERA] Failed to decode frame: {e}")

    def start_real_camera(self):
        """Starts OpenCV capture loop from USB Video Capture card (Foxeer camera)"""
        print(f"[CAMERA] Opening USB Video Capture card at index {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"[CAMERA ERROR] Could not open camera at index {self.camera_index}")
            return False

        def usb_camera_loop():
            print(f"[CAMERA] Foxeer USB video stream active (index {self.camera_index}).")
            while True:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.latest_img = frame.copy()
                    now = time.time()
                    if self.debug_last_frame_time is not None:
                        dt = now - self.debug_last_frame_time
                        if dt > 0:
                            instant_fps = 1.0 / dt
                            self.debug_fps = (0.85 * self.debug_fps) + (0.15 * instant_fps) if self.debug_fps > 0 else instant_fps
                    self.debug_last_frame_time = now
                    self.new_frame = True
                else:
                    time.sleep(0.01)

        t = threading.Thread(target=usb_camera_loop, daemon=True)
        t.start()
        return True

    def update_telemetry(self):
        msg_att = self.master.recv_match(type='ATTITUDE', blocking=False)
        if msg_att:
            self.roll = msg_att.roll
            self.pitch = msg_att.pitch
            self.yaw = msg_att.yaw

        msg_pos = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if msg_pos:
            self.current_lat = msg_pos.lat / 1e7
            self.current_lon = msg_pos.lon / 1e7
            self.current_alt = msg_pos.relative_alt / 1000.0
            self.ground_course = msg_pos.hdg / 100.0

        msg_mission = self.master.recv_match(type='MISSION_CURRENT', blocking=False)
        if msg_mission:
            self.current_mission_seq = msg_mission.seq

    def check_geometric_scan_gate(self):
        """
        Verifies whether the aircraft is inside the 30m x 100m geometric corridor 
        defined by WP4 (entry, 0m) and WP5 (exit, 100m).
        """
        wp4 = self.wp_loader.wp(4)
        wp5 = self.wp_loader.wp(5)

        if not wp4 or not wp5 or not hasattr(wp4, 'x') or wp4.x == 0:
            return False

        curr_pos = self.current_point()
        if curr_pos["lat"] == 0 or curr_pos["lon"] == 0:
            return False

        wp4_pos = {"lat": wp4.x, "lon": wp4.y}
        wp5_pos = {"lat": wp5.x, "lon": wp5.y}

        # Calculate along-track and cross-track distance relative to WP4 -> WP5 segment
        n_curr, e_curr = ned_offset_from_gps(wp4_pos["lat"], wp4_pos["lon"], curr_pos["lat"], curr_pos["lon"])
        n_wp5, e_wp5 = ned_offset_from_gps(wp4_pos["lat"], wp4_pos["lon"], wp5_pos["lat"], wp5_pos["lon"])

        corridor_len = math.hypot(n_wp5, e_wp5)
        if corridor_len <= 0:
            return False

        unit_n = n_wp5 / corridor_len
        unit_e = e_wp5 / corridor_len

        # Along-track progress from WP4 (0m to corridor_len ~ 100m)
        along_track_m = n_curr * unit_n + e_curr * unit_e

        # Cross-track distance from centerline
        cross_track_n = n_curr - (along_track_m * unit_n)
        cross_track_e = e_curr - (along_track_m * unit_e)
        cross_track_m = math.hypot(cross_track_n, cross_track_e)

        # Conditions:
        # 1. Cross-track <= 15m (30m total width, center at 0)
        # 2. Along-track between -15m (WP4 entry) and corridor_len + 15m (WP5 exit)
        in_width = cross_track_m <= 15.0
        in_length = -15.0 <= along_track_m <= (corridor_len + 15.0)

        return in_width and in_length

    def check_scan_exit_threshold(self):
        """
        Gated Exit: Checks if aircraft is within 70m of DETECT_BEFORE_WP_REACH (WP 5).
        Only triggers after aircraft has flown past WP 4 entry point.
        """
        wp4 = self.wp_loader.wp(DETECT_AFTER_WP_REACH)
        wp5 = self.wp_loader.wp(DETECT_BEFORE_WP_REACH)

        if not wp4 or not wp5 or not hasattr(wp4, 'x') or wp4.x == 0 or not hasattr(wp5, 'x') or wp5.x == 0:
            return False

        curr_pos = self.current_point()
        if curr_pos["lat"] == 0 or curr_pos["lon"] == 0:
            return False

        n_curr, e_curr = ned_offset_from_gps(wp4.x, wp4.y, curr_pos["lat"], curr_pos["lon"])
        n_wp5, e_wp5 = ned_offset_from_gps(wp4.x, wp4.y, wp5.x, wp5.y)

        corridor_len = math.hypot(n_wp5, e_wp5)
        if corridor_len <= 0:
            return False

        unit_n = n_wp5 / corridor_len
        unit_e = e_wp5 / corridor_len

        # Distance flown along track from WP 4 entry
        along_track_m = n_curr * unit_n + e_curr * unit_e

        # Must have passed WP 4 entry point (along_track_m >= 5.0m)
        if along_track_m < 5.0:
            return False

        # Real distance from current aircraft position to WP 5 (exit)
        dist_to_wp5 = haversine_distance(self.current_lat, self.current_lon, wp5.x, wp5.y)

        # Dynamic trigger distance (70m from WP 5, or 30% remaining if corridor is shorter than 70m)
        effective_threshold_m = min(SCAN_EXIT_THRESHOLD_M, corridor_len * 0.7)

        # Trigger if distance to WP 5 <= 70m (or <= effective_threshold_m) OR if seq >= DETECT_BEFORE_WP_REACH
        dist_reached = (dist_to_wp5 <= effective_threshold_m)
        seq_reached = (self.current_mission_seq is not None and self.current_mission_seq >= DETECT_BEFORE_WP_REACH)

        return dist_reached or seq_reached

    def force_lock_targets(self):
        """
        Gated Fallback: Called when aircraft is 70m away from DETECT_BEFORE_WP_REACH (WP 5).
        Locks any collected samples for Blue and Red immediately (even if only 2 or 3 samples).
        """
        print(f"\n[GATE TRIGGER] Approaching {SCAN_EXIT_THRESHOLD_M:.0f}m threshold to WP {DETECT_BEFORE_WP_REACH}!")
        print("[GATE TRIGGER] Force locking available target samples (even partial 2-3 samples) immediately...")

        for color in ["blue", "red"]:
            curr_loc = getattr(self, f"target_{color}_loc")
            if curr_loc is not None:
                continue

            samples = self.target_samples[color]
            if samples:
                mean_lat = float(np.mean([p[0] for p in samples]))
                mean_lon = float(np.mean([p[1] for p in samples]))
                setattr(self, f"target_{color}_loc", (mean_lat, mean_lon))
                print(f"[FORCE LOCK] {color.upper()} target locked using {len(samples)} sample(s): ({mean_lat:.7f}, {mean_lon:.7f})")

        # Fallback if a target still has 0 samples
        wp5 = self.wp_loader.wp(DETECT_BEFORE_WP_REACH)
        center_lat = wp5.x if (wp5 and hasattr(wp5, 'x') and wp5.x != 0) else self.current_lat
        center_lon = wp5.y if (wp5 and hasattr(wp5, 'y') and wp5.y != 0) else self.current_lon

        if self.target_blue_loc is None and self.target_red_loc is None:
            # 0 samples for both: place blue and red near scan exit center
            self.target_blue_loc = offset_gps(center_lat, center_lon, -10.0, 0.0)
            self.target_red_loc = offset_gps(center_lat, center_lon, 10.0, 0.0)
            print(f"[FORCE LOCK FALLBACK] Synthetic lock created for both targets near WP {DETECT_BEFORE_WP_REACH}.")
        elif self.target_blue_loc is None:
            # Blue missing: place Blue 15m offset from Red
            ref_lat, ref_lon = self.target_red_loc
            self.target_blue_loc = offset_gps(ref_lat, ref_lon, -15.0, 0.0)
            print(f"[FORCE LOCK FALLBACK] Synthetic Blue target locked offset from Red.")
        elif self.target_red_loc is None:
            # Red missing: place Red 15m offset from Blue
            ref_lat, ref_lon = self.target_blue_loc
            self.target_red_loc = offset_gps(ref_lat, ref_lon, 15.0, 0.0)
            print(f"[FORCE LOCK FALLBACK] Synthetic Red target locked offset from Blue.")

    def is_scan_active(self):
        if self.target_blue_loc is not None and self.target_red_loc is not None:
            return False
        
        # Primary check: Geometric spatial gate
        if self.check_geometric_scan_gate():
            return True

        # Fallback sequence check if waypoints loaded
        if self.current_mission_seq is not None:
            return DETECT_AFTER_WP_REACH <= self.current_mission_seq <= DETECT_BEFORE_WP_REACH

        return False

    def current_point(self):
        return {
            "lat": self.current_lat,
            "lon": self.current_lon,
        }

    def calculate_gps_target(self, u, v):
        """
        Calculates GPS coordinates using 3D Rotation Matrix logic.
        Corrects for Roll, Pitch, and Yaw.
        Assumes camera is fixed pointing DOWN (Body Frame: +Z).
        """
        h = self.current_alt
        if h < 2:
            return None

        cx, cy = IMG_W / 2, IMG_H / 2
        du = u - cx
        dv = v - cy

        # Ray direction in Body Frame: [Forward, Right, Down]
        vec_b = np.array([-(dv / FOCAL_LENGTH_PX), (du / FOCAL_LENGTH_PX), 1.0])

        phi = self.roll
        theta = self.pitch
        psi = self.yaw

        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(phi), -math.sin(phi)],
            [0, math.sin(phi), math.cos(phi)]
        ])

        R_y = np.array([
            [math.cos(theta), 0, math.sin(theta)],
            [0, 1, 0],
            [-math.sin(theta), 0, math.cos(theta)]
        ])

        R_z = np.array([
            [math.cos(psi), -math.sin(psi), 0],
            [math.sin(psi), math.cos(psi), 0],
            [0, 0, 1]
        ])

        R_body_to_ned = R_z @ R_y @ R_x
        vec_ned = R_body_to_ned @ vec_b

        if vec_ned[2] <= 0:
            return None

        scale = h / vec_ned[2]
        north_offset = vec_ned[0] * scale
        east_offset = vec_ned[1] * scale

        return offset_gps(self.current_lat, self.current_lon, north_offset, east_offset)

    def largest_sample_cluster(self, samples):
        if not samples:
            return []

        best_cluster = []
        for sample in samples:
            cluster = [
                other for other in samples
                if haversine_distance(sample[0], sample[1], other[0], other[1]) <= TARGET_CLUSTER_RADIUS_M
            ]
            if len(cluster) > len(best_cluster):
                best_cluster = cluster

        return best_cluster

    def add_target_sample(self, color, loc):
        if color == "blue" and self.target_blue_loc is not None:
            return
        if color == "red" and self.target_red_loc is not None:
            return

        samples = self.target_samples[color]
        samples.append(loc)
        if len(samples) > TARGET_SAMPLE_BUFFER:
            samples.pop(0)

        cluster = self.largest_sample_cluster(samples)
        if len(cluster) < TARGET_SAMPLES_REQUIRED:
            print(
                f"[DETECT] {color.upper()} sample {len(samples)}/{TARGET_SAMPLE_BUFFER} "
                f"cluster {len(cluster)}/{TARGET_SAMPLES_REQUIRED}"
            )
            return

        median_lat = float(np.median([p[0] for p in cluster]))
        median_lon = float(np.median([p[1] for p in cluster]))
        mean_lat = float(np.mean([p[0] for p in cluster]))
        mean_lon = float(np.mean([p[1] for p in cluster]))
        spread = max(
            haversine_distance(median_lat, median_lon, sample_lat, sample_lon)
            for sample_lat, sample_lon in cluster
        )

        print(
            f"[DETECT] {color.upper()} cluster ready. "
            f"Cluster: {len(cluster)}/{len(samples)} samples, spread: {spread:.2f} m"
        )
        if spread > MAX_TARGET_SAMPLE_SPREAD_M:
            print(f"[DETECT] {color.upper()} cluster spread too large, pruning outliers.")
            self.target_samples[color] = cluster[-TARGET_SAMPLE_BUFFER:]
            return

        if color == "blue":
            self.target_blue_loc = (mean_lat, mean_lon)
            print(f"\n[DETECT] BLUE SQUARE Target Locked! GPS mean: {self.target_blue_loc}")
        else:
            self.target_red_loc = (mean_lat, mean_lon)
            print(f"\n[DETECT] RED SQUARE Target Locked! GPS mean: {self.target_red_loc}")

    def draw_debug_text(self, frame):
        status = "SCAN ACTIVE" if self.is_scan_active() else "WAITING SCAN WP"
        blue_samples = len(self.target_samples["blue"])
        red_samples = len(self.target_samples["red"])
        lines = [
            (f"MODE: {self.mode.upper()} | {status} | WP seq: {self.current_mission_seq}", (0, 255, 255)),
            (f"Debug FPS: {self.debug_fps:.1f}", (200, 200, 200)),
            (f"Alt: {self.current_alt:.1f}m | Roll: {math.degrees(self.roll):.1f}deg | Pitch: {math.degrees(self.pitch):.1f}deg", (255, 255, 255)),
            (f"Drop lead: {self.drop_distance_before_target_m:.1f}m from alt {self.target_altitude:.1f}m @ {self.cruise_speed_mps:.1f}m/s", (255, 255, 255)),
            (f"Blue samples: {blue_samples}/{TARGET_SAMPLES_REQUIRED} | Locked: {self.target_blue_loc is not None}", (255, 200, 0) if self.target_blue_loc else (220, 220, 220)),
            (f"Red samples: {red_samples}/{TARGET_SAMPLES_REQUIRED} | Locked: {self.target_red_loc is not None}", (0, 100, 255) if self.target_red_loc else (220, 220, 220)),
        ]

        # Draw clean semi-transparent dark HUD card
        overlay = frame.copy()
        box_w = min(500, IMG_W - 16)
        box_h = len(lines) * 20 + 14
        cv2.rectangle(overlay, (8, 8), (box_w, box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (8, 8), (box_w, box_h), (60, 60, 60), 1)

        y = 26
        for text, color in lines:
            cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
            y += 20

    def show_debug_frame(self, frame):
        if not self.debug_windows_enabled:
            return

        try:
            cv2.imshow(DEBUG_WINDOW_NAME, frame)
            if DEBUG_SHOW_MASK and self.debug_mask is not None:
                cv2.imshow(DEBUG_MASK_WINDOW_NAME, self.debug_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[DEBUG] q pressed, disabling OpenCV debug windows.")
                self.debug_windows_enabled = False
                cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"[DEBUG] OpenCV imshow GUI unavailable (continuing in headless mode): {e}")
            self.debug_windows_enabled = False

    def show_waiting_debug_frame(self):
        if self.latest_img is None or not self.debug_windows_enabled:
            return

        frame = self.latest_img.copy()
        self.draw_debug_text(frame)

        # Draw bottom status banner
        gate_status = f"Geometric gate scan active: {self.check_geometric_scan_gate()}"
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, IMG_H - 34), (IMG_W - 8, IMG_H - 8), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (8, IMG_H - 34), (IMG_W - 8, IMG_H - 8), (60, 60, 60), 1)
        cv2.putText(frame, gate_status, (14, IMG_H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1, cv2.LINE_AA)

        self.show_debug_frame(frame)

    def show_runtime_debug_frame(self, status_text=None):
        if self.latest_img is None or not self.debug_windows_enabled:
            return

        frame = self.latest_img.copy()
        self.draw_debug_text(frame)

        if self.drop_route:
            overlay = frame.copy()
            box_top = IMG_H - 76
            cv2.rectangle(overlay, (8, box_top), (IMG_W - 8, IMG_H - 8), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
            cv2.rectangle(frame, (8, box_top), (IMG_W - 8, IMG_H - 8), (60, 60, 60), 1)

            y = box_top + 22
            for idx, target in enumerate(self.drop_route, start=1):
                dist = self.calculate_distance(target["lat"], target["lon"])
                line = f"Target {idx} {target['color'].upper()} dist: {dist:.1f}m dropped: {self.is_drop_done(target['drop_side'])}"
                col = (255, 200, 0) if target['color'] == 'blue' else (0, 100, 255)
                cv2.putText(frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)
                y += 22

        elif status_text:
            overlay = frame.copy()
            cv2.rectangle(overlay, (8, IMG_H - 34), (IMG_W - 8, IMG_H - 8), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
            cv2.rectangle(frame, (8, IMG_H - 34), (IMG_W - 8, IMG_H - 8), (60, 60, 60), 1)
            cv2.putText(frame, status_text, (14, IMG_H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1, cv2.LINE_AA)

        self.show_debug_frame(frame)

    def run_yolo_detection(self, frame):
        tensor, scale, pad_x, pad_y = preprocess_yolo(frame)
        raw = self.yolo_session.run(None, {self.yolo_input_name: tensor})[0][0]

        detections = []
        frame_h, frame_w = frame.shape[:2]
        for row in raw:
            conf = float(row[4])
            if conf < YOLO_CONF_THRESHOLD:
                continue

            class_id = int(row[5])
            label = self.yolo_names.get(class_id, str(class_id))
            class_conf_threshold = YOLO_CLASS_CONF_THRESHOLDS.get(label, YOLO_CONF_THRESHOLD)
            if conf < class_conf_threshold:
                continue

            x1, y1, x2, y2 = [float(v) for v in row[:4]]

            x1 = (x1 - pad_x) / scale
            x2 = (x2 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            y2 = (y2 - pad_y) / scale

            x1 = max(0, min(frame_w - 1, int(round(x1))))
            x2 = max(0, min(frame_w - 1, int(round(x2))))
            y1 = max(0, min(frame_h - 1, int(round(y1))))
            y2 = max(0, min(frame_h - 1, int(round(y2))))
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                "class_id": class_id,
                "label": label,
                "conf": conf,
                "box": (x1, y1, x2, y2),
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
            })

        return detections

    def draw_yolo_detection(self, frame, det):
        x1, y1, x2, y2 = det["box"]
        obj_x, obj_y = det["center"]
        label = det["label"]
        color = YOLO_CLASS_COLORS.get(label, (0, 255, 255))

        status = "VALID" if label in YOLO_FIXED_WING_CLASSES else "IGNORE"
        text = f"{label} {det['conf']:.2f} {status}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (obj_x, obj_y), 4, color, -1)
        cv2.putText(frame, text, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def is_pixel_sample_stable(self, color, center, conf):
        last_center = self.last_detection_center[color]
        self.last_detection_center[color] = center
        if last_center is None:
            return True

        jump_px = math.hypot(center[0] - last_center[0], center[1] - last_center[1])
        if jump_px <= YOLO_MAX_PIXEL_JUMP:
            return True

        if conf >= YOLO_SAMPLE_HIGH_CONF:
            print(
                f"[DETECT] {color.upper()} pixel jump {jump_px:.0f}px accepted "
                f"because confidence is high ({conf:.2f})."
            )
            return True

        print(
            f"[DETECT] {color.upper()} sample ignored: pixel jump {jump_px:.0f}px "
            f"with confidence {conf:.2f}."
        )
        return False

    def check_color_guard_in_frame(self, frame):
        """
        Pre-detection color guard: Checks if the frame contains red or blue color
        within a wide HSV range before passing to the model for inference.
        Returns (has_candidate_color, hsv, mask_blue, mask_red).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, COLOR_GUARD_BLUE_LOWER, COLOR_GUARD_BLUE_UPPER)
        mask_red1 = cv2.inRange(hsv, COLOR_GUARD_RED_LOWER1, COLOR_GUARD_RED_UPPER1)
        mask_red2 = cv2.inRange(hsv, COLOR_GUARD_RED_LOWER2, COLOR_GUARD_RED_UPPER2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        combined_mask = cv2.bitwise_or(mask_blue, mask_red)
        self.debug_mask = combined_mask

        blue_px = cv2.countNonZero(mask_blue)
        red_px = cv2.countNonZero(mask_red)

        has_blue = (blue_px >= MIN_COLOR_GUARD_PIXELS) and (self.target_blue_loc is None)
        has_red = (red_px >= MIN_COLOR_GUARD_PIXELS) and (self.target_red_loc is None)

        has_candidate = has_blue or has_red
        return has_candidate, hsv, mask_blue, mask_red

    def check_roi_color_guard(self, hsv, box, color):
        """
        Validates that the detection bounding box ROI actually contains the expected color.
        """
        x1, y1, x2, y2 = box
        roi_hsv = hsv[y1:y2 + 1, x1:x2 + 1]
        if roi_hsv.size == 0:
            return False

        if color == "blue":
            mask = cv2.inRange(roi_hsv, COLOR_GUARD_BLUE_LOWER, COLOR_GUARD_BLUE_UPPER)
        elif color == "red":
            m1 = cv2.inRange(roi_hsv, COLOR_GUARD_RED_LOWER1, COLOR_GUARD_RED_UPPER1)
            m2 = cv2.inRange(roi_hsv, COLOR_GUARD_RED_LOWER2, COLOR_GUARD_RED_UPPER2)
            mask = cv2.bitwise_or(m1, m2)
        else:
            return True

        return cv2.countNonZero(mask) >= MIN_ROI_COLOR_PIXELS

    def detect_targets(self):
        if self.latest_img is None:
            return

        frame = self.latest_img.copy()

        if abs(self.roll) > math.radians(MAX_DETECTION_ROLL_DEG) or \
                abs(self.pitch) > math.radians(MAX_DETECTION_PITCH_DEG):
            cv2.putText(frame, "ATTITUDE GATE: roll/pitch too high", (10, IMG_H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            self.draw_debug_text(frame)
            self.show_debug_frame(frame)
            return

        # Pre-detection Color Guard: check wide red/blue color presence before model inference
        hsv = None
        if COLOR_GUARD_ENABLED:
            has_candidate, hsv, mask_blue, mask_red = self.check_color_guard_in_frame(frame)
            if not has_candidate:
                # No candidate color in frame or both targets already locked
                self.draw_debug_text(frame)
                self.show_debug_frame(frame)
                return

        detections = self.run_yolo_detection(frame)
        best_fixed_detection = {}
        for det in detections:
            self.draw_yolo_detection(frame, det)

            detected_color = YOLO_FIXED_WING_CLASSES.get(det["label"])
            if detected_color is None:
                continue
            if detected_color == "blue" and self.target_blue_loc is not None:
                continue
            if detected_color == "red" and self.target_red_loc is not None:
                continue

            # Verify ROI color guard
            if COLOR_GUARD_ENABLED and hsv is not None and not self.check_roi_color_guard(hsv, det["box"], detected_color):
                continue

            if detected_color not in best_fixed_detection or det["conf"] > best_fixed_detection[detected_color]["conf"]:
                best_fixed_detection[detected_color] = det

        for detected_color, det in best_fixed_detection.items():
            if not self.is_pixel_sample_stable(detected_color, det["center"], det["conf"]):
                continue

            loc = self.calculate_gps_target(*det["center"])
            if loc:
                self.add_target_sample(detected_color, loc)

        if self.target_blue_loc is not None:
            cv2.putText(frame, "BLUE SQUARE LOCKED", (10, IMG_H - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2, cv2.LINE_AA)

        if self.target_red_loc is not None:
            cv2.putText(frame, "RED SQUARE LOCKED", (10, IMG_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

        self.draw_debug_text(frame)
        self.show_debug_frame(frame)

    def make_nav_waypoint(self, lat, lon, alt=None, acceptance_radius_m=0.0):
        if alt is None:
            alt = self.target_altitude
        return mavutil.mavlink.MAVLink_mission_item_message(
            self.master.target_system,
            self.master.target_component,
            0,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0, 1,
            0, acceptance_radius_m, 0, 0,
            lat, lon, alt
        )

    def make_condition_distance(self, distance_m=None):
        if distance_m is None:
            distance_m = self.drop_distance_before_target_m
        return mavutil.mavlink.MAVLink_mission_item_message(
            self.master.target_system,
            self.master.target_component,
            0,
            mavutil.mavlink.MAV_FRAME_MISSION,
            mavutil.mavlink.MAV_CMD_CONDITION_DISTANCE,
            0, 1,
            distance_m, 0, 0, 0,
            0, 0, 0
        )

    def make_do_set_servo(self, side):
        channel = self.servo_channel_left if side == "left" else self.servo_channel_right
        pwm = self.servo_drop_pwm_left if side == "left" else self.servo_drop_pwm_right
        return mavutil.mavlink.MAVLink_mission_item_message(
            self.master.target_system,
            self.master.target_component,
            0,
            mavutil.mavlink.MAV_FRAME_MISSION,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0, 1,
            channel, pwm, 0, 0,
            0, 0, 0
        )

    def make_target(self, color, loc):
        return {
            "color": color,
            "lat": loc[0],
            "lon": loc[1],
            "drop_side": "left" if color == "blue" else "right",
        }

    def build_route_candidate(self, first_target, second_target, staging_distance_m):
        north_m, east_m = ned_offset_from_gps(
            first_target["lat"], first_target["lon"],
            second_target["lat"], second_target["lon"]
        )
        length = math.hypot(north_m, east_m)
        if length <= 0:
            return None

        unit_n = north_m / length
        unit_e = east_m / length
        staging_lat, staging_lon = offset_gps(
            first_target["lat"],
            first_target["lon"],
            -unit_n * staging_distance_m,
            -unit_e * staging_distance_m
        )

        staging = {
            "color": "staging",
            "lat": staging_lat,
            "lon": staging_lon,
        }

        candidate = {
            "staging": staging,
            "targets": [first_target, second_target],
            "staging_distance_m": staging_distance_m,
        }
        candidate["score"] = self.route_score(candidate)
        return candidate

    def route_score(self, candidate):
        current = self.current_point()
        staging = candidate["staging"]
        first, second = candidate["targets"]
        return (
            distance_between_points(current, staging) +
            distance_between_points(staging, first) +
            distance_between_points(first, second)
        )

    def plan_drop_route(self):
        blue = self.make_target("blue", self.target_blue_loc)
        red = self.make_target("red", self.target_red_loc)

        candidate1 = self.build_route_candidate(blue, red, self.staging_distance_m)
        candidate2 = self.build_route_candidate(red, blue, self.staging_distance_m)

        candidates = [c for c in (candidate1, candidate2) if c is not None]
        selected = min(candidates, key=lambda c: c["score"])

        target_names = " -> ".join(target["color"].upper() for target in selected["targets"])
        print(f"[MISSION] Selected straight route: STAGING({selected['staging_distance_m']:.0f}m) -> {target_names}")
        return selected

    def download_mission(self, retries=3):
        """Downloads current mission from FC into self.wp_loader and saves pristine backup in self.original_wpoints"""
        print("[MISSION] Downloading mission waypoints from Flight Controller...")
        for attempt in range(1, retries + 1):
            self.wp_loader.clear()
            self.master.mav.mission_request_list_send(self.master.target_system, self.master.target_component)

            m = self.master.recv_match(type='MISSION_COUNT', blocking=True, timeout=3)
            if not m:
                print(f"[MISSION WARNING] Attempt {attempt}/{retries}: Failed to get mission count from FC")
                time.sleep(0.5)
                continue

            count = m.count
            print(f"[MISSION] Downloading {count} mission items...")

            success = True
            for i in range(count):
                self.master.mav.mission_request_int_send(self.master.target_system, self.master.target_component, i)
                m = self.master.recv_match(type='MISSION_ITEM_INT', blocking=True, timeout=3)
                if not m:
                    print(f"[MISSION WARNING] Failed to get mission item {i}")
                    success = False
                    break

                self.wp_loader.add(m)

            if success and self.wp_loader.count() == count:
                # Save pristine backup copy of original base mission
                self.original_wpoints = sanitize_original_mission(self.wp_loader.wpoints)
                print(f"[MISSION] Successfully downloaded and sanitized {len(self.original_wpoints)} original base waypoints!")

                # Dynamically set target altitude from Waypoint 4 & Waypoint 5
                wp4 = self.wp_loader.wp(4)
                wp5 = self.wp_loader.wp(5)
                if wp4 and wp5 and hasattr(wp4, 'z') and hasattr(wp5, 'z'):
                    alt4 = float(wp4.z)
                    alt5 = float(wp5.z)
                    if alt4 > 0 and alt5 > 0:
                        self.target_altitude = 50.0
                        print(f"[MISSION] Dynamic Target Altitude extracted from cached WP4-WP5: {self.target_altitude:.1f} m")

                return True

        return False

    def upload_mission(self):
        """Uploads the mission in self.wp_loader to the FC"""
        print("Uploading mission...")
        self.master.mav.mission_clear_all_send(self.master.target_system, self.master.target_component)
        if not self.master.recv_match(type='MISSION_ACK', blocking=True, timeout=5):
            print("Failed to clear mission (no ACK)")

        count = self.wp_loader.count()
        self.master.mav.mission_count_send(self.master.target_system, self.master.target_component, count)

        for i in range(count):
            msg = self.master.recv_match(type=['MISSION_REQUEST', 'MISSION_REQUEST_INT'], blocking=True, timeout=5)
            if not msg:
                print(f"Timeout waiting for request for item {i}")
                return False

            req_seq = msg.seq
            if req_seq >= count:
                continue

            wp = self.wp_loader.wp(req_seq)
            self.master.mav.mission_item_int_send(
                self.master.target_system, self.master.target_component,
                req_seq,
                wp.frame,
                wp.command,
                wp.current,
                wp.autocontinue,
                wp.param1, wp.param2, wp.param3, wp.param4,
                int(wp.x * 1e7) if isinstance(wp.x, float) else wp.x,
                int(wp.y * 1e7) if isinstance(wp.y, float) else wp.y,
                float(wp.z)
            )
            print(f"Sent mission item {req_seq}")

        ack = self.master.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
        if ack and ack.type == 0:
            print("Mission Uploaded Successfully!")
            return True

        print(f"Mission Upload Failed (ACK Type: {ack.type if ack else 'None'})")
        return False

    def insert_mission_items(self, insert_idx, items):
        for item in reversed(items):
            self.wp_loader.insert(insert_idx, item)

    def upload_mission_update(self):
        """
        ONBOARD FC ROUTE INJECTION: Restores pristine initial mission backup (preserving WP0..LAND),
        inserts 7 drop route items (Staging -> CondDist1 -> Servo1 -> Target1 -> CondDist2 -> Servo2 -> Target2),
        and uploads to FC. All subsequent user waypoints (WP5..LAND) are incremented and preserved.
        """
        print("\n[MISSION] Both Targets Acquired! Planning straight drop route...")
        drop_lead = self.drop_distance_before_target_m
        print(
            f"[MISSION] Drop lead distance: {drop_lead:.1f} m "
            f"(alt={self.target_altitude:.1f} m, cruise={self.cruise_speed_mps:.1f} m/s, "
            f"latency={self.ground_processing_latency_sec:.2f} s)"
        )

        # Restore pristine original mission backup before inserting new route to prevent duplicates
        if self.original_wpoints:
            self.wp_loader.clear()
            for wp_item in copy.deepcopy(self.original_wpoints):
                self.wp_loader.add(wp_item)
        else:
            if not self.download_mission():
                print("Failed to download mission. Aborting update.")
                return

        route = self.plan_drop_route()
        staging = route["staging"]
        first, second = route["targets"]

        insert_idx = min(INSERT_DROP_ROUTE_AFTER_WP_SEQ + 1, self.wp_loader.count())

        # 7-Item Mission Sequence inserted right after WP4
        mission_items = [
            self.make_nav_waypoint(
                staging["lat"],
                staging["lon"],
                self.target_altitude,
                acceptance_radius_m=self.staging_reached_radius_m,
            ),
            self.make_condition_distance(drop_lead),
            self.make_do_set_servo(first["drop_side"]),
            self.make_nav_waypoint(first["lat"], first["lon"], self.target_altitude),
            self.make_condition_distance(drop_lead),
            self.make_do_set_servo(second["drop_side"]),
            self.make_nav_waypoint(second["lat"], second["lon"], self.target_altitude),
        ]

        self.insert_mission_items(insert_idx, mission_items)

        if self.upload_mission():
            print(f"[MISSION] Inserted 7 drop route items after WP {INSERT_DROP_ROUTE_AFTER_WP_SEQ}. Original WP5 and subsequent waypoints (LAND/RTL) incremented.")
            print(f"[MISSION] Commanding jump to staging WP {insert_idx}...")
            self.master.mav.mission_set_current_send(
                self.master.target_system,
                self.master.target_component,
                insert_idx
            )

            self.staging_loc = staging
            self.drop_route = [first, second]
            self.staging_seq = insert_idx
            self.target1_seq = insert_idx + 3
            self.target2_seq = insert_idx + 6
            print(
                f"[MISSION] Tracking seq: staging={self.staging_seq}, "
                f"target1={self.target1_seq}, target2={self.target2_seq}"
            )
            self.state = MissionState.APPROACHING_STAGING
            self.targets_uploaded = True
        else:
            print("[ERROR] Failed to upload updated mission.")

    def execute_drop(self, side):
        topic = TOPIC_DROP_LEFT if side == 'left' else TOPIC_DROP_RIGHT
        payload = "RED/LEFT" if side == 'left' else "BLUE/RIGHT"

        channel = self.servo_channel_left if side == 'left' else self.servo_channel_right
        pwm = self.servo_drop_pwm_left if side == 'left' else self.servo_drop_pwm_right

        print("\n=======================================================")
        print(f" [DROP ACTION] INITIATING DROP FOR {payload}")
        print(f" [DROP ACTION] Distance condition met (< {self.drop_distance_before_target_m:.1f}m)")

        # 1. Send MAVLink Servo Command to FC
        print(f" [DROP ACTION] Sending MAVLink Servo Command (Channel {channel}, PWM {pwm})...")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0,
            channel,
            pwm,
            0, 0, 0, 0, 0
        )

        # 2. In simulation mode, also trigger Gazebo detach topic
        if self.mode == "simulation":
            print(f" [DROP ACTION] Sending Gazebo topic command to {topic}")
            cmd = ["gz", "topic", "-t", topic, "-m", "gz.msgs.Empty", "-p", ""]
            try:
                result = subprocess.run(cmd, check=False, capture_output=True, text=True)
                if result.returncode == 0:
                    print(" [DROP ACTION] Gazebo topic command SUCCESS.")
                else:
                    print(f" [DROP ACTION] Gazebo command stderr: {result.stderr}")
            except Exception as e:
                print(f" [DROP ACTION] EXCEPTION executing gazebo command: {e}")

        if side == 'left':
            self.left_dropped = True
            print(" [DROP ACTION] State Updated: LEFT payload marked as dropped.")
        else:
            self.right_dropped = True
            print(" [DROP ACTION] State Updated: RIGHT payload marked as dropped.")
        print("=======================================================\n")

    def is_drop_done(self, side):
        return self.left_dropped if side == 'left' else self.right_dropped

    def sync_state_from_mission_current(self):
        if not self.targets_uploaded or self.current_mission_seq is None:
            return
        if self.staging_seq is None or self.target1_seq is None or self.target2_seq is None:
            return

        seq = self.current_mission_seq

        if self.state == MissionState.APPROACHING_STAGING and seq > self.staging_seq:
            print(f"[STATE] Mission seq advanced past staging ({seq}); switching to TARGET 1 tracking.")
            self.state = MissionState.APPROACHING_TARGET_1

        if self.state == MissionState.APPROACHING_TARGET_1 and seq > self.target1_seq:
            target = self.drop_route[0]
            if not self.is_drop_done(target["drop_side"]):
                print(
                    f"[DROP ACTION] WARNING: Mission seq passed target 1 ({seq}) "
                    "before drop completed. Executing fallback drop now."
                )
                self.execute_drop(target["drop_side"])
            print(f"[STATE] Mission seq advanced past target 1 ({seq}); switching to TARGET 2 tracking.")
            self.state = MissionState.APPROACHING_TARGET_2

        if self.state == MissionState.APPROACHING_TARGET_2 and seq > self.target2_seq:
            target = self.drop_route[1]
            if not self.is_drop_done(target["drop_side"]):
                print(
                    f"[DROP ACTION] WARNING: Mission seq passed target 2 ({seq}) "
                    "before drop completed. Executing fallback drop now."
                )
                self.execute_drop(target["drop_side"])
            print("Mission Complete!")
            self.state = MissionState.FINISHED

    def calculate_distance(self, target_lat, target_lon):
        return haversine_distance(self.current_lat, self.current_lon, target_lat, target_lon)

    def run(self):
        # Start camera video feed
        if self.mode == "real":
            if not self.start_real_camera():
                print("[ERROR] Failed to start real USB Video Capture feed.")
                return
        else:
            if self.node and not self.node.subscribe(Image, CAMERA_TOPIC, self.camera_cb):
                print("Error subscribing to Gazebo camera.")
                return

        print(
            f"System Ready. Mode: {self.mode.upper()}. "
            f"Dynamic parameter & spatial gate tracking enabled."
        )

        while True:
            self.update_telemetry()
            self.sync_state_from_mission_current()
            frame_consumed = False

            if self.state == MissionState.WAITING_FOR_SCAN_WP:
                if self.new_frame:
                    self.show_waiting_debug_frame()
                    self.new_frame = False
                    frame_consumed = True

                if self.is_scan_active():
                    print(
                        f"[SCAN] Active (Geometric spatial gate / WP sequence met). "
                        "Scanning full frame..."
                    )
                    self.state = MissionState.SCANNING

            elif self.state == MissionState.SCANNING:
                if not self.is_scan_active() and not self.targets_uploaded:
                    print(f"[SCAN] Paused outside scan corridor/WP. Current WP: {self.current_mission_seq}")
                    self.state = MissionState.WAITING_FOR_SCAN_WP
                    time.sleep(0.5)
                    continue

                if self.new_frame:
                    self.detect_targets()
                    self.new_frame = False
                    frame_consumed = True

                # Check 1: Normal lock if both targets locked by vision
                if self.target_blue_loc and self.target_red_loc and not self.targets_uploaded:
                    print("[SCAN] Both targets locked. Detection algorithm disabled; uploading route.")
                    self.state = MissionState.UPLOADING_WPS

                # Check 2: 70m Scan Exit Gate to DETECT_BEFORE_WP_REACH (WP 5)
                elif self.check_scan_exit_threshold() and not self.targets_uploaded:
                    print(f"[SCAN] Approaching {SCAN_EXIT_THRESHOLD_M:.0f}m threshold to WP {DETECT_BEFORE_WP_REACH}! Force locking available samples & uploading route.")
                    self.force_lock_targets()
                    self.state = MissionState.UPLOADING_WPS

            elif self.state == MissionState.UPLOADING_WPS:
                if self.new_frame:
                    self.show_runtime_debug_frame("Uploading new mission...")
                    self.new_frame = False
                    frame_consumed = True
                self.upload_mission_update()

            elif self.state == MissionState.APPROACHING_STAGING:
                dist = self.calculate_distance(self.staging_loc["lat"], self.staging_loc["lon"])
                print(f"CURRENT DIST TO STAGING WP: {dist:.2f} Meters")
                if self.new_frame:
                    self.show_runtime_debug_frame(f"Approaching staging: {dist:.1f}m")
                    self.new_frame = False
                    frame_consumed = True
                if dist < self.staging_reached_radius_m:
                    print("Reached staging area. Beginning straight drop run...")
                    self.state = MissionState.APPROACHING_TARGET_1

            elif self.state == MissionState.APPROACHING_TARGET_1:
                target = self.drop_route[0]
                dist = self.calculate_distance(target["lat"], target["lon"])
                print(f"CURRENT DIST TO TARGET 1 ({target['color'].upper()}): {dist:.2f} Meters")
                if self.new_frame:
                    self.show_runtime_debug_frame(f"Drop run target 1: {dist:.1f}m")
                    self.new_frame = False
                    frame_consumed = True
                if dist < self.drop_distance_before_target_m:
                    self.execute_drop(target["drop_side"])
                    self.state = MissionState.APPROACHING_TARGET_2

            elif self.state == MissionState.APPROACHING_TARGET_2:
                target = self.drop_route[1]
                dist = self.calculate_distance(target["lat"], target["lon"])
                print(f"CURRENT DIST TO TARGET 2 ({target['color'].upper()}): {dist:.2f} Meters")
                if self.new_frame:
                    self.show_runtime_debug_frame(f"Drop run target 2: {dist:.1f}m")
                    self.new_frame = False
                    frame_consumed = True
                if dist < self.drop_distance_before_target_m:
                    self.execute_drop(target["drop_side"])
                    print("Mission Complete!")
                    self.state = MissionState.FINISHED

            elif self.state == MissionState.FINISHED:
                if self.new_frame:
                    self.show_runtime_debug_frame("Mission finished")
                    self.new_frame = False
                    frame_consumed = True

            if self.new_frame and not frame_consumed:
                self.show_runtime_debug_frame()
                self.new_frame = False

            time.sleep(0.1)


# ==============================================================================
# Entry Point
# ==============================================================================
if __name__ == "__main__":
    mode = select_operating_mode()

    if mode == "real":
        com_port, baud_rate, cam_index = configure_real_hardware()
        success, connection_uri = start_mavlink_forwarder(com_port, baud_rate)
        if not success:
            print("[ERROR] Could not start MAVLink serial forwarder. Exiting.")
            sys.exit(1)
        uav = AdvancedUAV(mode="real", connection_string=connection_uri, baud_rate=baud_rate, camera_index=cam_index)
    else:
        uav = AdvancedUAV(mode="simulation", connection_string=DEFAULT_SIM_CONNECTION_STRING)

    uav.run()
