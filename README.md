# 🦅 TEKNOFEST AUTONOMY & VISION - DUAL PAYLOAD DROPPING SYSTEM

Sistem Otonom Deteksi Terpal dan Pelepasan Payload Ganda (*Dual Payload Dropping*) untuk Wahana UAV Tim Teknofest.

---

## 📋 Daftar Isi
1. [Struktur Berkas & Repositori](#-struktur-berkas--repositori)
2. [Kebutuhan Sistem & Instalasi (Requirements)](#-kebutuhan-sistem--instalasi-requirements)
3. [Arsitektur Logika Cross-Dropping](#-arsitektur-logika-cross-dropping)
4. [Petunjuk Penggunaan Script Utama (`teknofest_dual_dropping_mission.py`)](#-petunjuk-penggunaan-script-utama)
5. [Konfigurasi Servo & Payload](#-konfigurasi-servo--payload)
6. [Fitur Clean Video Enhancer (EasyCap VRX)](#-fitur-clean-video-enhancer-easycap-vrx)
7. [MAVLink Telemetry & Mission Planner Forwarder](#-mavlink-telemetry--mission-planner-forwarder)
8. [Panduan Troubleshooting & Tips Kompetisi](#-panduan-troubleshooting--tips-kompetisi)
9. [Catatan Penting untuk Anggota Tim / Penerus](#-catatan-penting-untuk-anggota-tim--penerus)

---

## 📂 Struktur Berkas & Repositori

```text
c:\Users\athif\Downloads\Teknofest\Kode\
│
├── teknofest_dual_dropping_mission.py   # ⭐ SCRIPT UTAMA FULL MISI KOMPETISI
├── v1_gazbmodel_exp.onnx                # 🧠 Model YOLO ONNX (Deteksi square_blue & square_red)
├── requirements.txt                     # 📦 Daftar Library Python
├── README.md                            # 📖 Panduan Teknis Lengkap ini
│
└── archive/                             # 🗄️ Folder Arsip Script Legacy, Modul Uji & Snapshot
    ├── compare_v1_v2.py                 # (Arsip) Viewer Komparasi Video
    ├── easycap_video_enhancer.py        # (Arsip) Modul Enhancer V1
    ├── easycap_video_enhancer_v2.py     # (Arsip) Modul Enhancer V2
    ├── easycap_config.json              # (Arsip) Konfigurasi Preset V1
    ├── easycap_config_v2.json           # (Arsip) Konfigurasi Preset V2
    ├── teknofest_mission2.py            # (Arsip) Script Misi Legacy
    ├── test servo dan deteksi dropping.py # (Arsip) Script Uji Coba Awal
    ├── perfect_easycap_result.png       # (Arsip) Foto Bukti Hasil Enhancer
    └── snapshots_and_tests/             # (Arsip) Hasil Snapshot & Pengujian Kamera
```

---

## 📦 Kebutuhan Sistem & Instalasi (Requirements)

### 1. Sistem Operasi & Driver
* **Sistem Operasi**: Windows 10 / 11 (64-bit).
* **Python**: Python 3.8 s/d Python 3.11 (Disarankan Python 3.10).
* **Driver EasyCap**: USB Video Capture Adapter (DirectShow standard).
* **Driver Telemetri**: SiLabs CP210x / FTDI Driver untuk Radio RFD900 / 3DR Telemetry.

### 2. Instalasi Dependensi Python (Otomatis & Mandiri)
Kedua program (`teknofest_dual_dropping_mission.py` dan `vision_recorder_tester.py`) telah dilengkapi dengan **Auto-Dependency Installer**. Saat pertama kali dijalankan di komputer baru, sistem akan mendeteksi apakah ada modul yang belum terpasang. Jika belum lengkap, program akan otomatis mengunduh dan memasangnya via `pip` sebelum program dimulai!

Jika ingin menginstal secara manual terlebih dahulu:
```powershell
pip install -r requirements.txt
```

Isi paket dependensi:
* `opencv-python>=4.8.0` : Pemrosesan citra dan GUI OpenCV.
* `onnxruntime>=1.16.0`  : Inferensi model deep learning YOLO ONNX di CPU.
* `pymavlink>=2.4.37`    : Komunikasi protokol MAVLink ke Flight Controller (ArduPilot/Pixhawk).
* `pyserial>=3.5`        : Komunikasi serial port USB Radio Telemetri.
* `numpy>=1.23.0`        : Komputasi matriks citra cepat.

---

## 🎯 Arsitektur Logika Cross-Dropping

Sesuai regulasi Teknofest, mekanisme penjatuhan payload menggunakan aturan **Cross-Dropping**:

| Target Terdeteksi di Lapangan | Aksi Dropping Wahana | Channel Servo FC | PWM Standby | PWM Drop |
| :--- | :--- | :---: | :---: | :---: |
| **`square_blue`** (Terpal Biru) | 🔴 **Jatuhkan Payload MERAH** | **Servo 7** (AUX 1) | `1100` | `2200` |
| **`square_red`** (Terpal Merah) | 🔵 **Jatuhkan Payload BIRU** | **Servo 8** (AUX 2) | `2100` | `1100` |

### Sistem Kontrol Deteksi, Dropping & Target Invalid:
Sistem beroperasi dalam mode **Direct Action (Tanpa Blokir Safeguard)** dengan kendali penuh di tangan operator:

1. **Master Toggle [CTRL]**:
   - Proses inferensi deteksi target dan pengiriman perintah dropping otomatis dapat dinyalakan atau dijeda seketika dengan menekan tombol **`[CTRL]`** pada keyboard atau mengklik kartu master di GUI.
   - Status live terlihat jelas di HUD: **`● DETEKSI & DROP: AKTIF`** (Hijau) vs **`○ DETEKSI & DROP: PAUSED`** (Oranye).
2. **Proteksi Target Invalid (`hexagon_blue` & `triangle_red`)**:
   - Model YOLO mendeteksi 4 kelas: `0: triangle_red`, `1: hexagon_blue`, `2: square_red`, `3: square_blue`.
   - Objek `triangle_red` dan `hexagon_blue` **tetap digambar kotak pembatasnya (bounding box)** di layar dengan tag **`[INVALID]`**, namun **TIDAK PERNAH mengirimkan perintah dropping ke Flight Controller**.
3. **Kontrol Tombol Samping Mouse (Fantech & Gaming Mouse)**:
   - Dilengkapi pendeteksian hardware mouse tingkat rendah (*low-latency Windows API*) untuk 2 tombol jempol samping (*thumb side buttons*):
     - **Tombol Samping Atas** (*Forward / Button 5*): Dropping manual untuk sasaran **Merah Square** (Cross-Drop: Servo 8 melepaskan Payload Biru).
     - **Tombol Samping Bawah** (*Back / Button 4*): Dropping manual untuk sasaran **Biru Square** (Cross-Drop: Servo 7 melepaskan Payload Merah).
   - Mode dapat disesuaikan pada variabel `MOUSE_SIDE_BUTTON_MODE = "TARGET"` atau `"PAYLOAD"`.
4. **Telemetri Monitoring Pasif**:
   - Data sikap wahana (Roll, Pitch, Altitude AGL, Flight Mode) tetap dimonitor dan ditampilkan secara halus di HUD sebagai referensi visual pilot tanpa mengunci (*HOLD*) servo.

---

## 🚀 Petunjuk Penggunaan Script Utama (Fast Preparation)

### Menjalankan Misi Nyata (Ground Test / Real Flight)
Jalankan perintah berikut di PowerShell / CMD:

```powershell
python teknofest_dual_dropping_mission.py
```

1. **Pilih Port COM Telemetri**: Program mendeteksi port radio RFD900 Anda (misal `COM7`). Tekan **`[ENTER]`** untuk default.
2. **Pilih Sumber Video Input**:
   * `[0]` EasyCap VRX (Index 0) $\rightarrow$ Kamera analog FPV lomba.
   * `[1]` Kamera Laptop / USB Webcam Lain.
   * `[2]` Berkas Video rekaman di folder `vid_input/` (untuk pengujian offline).
3. **Pilih Model Deteksi YOLO (.onnx)**:
   * `[1]` `v1_gazbmodel_exp.onnx` [DEFAULT]
   * `[2]` `v1main.onnx` / `v2.onnx` (atau berkas model baru lainnya).
4. **Buka Mission Planner** $\rightarrow$ Pilih **`UDP`** di pojok kanan atas $\rightarrow$ Klik **`Connect`** (Port: `14550`).
5. **Hasil**: Mission Planner langsung terkoneksi dalam **0.2 detik dengan Sinyal 100% dan Zero Packet Loss**, sementara script otonom mendeteksi target dan siap melakukan dropping secara simultan!

---

## 🎛️ Kontrol GUI & Tombol Interaktif

Di layar GUI terdapat panel samping kanan dengan tombol uji interaktif yang dapat diklik langsung dengan mouse atau ditekan melalui keyboard:

| Kontrol Input | Pintasan / Tombol | Fungsi Aksi |
| :--- | :---: | :--- |
| **Toggle Deteksi & Auto-Drop** | **`[CTRL]`** / Klik Kartu | **Menyalakan / Menjeda (Pause) deteksi & perintah dropping otomatis**. |
| **Dropping Merah Square** | **`[Side Btn Atas]`** | **Dropping manual sasaran Merah Square** (Servo 8 $\rightarrow$ 1100 PWM lepas Payload Biru). |
| **Dropping Biru Square** | **`[Side Btn Bawah]`** | **Dropping manual sasaran Biru Square** (Servo 7 $\rightarrow$ 2200 PWM lepas Payload Merah). |
| **`[1] MANUAL DROP MERAH`** | **`[1]`** | Memicu pelepasan manual Payload Merah (Servo 7 $\rightarrow$ 2200 PWM). |
| **`[2] RESET SERVO MERAH`** | **`[2]`** | Mengembalikan Servo Merah ke posisi kunci (Servo 7 $\rightarrow$ 1100 PWM). |
| **`[3] MANUAL DROP BIRU`** | **`[3]`** | Memicu pelepasan manual Payload Biru (Servo 8 $\rightarrow$ 1100 PWM). |
| **`[4] RESET SERVO BIRU`** | **`[4]`** | Mengembalikan Servo Biru ke posisi kunci (Servo 8 $\rightarrow$ 2100 PWM). |
| **`[X] RESET SEMUA SERVO`** | **`[X]`** | Mengembalikan kedua servo secara bersamaan ke posisi standby. |
| **`TOGGLE ENHANCER` (Card)** | **`[SPACE]`** | **Toggle Video Enhancer ON/OFF** (Default: **NONAKTIF / Raw Camera**). Status live terlihat pada badge video & kartu sidebar. |
| **Toggle Rekam Video** | **`[R]`** | **Mulai / Hentikan Rekam Video Full** ke folder `video_rec/` (`rec_YYYYMMDD_HHMMSS.mp4`). |
| **Keluar** | **`[Q]` / `[ESC]`** | Menutup program dan melepaskan port secara aman. |ttitude Level Guard (Bypass untuk bench test). |
| **Keluar** | **`[Q]` / `[ESC]`** | Menutup program dan melepaskan port secara aman. |

---

## 📸 Penyimpanan Otomatis Foto Bukti Deteksi (`detected_proof/`)

Setiap kali kode `teknofest_dual_dropping_mission.py` dijalankan:
1. Sistem akan otomatis membuat subfolder baru di dalam `detected_proof/` berdasarkan **tanggal dan jam saat program mulai start** (format: `detected_proof/YYYY-MM-DD_HH-MM-SS/`).
2. Setiap kali target (`square_blue` / `square_red`) terdeteksi di kamera:
   * Foto frame deteksi disimpan **secara langsung tanpa jeda** ke dalam subfolder sesi tersebut.
   * Format nama file: `det_YYYYMMDD_HHMMSS_ms_fXXXXXX_<label>_<conf>pct.jpg`.
   * Foto dilengkapi watermark metadata di bagian bawah (Label, Nilai Confidence, Nomor Frame, Ketinggian Telemetri, dan Status Safeguards).
   * Proses penyimpanan dijalankan pada *background thread* terpisah sehingga **tidak menimbulkan jeda (0 lag)** pada pemrosesan video kamera dan penjatuhan servo.

---

## ⚙️ Konfigurasi Servo, Safeguards & Waypoint

Jika di kemudian hari terdapat perubahan nomor Waypoint, batas kemiringan, atau channel servo, ubah bagian konfigurasi di baris 75–115 pada file [`teknofest_dual_dropping_mission.py`](file:///c:/Users/athif/Downloads/Teknofest/Kode/teknofest_dual_dropping_mission.py):

```python
# ==============================================================================
# KONFIGURASI PAYLOAD & FLIGHT SAFEGUARDS (USER SECTION)
# ==============================================================================

# -- PAYLOAD MERAH (Diturunkan saat Target SQUARE BLUE terdeteksi)
SERVO_RED_CHANNEL = 7          # Channel Servo di Flight Controller (AUX 1 / SERVO 7)
PWM_RED_START = 1000           # PWM Standby / Posisi Mengunci
PWM_RED_DROP = 2200            # PWM Release / Posisi Membuka

# -- PAYLOAD BIRU (Diturunkan saat Target SQUARE RED terdeteksi)
SERVO_BLUE_CHANNEL = 8         # Channel Servo di Flight Controller (AUX 2 / SERVO 8)
PWM_BLUE_START = 2100          # PWM Standby / Posisi Mengunci
PWM_BLUE_DROP = 1100           # PWM Release / Posisi Membuka

# -- FLIGHT SAFEGUARDS (PENGAMAN PENERBANGAN SEBELUM DROPPING)
AUTO_MODE_GUARD_ENABLED = True     # Hanya boleh drop jika Flight Controller di Mode AUTO
TAKEOFF_GUARD_ENABLED = True       # Hanya boleh drop jika ketinggian sudah mencukupi (Takeoff Complete)
LEVEL_GUARD_ENABLED = True         # Hanya boleh drop jika pesawat datar (+/- 10 deg Roll, +/- 8 deg Pitch)
WAYPOINT_GUARD_ENABLED = True      # True: Mulai deteksi & drop HANYA di Waypoint target, False: Bypass
TARGET_WAYPOINTS = [3]             # Nomor Waypoint target (Contoh: [3] atau [3, 4] atau range [3, 4, 5])
MIN_TAKEOFF_ALT_METERS = 30.0      # Ketinggian minimal lepas landas (30 meter AGL / di atas 30%)
TAKEOFF_ALT_PERCENT = 30.0         # Batas ambang minimal: 30% dari Target Ketinggian Misi
MAX_ABS_ROLL_DEG = 10.0            # Toleransi Roll Maksimal (+/- 10 Derajat)
MAX_ABS_PITCH_DEG = 8.0            # Toleransi Pitch Maksimal (+/- 8 Derajat)

# -- YOLO & VISION CONFIGURATION
YOLO_MODEL_PATH = "v1_gazbmodel_exp.onnx"
YOLO_INPUT_SIZE = 640
YOLO_CONF_THRESHOLD = 0.80     # Minimal Confidence 80%
MIN_CONSECUTIVE_FRAMES = 2     # Konfirmasi frame berturut-turut
```

---

## 📹 Tool Pendukung: Vision Recorder & Testing Suite (`vision_recorder_tester.py`)

Selain script misi utama, terdapat script **`vision_recorder_tester.py`** yang berfungsi untuk **merekam video analog EasyCap** saat terbang/uji darat, dan **menguji deteksi model YOLO** menggunakan berkas video pengujian atau kamera live:

### 📁 Struktur Folder Video:
* **`vid input/`** (atau `vid_input/`): **Folder Input**. Tempat Anda menaruh berkas video uji (misal video rekaman penerbangan sebelumnya `.mp4` / `.avi`) untuk dievaluasi oleh model YOLO.
* **`video_rec/`**: **Folder Output Rekaman**. Semua rekaman baru yang diambil melalui kamera live saat Anda menekan tombol **`[R]`** akan otomatis tersimpan di sini (`rec_YYYYMMDD_HHMMSS.mp4`).

### 1. Menjalankan Tool
```powershell
# Jalankan menu interaktif:
python vision_recorder_tester.py

# Atau langsung jalankan Mode Rekam Kamera Live (EasyCap Index 0):
python vision_recorder_tester.py --mode live --cam 0

# Atau langsung uji video tertentu di folder 'vid_input/' dengan Model v2:
python vision_recorder_tester.py --video "vid_input/20110814_042946.avi" --model v2.onnx
```

### 2. Fitur & Pintasan Keyboard
* **`Slider [Timeline]`**: Geser slider trackbar di bagian atas jendela untuk melompat (*seek*) langsung ke frame/detik video mana pun yang diinginkan.
* **`[SPACE]`**: Pause / Resume video (sangat berguna saat evaluasi frame video rekaman).
* **`[D]` / `[A]`**: Maju / Mundur 1 frame (*step frame*) saat video di-pause untuk inspeksi presisi.
* **`[W]`**: Rewind / putar ulang video dari awal (frame 0).
* **`[R]`**: Mulai / Stop merekam video. File video otomatis tersimpan ke folder **`video_rec/`** dengan format `rec_YYYYMMDD_HHMMSS.mp4`. Indikator kedip merah `● REC` akan muncul di layar.
* **`[TAB]`**: Ganti mode tampilan (Fullscreen Enhanced + YOLO / Side-by-Side `[RAW | ENHANCED]` / Clean Enhanced / Raw).
* **`[+]` / `[-]`**: Naikkan / turunkan ambang batas *confidence* YOLO ($\pm 5\%$) secara live.
* **`[G]`**: Aktifkan / nonaktifkan filter HSV Color Guard.
* **`[E]`**: Aktifkan / nonaktifkan pemrosesan citra Enhancer V2.
* **`[C]`**: Ambil snapshot foto resolusi tinggi ke folder `video_rec/`.
* **`[Q]` / `[ESC]`**: Keluar dari aplikasi.

---

## 👁️ Fitur Clean Video Enhancer (EasyCap VRX)

Sinyal analog 5.8 GHz dari receiver VRX yang masuk melalui USB EasyCap memiliki kelemahan: garis sisir (*interlacing*), *noise grain*, dan warna pudar. Script ini mengintegrasikan pipeline pemrosesan citra real-time:

1. **Instant Bob Deinterlacing**: Menghapus 100% efek robek garis horizontal akibat gerakan cepat pesawat.
2. **Auto White Balance (AWB)**: Mengoreksi *color cast* pada ruang warna YCrCb.
3. **Gentle Chroma Denoise**: Membersihkan bintik merah/hijau derau RF analog.
4. **Smart Luma Surface Smoothing**: Menghilangkan derau bintik semut pada rumput dan aspal namun tetap mempertahankan ketajaman garis tepi terpal.
5. **Controlled CLAHE (1.4 Clip)**: Memperjelas kontras target di bawah bayangan awan atau terik matahari.
6. **Saturation Boost (1.30x)**: Mempertegas warna biru dan merah terpal agar bounding box YOLO sangat akurat.

---

## 📡 MAVLink Telemetry & Mission Planner Forwarder

Script ini memiliki fitur **Dual UDP Forwarder**:
* Menghubungkan serial radio RFD900 langsung ke script.
* Meneruskan paket data MAVLink secara dua arah ke **UDP Port `14550`**.
* **Keuntungan**: Anda dapat membuka **Mission Planner** di laptop yang sama secara bersamaan (pilih koneksi `UDP` port `14550`) untuk melihat peta navigasi UAV, status baterai, dan waypoint tanpa bentrok dengan script otonom!

---

## 🛠️ Panduan Troubleshooting & Tips Kompetisi

### 1. Layar Kamera Putih Polos / Overexposed
* **Penyebab**: Sinyal capture card dibuka di index yang salah atau terpicu bug auto gain Windows.
* **Solusi**: Pastikan kabel RCA EasyCap terhubung kencang ke VRX dan VRX dalam posisi ON sebelum menjalankan script. EasyCap berada di **Index 0**. (Index 1 adalah Webcam Laptop).

### 2. Port COM "Access is Denied" / "PermissionError"
* **Penyebab**: Port COM RFD900 sedang dibuka oleh software lain (seperti Mission Planner yang terhubung langsung via COM, Arduino IDE, atau terminal lain).
* **Solusi**: Disconnect koneksi COM di Mission Planner terlebih dahulu. Jalankan script ini, lalu hubungkan Mission Planner melalui koneksi **UDP 14550**.

### 3. Model YOLO Tidak Terdeteksi
* Pastikan file model ONNX bernama **`v1_gazbmodel_exp.onnx`** berada di folder yang sama dengan script.

---

## 📝 Catatan Penting untuk Anggota Tim / Penerus

1. **Jangan mengubah hardware properties EasyCap secara paksa** (`cap.set(cv2.CAP_PROP_CONTRAST)`): Driver USB EasyCap pada Windows sangat rentan mengalami *register lock* jika nilai kontras/brightness diubah melalui OpenCV. Biarkan hardware pada default pabrik dan lakukan penyesuaian visual lewat software enhancer.
2. **Kalibrasi Servo di Lapangan**: Selalu lakukan uji manual menggunakan tombol **`[1]`** dan **`[3]`** di darat sebelum pesawat *takeoff* untuk memastikan pin mekanik membuka dengan sempurna.
3. **Pencatatan Log Misi**: Setiap aksi penjatuhan payload akan mencetak log waktu, nilai PWM, dan tingkat *confidence* ke terminal untuk keperluan review pasca penerbangan.

---
*Dikembangkan untuk Divisi Wahana Dirgantara / Autonomy Vision UAV Teknofest.*
