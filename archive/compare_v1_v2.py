"""
================================================================================
          EASYCAP VRX CLEAN FULLSCREEN VIEWER & COMPARISON
                         Teknofest Autonomy & Vision
================================================================================

Deskripsi:
Viewer real-time layar penuh murni untuk EasyCap VRX.
Bebas slider, bebas lag, dan tidak merusak hardware register kamera.

Kontrol Keyboard:
  [SPACE]     : Toggle ON / OFF Enhancer (Bandingkan Langsung Fullscreen)
  [TAB]       : Mode Side-by-Side (Kiri: RAW Asli, Kanan: ENHANCED)
  [0]         : Pilih Camera Index 0 (EasyCap VRX)
  [1]         : Pilih Camera Index 1 (Webcam Laptop)
  [2]         : Pilih Camera Index 2 (OBS Virtual Camera)
  [F]         : Toggle Layar Penuh (True Fullscreen)
  [C]         : Simpan Snapshot Foto
  [Q] / [ESC] : Keluar
================================================================================
"""

import sys
import os
import time
import numpy as np
import cv2


class CleanVideoEnhancer:
    """
    Engine pemrosesan citra yang terbukti paling stabil, bersih, dan tajam
    untuk sinyal analog EasyCap VRX Teknofest.
    """
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))

    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]

        # 1. Instantaneous Field Deinterlacing (Hapus Garis Sisir)
        field = frame[1::2, :, :]
        deint = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)

        # 2. YCrCb Denoising & Color Separation
        ycrcb = cv2.cvtColor(deint, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # Fast Auto White Balance
        cr_off = int(np.clip(np.mean(cr) - 128, -20, 20))
        cb_off = int(np.clip(np.mean(cb) - 128, -20, 20))
        if cr_off != 0: cr = cv2.subtract(cr, cr_off)
        if cb_off != 0: cb = cv2.subtract(cb, cb_off)

        # Gentle Chroma Denoising
        cr = cv2.GaussianBlur(cr, (7, 7), 1.5)
        cb = cv2.GaussianBlur(cb, (7, 7), 1.5)

        # Luma Surface Smoothing (Hapus Bintik Semut, Jaga Garis Tepi)
        blur_y = cv2.GaussianBlur(y, (5, 5), 1.5)
        diff = cv2.absdiff(y, blur_y)
        mask_edge = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)[1]
        y_smooth = np.where(mask_edge == 255, y, blur_y)

        # Controlled CLAHE (Kontras Lokal)
        y_clahe = self.clahe.apply(y_smooth)

        # Clean Unsharp Masking
        blur_s = cv2.GaussianBlur(y_clahe, (0, 0), sigmaX=1.0)
        diff_s = cv2.absdiff(y_clahe, blur_s)
        mask_p = cv2.threshold(diff_s, 6, 255, cv2.THRESH_BINARY)[1]
        boost = cv2.addWeighted(y_clahe, 1.35, blur_s, -0.35, 0)
        y_final = np.where(mask_p == 255, boost, y_clahe)

        # Recombine
        merged = cv2.merge([y_final, cr, cb])
        bgr = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

        # Saturation Boost (Warna Terpal Biru & Merah Jadi Pop)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_c, s_c, v_c = cv2.split(hsv)
        s_c = np.clip(s_c.astype(np.float32) * 1.30, 0, 255).astype(np.uint8)
        enhanced = cv2.cvtColor(cv2.merge([h_c, s_c, v_c]), cv2.COLOR_HSV2BGR)

        # Crop border analog kotor
        crop_m = 8
        enhanced = cv2.resize(enhanced[crop_m:-crop_m, crop_m:-crop_m], (w, h), interpolation=cv2.INTER_CUBIC)

        return enhanced


def main(default_camera_index=0):
    enhancer = CleanVideoEnhancer()
    current_idx = default_camera_index

    cap = cv2.VideoCapture(current_idx)
    if not cap.isOpened():
        print(f"[WARN] Gagal membuka camera index {current_idx}. Mencoba index 1...")
        current_idx = 1
        cap = cv2.VideoCapture(current_idx)

    win_name = "EasyCap VRX Viewer - Teknofest Autonomy"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    # State:
    # mode: 'ENHANCED_FULL' (default), 'RAW_FULL', 'SIDE_BY_SIDE'
    view_mode = 0  # 0: Enhanced Fullscreen, 1: Side-by-Side (Before/After), 2: Raw Fullscreen
    mode_names = ["ENHANCED FULLSCREEN", "SIDE-BY-SIDE (RAW | ENHANCED)", "RAW FULLSCREEN"]
    is_fullscreen = False

    status_msg = "Siap. [SPACE] Toggle Filter, [TAB] Side-by-Side, [0/1/2] Ganti Kamera, [F] Fullscreen."
    status_timer = time.time() + 4.0
    fps_time = time.time()
    fps_count = 0.0

    print("\n" + "=" * 65)
    print(f" VIEWER EASYCAP VRX AKTIF (Camera Index: {current_idx}):")
    print("  [SPACE]     : Toggle ON / OFF Filter Langsung")
    print("  [TAB]       : Tampilan Side-by-Side (RAW | ENHANCED)")
    print("  [0]         : Beralih ke Camera Index 0 (EasyCap VRX)")
    print("  [1]         : Beralih ke Camera Index 1 (Webcam Laptop)")
    print("  [2]         : Beralih ke Camera Index 2 (OBS Virtual)")
    print("  [F]         : Toggle Fullscreen")
    print("  [C]         : Simpan Foto Snapshot")
    print("  [Q] / [ESC] : Keluar")
    print("=" * 65 + "\n")

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                time.sleep(0.01)
                continue

            t0 = time.perf_counter()
            enhanced_frame = enhancer.process(raw_frame)
            proc_ms = (time.perf_counter() - t0) * 1000.0

            # Hitung FPS
            now = time.time()
            dt = now - fps_time
            if dt > 0:
                inst = 1.0 / dt
                fps_count = (0.85 * fps_count) + (0.15 * inst) if fps_count > 0 else inst
            fps_time = now

            h, w = raw_frame.shape[:2]
            enh_h, enh_w = enhanced_frame.shape[:2]
            raw_matched = cv2.resize(raw_frame, (enh_w, enh_h))

            # Render Sesuai Mode
            display_img = None
            if view_mode == 0:
                # Enhanced Fullscreen
                display_img = enhanced_frame.copy()
            elif view_mode == 1:
                # Side-by-Side
                lbl_raw = raw_matched.copy()
                lbl_enh = enhanced_frame.copy()
                cv2.rectangle(lbl_raw, (0, 0), (enh_w, 35), (20, 20, 20), -1)
                cv2.putText(lbl_raw, "[RAW VRX ANALOG]", (15, 24), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 255), 2)

                cv2.rectangle(lbl_enh, (0, 0), (enh_w, 35), (20, 20, 20), -1)
                cv2.putText(lbl_enh, f"[ENHANCED] ({proc_ms:.1f}ms)", (15, 24), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 0), 2)
                display_img = np.hstack([lbl_raw, lbl_enh])
            else:
                # Raw Fullscreen
                display_img = raw_matched.copy()

            # HUD Bar Minimalis
            dh, dw = display_img.shape[:2]
            hud = np.zeros((36, dw, 3), dtype=np.uint8)
            hud[:] = (20, 20, 20)

            cv2.putText(hud, f"Cam Index [{current_idx}] | {fps_count:4.1f} FPS | Process: {proc_ms:3.1f}ms | Mode: {mode_names[view_mode]}", 
                        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            if time.time() < status_timer:
                cv2.putText(hud, f">> {status_msg}", (dw - 620, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

            final_view = np.vstack([display_img, hud])
            cv2.imshow(win_name, final_view)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q'), ord('Q')]:
                break
            elif key == 32: # SPACE: Toggle Enhanced vs Raw Fullscreen
                view_mode = 2 if view_mode == 0 else 0
                status_msg = f"Tampilan: {mode_names[view_mode]}"
                status_timer = time.time() + 2.0
            elif key in [9, ord('t'), ord('T')]: # TAB: Side by Side
                view_mode = 1 if view_mode != 1 else 0
                status_msg = f"Tampilan: {mode_names[view_mode]}"
                status_timer = time.time() + 2.0
            elif key in [ord('0'), ord('1'), ord('2')]: # Switch Camera Index
                target_idx = int(chr(key))
                if target_idx != current_idx:
                    status_msg = f"Membuka Camera Index [{target_idx}]..."
                    cap.release()
                    cap = cv2.VideoCapture(target_idx)
                    if cap.isOpened():
                        current_idx = target_idx
                        status_msg = f"Berhasil beralih ke Camera Index [{current_idx}]"
                    else:
                        print(f"[ERROR] Gagal membuka Index [{target_idx}], kembali ke [{current_idx}]")
                        cap = cv2.VideoCapture(current_idx)
                        status_msg = f"Gagal membuka [{target_idx}], kembali ke [{current_idx}]"
                    status_timer = time.time() + 3.0
            elif key in [ord('f'), ord('F')]:
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                status_msg = f"Fullscreen: {'AKTIF' if is_fullscreen else 'OFF'}"
                status_timer = time.time() + 2.0
            elif key in [ord('c'), ord('C')]:
                ts = int(time.time())
                fn = f"easycap_snapshot_{ts}.png"
                comp_snap = np.hstack([raw_matched, enhanced_frame])
                cv2.imwrite(fn, comp_snap)
                print(f"[SNAPSHOT] Disimpan ke {fn}")
                status_msg = f"Snapshot disimpan ({fn})!"
                status_timer = time.time() + 3.0

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[VIEWER] Ditutup.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EasyCap VRX Clean Fullscreen Viewer")
    parser.add_argument("--index", "-i", type=int, default=0, help="Camera Index (0: EasyCap VRX, 1: Webcam)")
    args = parser.parse_args()

    main(default_camera_index=args.index)
