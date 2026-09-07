import time
import serial

PORT = "COM8"
BAUD = 115200

try:
    ser = serial.Serial(PORT, BAUD, timeout=0.2)
    time.sleep(1)

    print("1. Mengunci kedua payload (Kirim '0')...")
    ser.write(b"0")
    time.sleep(3)

    print("2. Uji Pelepasan Payload Merah (Kirim '1')...")
    ser.write(b"1")
    time.sleep(3)

    print("3. Uji Pelepasan Payload Biru (Kirim '2')...")
    ser.write(b"2")
    time.sleep(3)

    print("4. Mengembalikan status kunci (Kirim '0')...")
    ser.write(b"0")
    time.sleep(1)

    ser.close()
    print("Pengujian selesai!")

except Exception as e:
    print(f"Error: {e}")