import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(2)

for i in range(10):
    ret, frame = cap.read()
    print(f"Frame {i}: ret={ret}, shape={frame.shape if ret else 'None'}")

cap.release()