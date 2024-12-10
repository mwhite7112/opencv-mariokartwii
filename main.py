import cv2 as cv
import time
# git push origin main

def safe_capture():
    while True:
        try:
            cap = cv.VideoCapture(0)
            if cap.isOpened():
                return cap
            else:
                print("Could not open video capture device. Retrying...")
                time.sleep(1)  # Wait for a second before retrying
        except Exception as e:
            print(f"Error while creating video capture: {e}")
            time.sleep(1)  # Wait for a second before retrying


def main():
    cap = safe_capture()
    while cap.isOpened():
        ret, frame = cap.read()
        print(frame)

main()