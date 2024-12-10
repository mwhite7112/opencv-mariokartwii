import cv2 as cv


def main():
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        print(frame)

main()