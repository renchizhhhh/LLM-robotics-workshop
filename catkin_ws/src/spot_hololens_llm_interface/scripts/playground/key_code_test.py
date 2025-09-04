
import cv2
import time
cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    key = cv2.waitKey(1)
    print(key)
    cv2.imshow('keytest', cv2.flip(image, 1))

    time.sleep(0.1) # 0.1 second delay