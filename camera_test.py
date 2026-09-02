import cv2

for i in range(5):
    print(f"\nTrying Camera {i}...")

    cap = cv2.VideoCapture(i)

    if cap.isOpened():
        ret, frame = cap.read()

        if ret:
            print(f"Camera {i} is working")

            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
        else:
            print("Opened but couldn't read frame")

        cap.release()
    else:
        print("Not available")