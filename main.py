import cv2 as cv
import mediapipe as mp
import time

face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

base_options = mp.tasks.BaseOptions(
    model_asset_path="hand_landmarker.task"
)
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode = mp.tasks.vision.RunningMode.VIDEO
)

landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

start_time = time.perf_counter()

cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', 1280, 720)
fingers = [4,8,12,16,20]
size = 10

def distance(p1, p2):
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]

    d = x**2 + y**2

    return d**0.5

while True:
    ret,frame = cap.read()

    if not  ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    current_time_ms = int((time.perf_counter() - start_time) * 1000)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray)

    image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_frame
    )

    result = landmarker.detect_for_video(image,current_time_ms)
    print("hands:", len(result.hand_landmarks))
    height, width = frame.shape[:2]

    for landmarks in result.hand_landmarks:
        p1 = (landmarks[5].x, landmarks[5].y)
        p2 = (landmarks[8].x, landmarks[8].y)
        d_58 = distance(p1, p2)
        p0 = (landmarks[0].x, landmarks[0].y)
        p5 = (landmarks[5].x, landmarks[5].y)

        d_05 = distance(p0, p5)
        ratio = d_58 / d_05
        print("ratio:", ratio)
        for landmark_id in fingers:

            point = landmarks[landmark_id]

            x_point = point.x
            y_point = point.y


            pixel_x =int (x_point * width)
            pixel_y =int (y_point * height)


            top_left = (pixel_x - size, pixel_y - size)
            bottom_right = (pixel_x + size, pixel_y + size)

            cv.rectangle(frame, top_left, bottom_right, (255, 255, 255), 1)

#    for x, y, w, h in faces:
#        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
#        cv.putText(frame,'pidor', (x, y + h + 25), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv.LINE_AA)
    cv.imshow('frame',frame)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

