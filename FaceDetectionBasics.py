import cv2
import mediapipe as mp
import time


def put_fps(img, pTime):
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(
        img, str(int(fps)), (20, 70),
        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    return pTime


def resize(img, scale_percent=50):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    imgr = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return imgr


mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

cap = cv2.VideoCapture('Videos/5.mp4')
pTime = 0

while True:
    success, img = cap.read()

    img = resize(img, 30)
    pTime = put_fps(img, pTime)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = faceDetection.process(imgRGB)
    if result.detections:
        for id, detection in enumerate(result.detections):
            # mpDraw.draw_detection(img, detection)
            h, w, c = img.shape
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(img, bbox, (255, 0, 0), 2)
            cv2.putText(
                img, str(int(detection.score[0] * 100)),
                (bbox[0], bbox[1] + 25),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
