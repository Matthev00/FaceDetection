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
    return img, pTime


def resize(img, scale_percent=50):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    imgr = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return imgr


class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            min_detection_confidence=minDetectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxs = []

        self.result = self.faceDetection.process(imgRGB)
        if self.result.detections:
            for id, detection in enumerate(self.result.detections):
                h, w, c = img.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                    int(bboxC.width * w), int(bboxC.height * h)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    cv2.rectangle(img, bbox, (255, 0, 0), 2)
                    cv2.putText(
                        img, str(int(detection.score[0] * 100)),
                        (bbox[0], bbox[1] + 25),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        return img, bboxs


def main():
    cap = cv2.VideoCapture('Videos/5.mp4')
    pTime = 0
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img = resize(img, 30)
        img, pTime = put_fps(img, pTime)

        img = detector.findFaces(img)[0]

        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
