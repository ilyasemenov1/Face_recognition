import cv2 
import sys
import time

class Video_face_recognition():

    def __init__(self):
        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.video = cv2.VideoCapture(0)

    def video_face_datection(self):
        while True:
            ret, frame =self.video.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()

class Face_recognition():

    def __init__(self, imagePath):
        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.image = cv2.imread(imagePath)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.faces = self.faceCascade.detectMultiScale(
            self.gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

    def face_datection(self):

        print("Found " + str(len(self.faces)) + " faces!      \r", end='')

        for (x, y, w, h) in self.faces:
            cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Faces found", self.image)
        cv2.waitKey(0)

#image_1 = Face_recognition('abba.png')

#if __name__ == '__main__':
    #image_1.face_datection()

video_cam_capture =  Video_face_recognition()

if __name__ == '__main__':
    video_cam_capture.video_face_datection()