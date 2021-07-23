# from retinaface import RetinaFace
# import faiss
import cv2
# import face_recognition
class FR_Services ():
    def __init__(self):
        pass
    def get_image(self, address=0):
        cap = cv2.VideoCapture(address)
        # det = RetinaFace(quality='fast')
        while True:
            _, frame = cap.read()
            if not _:
                break
            else:
                # frame = cv2.resize(frame, (320, 320))
                # faces = face_recognition.face_locations(frame)
                # faces = det.predict(frame)
                # image2 = det.draw(frame, faces)
                ret, jpg = cv2.imencode('.jpg', frame)
                image = jpg.tobytes()
                # print(faces[0])
                # img = frame[faces[0]['y1']:faces[0]['y2'], faces[0]['x1']:faces[0]['x2']]
                # for (x, y, w, h) in faces:
                #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
                cv2.waitKey(1)
    def detect_face(self, image):
        pass
    def gen_emb (self, image):
        pass
    def face_recognize(self, test):
        pass

