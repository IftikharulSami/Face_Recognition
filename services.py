import cv2
import face_recognition as fr
import numpy as np
import os

class FR_Services ():
    def __init__(self):
        pass
    def gen_labels_from_images(self):
        pass
    def get_image_from_camera(self, address=0):
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
    def get_image_from_file(self, address):
        img = cv2.imread(address)
        FR_Services.detect_face_locations(img)
    def detect_face_locations(self, image):
        face_locations = fr.face_locations(image, model='cnn')
        fr.face_encodings(image)
        FR_Services.gen_face_enc(image, face_locations)
    def gen_face_enc (self, image, face_locations):
        face_enc = fr.face_encodings(image, face_locations)[0]
        face_encoding = np.array(face_enc)
        np.save(r'Train/encodings.npy', face_encoding)
    def retrain(self, image, label):
        parent_dir = 'Train'
        directory = str(label)
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
        os.chdir('/Train')

        cv2.imwrite(r'Train/'+label, image)
        pass
    def face_recognize(test_image):
        train_faces = np.load(r'encoding/encoding.npy')
        train_names = np.load(r'encoding/labels.npy')
        test_image = fr.load_image_file(test_image)
        locations = fr.face_locations(test_image, model='cnn')
        test_enc = fr.face_encodings(test_image, locations)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
        for face_encoding, face_locations in zip(test_enc, locations):
            results = fr.compare_faces(train_faces, face_encoding, 0.7)
            match = None
            if True in results:
                match = train_names[results.index(True)]
                # print(match)
            return match
