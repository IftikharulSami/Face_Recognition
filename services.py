import cv2
import face_recognition as fr
import numpy as np
import os


class FR_Services ():
    def __init__(self):
        self.train_faces = np.load(r'encoding/encoding.npy')
        self.train_names = np.load(r'encoding/labels.npy')
    def gen_labels_from_images(self):
        pass
    def get_image_from_camera(address=0):
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
    def get_image_from_file(address):
        img = cv2.imread(address)
        FR_Services.detect_face_locations(img)
    def read_image(image):
        img = fr.load_image_file(image)
        return FR_Services.detect_face_locations(img)
    def update_enc_labels(self, image, label):
#         os.remove(r'encoding/encoding.npy')
#         os.remove(r'encoding/labels.npy')
#         train_faces = []
#         train_names = []
#         Train_images = r'images/Train'
#         for name in os.listdir(Train_images):
#             for filename in os.listdir(f'{Train_images}/{name}'):
#                 image = fr.load_image_file(f'{Train_images}/{name}/{filename}')
#                 locations = fr.face_locations(image, model='cnn')  # cnn
#                 encoding = fr.face_encodings(image, locations)[0]
#                 train_faces.append(encoding)
#                 train_names.append(name)
#         np.save(r'encoding/encoding.npy', train_faces)
#         np.save(r'encoding/labels.npy', train_names)
#         train_names.clear()
#         train_names.clear()
        new_enc = []
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        locations = fr.face_locations(frame, model='cnn')
        encoding = fr.face_encodings(frame, locations, num_jitters=10)[0]
        new_enc.append(encoding)
        new_enc = np.array(new_enc)
        new_name = [label]
        new_name = np.array(new_name)
        self.train_faces = np.concatenate((self.train_faces, new_enc), axis=0)
        self.train_names = np.concatenate((self.train_names, new_name), axis=0)
        return 'Model Retraining Complete!'

    def detect_face_locations(image):
        face_locations = fr.face_locations(image, model='cnn')
        # fr.face_encodings(image)
        return face_locations
    def gen_face_enc (image, face_locations):
        face_enc = fr.face_encodings(image, face_locations)[0]
        return face_enc
        # face_encoding = np.array(face_enc)
        # np.save(r'encoding/encodings.npy', face_encoding)
    def retrain(self, image, label):
        img = fr.load_image_file(image)
#         parent_dir = 'images/Train'
#         directory = label
#         path = os.path.join(parent_dir, directory)
#         if os.path.isdir(path):
#             count = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
#             os.chdir(path)
#             x = label.strip()
#             cv2.imwrite(f'{x[0]}{count+1}.jpg', img)
#         else:
#             os.makedirs(path)
#             os.chdir(path)
#             x = label.strip()
#             cv2.imwrite(f'{x[0]}+1.jpg', img)
        status = FR_Services.update_enc_labels(img, label)
        return status


    def face_recognize(self, test_image):
        encod = self.train_faces
        nam = self.train_names
        tst_image = fr.load_image_file(test_image)
        tst_image = cv2.cvtColor(tst_image, cv2.COLOR_RGB2BGR)
        resize_image = cv2.resize(tst_image, (280, 320), interpolation = cv2.INTER_AREA)
        locations = fr.face_locations(resize_image, model='cnn')
        test_enc = fr.face_encodings(resize_image, locations, 5)
        for face_encoding, face_locations in zip(test_enc, locations):
            match_idx = -1
            label = 'UnKnown Person'
            match = fr.face_distance(encod, face_encoding)
            match_idx = np.argmin(match)
            if match_idx != -1 and match[match_idx] <= 0.525:
                label = nam[match_idx]
#            results = fr.compare_faces(train_faces, face_encoding, 0.55)
 #           match = 'Unknown Person'
 #           if True in results:
 #               match = train_names[results.index(True)]
                # print(match)
            return label, match[match_idx]

