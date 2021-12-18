from flask import Flask, render_template, request, jsonify
import cv2
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from mtcnn import MTCNN
from deepface import DeepFace
from deepface.basemodels import Facenet
import faiss
import base64
import numpy as np

#-------------------------------------
graph = tf.get_default_graph()
#-------------------------------------
app = Flask(__name__)
sess = tf.Session()
set_session(sess)
my_model = Facenet.loadModel('Weights/facenet_weights.h5')
detector = MTCNN()
emb_dim = 128  # 128 for Facenet and 512 for Facenet512
train_emb = np.load('encodings/train_emb128.npy')
train_names = np.load('encodings/train_names128.npy')
index = faiss.IndexFlatL2(emb_dim)
index.add(train_emb)
label = ''
dist = ''


def b64ToImage(imgstr):
    encoded_data = imgstr.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # imgdata = base64.b64decode(str(imgstr))
    # image = Image.open(io.BytesIO(imgdata))
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    # cv2.imshow('Image', img)
    # cv2.waitKey(800)
    # cv2.destroyAllWindows()
    # image = image.astype('float32')
    # print(image.dtype)
    # (h, w, c) = image.shape
    # image = np.reshape(image, (w,h,c))
    # print(image.shape)
    # dlibFaceDet(image)
    MTCNNDetectFace(img)
    # cv2.imwrite('image111.png', image)
    # detectFace()

def MTCNNDetectFace(img):
    # cv2.imshow('image', img)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    with graph.as_default():
        set_session(sess)
        detected_faces = detector.detect_faces(img)
    # detected_faces = DeepFace.detectFace(img, detector_backend='mtcnn')
    num_faces = len(detected_faces)
    print(f'Faces = {num_faces}')
    if num_faces == 0:
        print('No face detected.')
    else:
        print('Face Detected')
        face = detected_faces[0]['box']
        (x, y, w, h) = face
        image = img[y:y + h, x:x + w]
        compare_dims(face, img, detected_faces)

def compare_dims(face, image, detected_faces):
    (x, y, w, h) = face
    # print(x,y,w,h)
    h_limit = 112
    w_limit = 112
    if (h >= h_limit and w >= w_limit):
        print('Yes')
        check_orientation(detected_faces, image)
    else:
        print('Less Face Dimension then required size of 112x112!')
        print(f'Height = {h} and Width = {w}')

def check_orientation(detected_face, image):
    (x, y, width, height) = detected_face[0]['box']
    mid_width = int(width/2)
    mid_height = int(height / 2)
    x_disp = int(width * .15)
    y_disp = int(height * .15)
    (x_nose_orig, y_nose_orig) = detected_face[0]['keypoints']['nose']
    x_nose =  x_nose_orig - x
    y_nose = y_nose_orig - y
    if (x_nose < mid_width - x_disp or x_nose > mid_width + x_disp) or (
            y_nose < mid_height - y_disp or y_nose > mid_height + y_disp):
        print('Profile Face')
    else:
        print('Fontal Face')
        recognition(image)

def recognition(face):
    global label
    global dist
    with graph.as_default():
        set_session(sess)
        tst_emb =  DeepFace.represent(face, model=my_model, detector_backend='mtcnn')
    tst_emb = np.array(tst_emb, dtype=np.float32)
    tst_emb = np.reshape(tst_emb, (1, emb_dim))
    # print(np.shape(tst_emb))
    D, I = index.search(tst_emb, 3)
    idx = I[0][0]
    dist = str(D[0][0])
    label = train_names[idx]
    # print(f'Person Identified as {label}')

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('welcome.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return render_template('welcome.html')
    return render_template('login.html', error=error)

@app.route('/welcome', methods=['GET', 'POST'])
def welcome():
    if request.method == 'POST':
        if request.form['submit']=='Image From Gallery':
            return render_template('RecFromFile.html')
        elif request.form['submit']=='Image from Live Stream':
            return render_template('RecFromCamera.html')
        elif request.form['submit']=='Add Image from Gallery':
            return render_template('AddImageFromFile.html')
        elif request.form['submit']=='Add Image from Live Stream':
            return render_template('welcome.html')

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        if request.files:
#             if (request.files['unknown_image']: # and not request.files['new_image']):
            unknown_image = request.files['unknown_image'].read()
            npimg = np.fromstring(unknown_image, np.uint8)
            # convert numpy array to image
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            # cv2.imshow('Image', img)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()
            MTCNNDetectFace(img)
            return render_template('RecFromFile.html', label=label, dist=dist)
    return render_template('RecFromFile.html')

@app.route('/recognizefromcamera', methods=['GET', 'POST'])
def recognizefromcamera():
    if request.method == 'POST':
        imgEnc = request.json['imageBase64']
        b64ToImage(imgEnc)
        return jsonify({'label': label, 'dist': dist})
    return render_template('RecFromCamera.html')

@app.route('/retrainfromfile', methods=['GET', 'POST'])
def retrainfromfile():
    if request.method == 'POST':
        if request.files:
#             if (request.files['unknown_image']: # and not request.files['new_image']):
            new_image = request.files['unknown_image']
            label = request.form['label']
            status = ser.retrain(new_image, label)
            return render_template('AddImageFromFile.html', label=label, status=status)
#             elif (not request.files['unknown_image'] and request.files['new_image']):
#                 new_image = request.files['new_image']
#                 value = request.form['new_label']
#                 value = value.title()
#                 reply = ser.retrain(new_image, value)
#                 return render_template('index.html', reply=reply)
    return render_template('AddImageFromFile.html')

# @app.route('/face_recognition/')
# def recog():
#     return render_template('face_recognition.html')
#
# @app.route('/retrain/')
# def retrain():
#     return Response(FR_Services.get_image(0), mimetype='multipart/x-mixed-replace; boundary=frame')
#     # return render_template('retrain.html')
#
# def gen(camera):
#     while True:
#         data = camera.get_frame()
#         frame = data[0]
#         yield (b'--frame\r\n'b'content-type: image/jpeg\r\n\r\n'+frame+b'\r\n')
#
    
# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route. Put this in the src attribute of an img tag."""
#     return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
