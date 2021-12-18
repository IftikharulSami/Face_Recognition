from flask import Flask, render_template, Response, request, url_for, jsonify
import face_recognition as fr
import requests
import cv2
import os
import numpy as np
import base64
import faiss


app = Flask(__name__)
train_emb = np.load('encoding/encodingnorm.npy')
train_emb = np.array(train_emb, dtype=np.float32)
train_names = np.load('encoding/labelsnorm.npy')
emb_dim = 128
index = faiss.IndexFlatL2(emb_dim)
index.add(train_emb)
label = ''; dist = ''

def face_recognition(face):
    global label
    global dist
    locations = fr.face_locations(face, model='cnn')
    test_emb = fr.face_encodings(face, locations, 10)[0]
    norm_enc = test_emb / np.sqrt(np.sum(np.multiply(test_emb, test_emb)))
    tst_emb = np.array(test_emb, dtype=np.float32)
    tst_emb = np.reshape(tst_emb, (1, emb_dim))
    # # print(np.shape(tst_emb))
    D, I = index.search(tst_emb, 3)
    idx = I[0][0]
    distance = D[0][0]
    if distance <= 0.9:
        label = train_names[idx]
        dist = str(distance)
    else:
        label = "Unidentified Person"

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
        # elif request.form['submit']=='Add Image from Gallery':
        #     return render_template('AddImageFromFile.html')
        # elif request.form['submit']=='Add Image from Live Stream':
        #     return render_template('welcome.html')

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        if request.files:
#             if (request.files['unknown_image']: # and not request.files['new_image']):
            unknown_image = request.files['unknown_image'].read()
            npimg = np.fromstring(unknown_image, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            face_recognition(img)
            return render_template('RecFromFile.html', label=label, dist=dist)
    return render_template('RecFromFile.html')

@app.route('/recognizefromcamera', methods=['GET', 'POST'])
def recognizefromcamera():
    counter = 1
    if request.method == 'POST':
        imgEnc = request.json['imageBase64']
        encoded_data = imgEnc.split(',')[1]
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # imgEnc = request.values['imageBase64']
        # (data, enc) = imgEnc.split(';')
        # (type, ext) = data.split('/')
        # (_, encod) = enc.split(',')
        face_recognition(img)
        return jsonify({'label': label, 'dist': dist})
    return render_template('RecFromCamera.html')


if __name__ == '__main__':
    app.run(debug=True)
