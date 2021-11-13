from flask import Flask, render_template, Response, request, url_for, jsonify
# from camera import VideoCamera
import requests
import cv2
import os

from services import FR_Services
app = Flask(__name__)
# cap = cv2.VideoCapture(0)
# app.config["IMAGE_UPLOADS"] = "Train"
ser = FR_Services()

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('welcome.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    usr = ['admin', 'atif', 'imran', 'husain', 'iftikhar']
    pwd = ['admin', 'atif', 'imran', 'husian', 'iftikhar']
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
            return render_template('welcome.html')
        elif request.form['submit']=='Add Image from Live Stream':
            return render_template('welcome.html')

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        if request.files:
#             if (request.files['unknown_image']: # and not request.files['new_image']):
            unknown_image = request.files['unknown_image']
            label, dist = ser.face_recognize(unknown_image)
            return render_template('RecFromFile.html', label=label, dist=dist)
#             elif (not request.files['unknown_image'] and request.files['new_image']):
#                 new_image = request.files['new_image']
#                 value = request.form['new_label']
#                 value = value.title()
#                 reply = ser.retrain(new_image, value)
#                 return render_template('index.html', reply=reply)
    return render_template('RecFromFile.html')

@app.route('/recognizefromcamera', methods=['GET', 'POST'])
def recognizefromcamera():
    counter = 1
    if request.method == 'POST':
        imgEnc = request.json['imageBase64']
        # imgEnc = request.values['imageBase64']
        (data, enc) = imgEnc.split(';')
        (type, ext) = data.split('/')
        (_, encod) = enc.split(',')
        name = 'Temp'
        imgPath = ser.b64toImg(encod, name, ext)
        print(imgPath)
        # st_time = timeit.default_timer()
        label, dist = ser.face_recognize(imgPath)
        # end_time = timeit.default_timer()
        print(label)
        # print('Elapsed Time: ', end_time - st_time)
        return jsonify({'label': label, 'dist': dist})
    return render_template('RecFromCamera.html')

if __name__ == '__main__':
    app.run(debug=True)
