from flask import Flask, render_template, Response, request, url_for
# from camera import VideoCamera
import requests
import cv2
from services import FR_Services

app = Flask(__name__)
# cap = cv2.VideoCapture(0)
# app.config["IMAGE_UPLOADS"] = "Train"
ser = FR_Services()
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if request.files:
            if (request.files['unknown_image'] and not request.files['new_image']):
                unknown_image = request.files['unknown_image']
                label, dist = ser.face_recognize(unknown_image)
                return render_template('index.html', label=label, dist=dist)
            elif (not request.files['unknown_image'] and request.files['new_image']):
                new_image = request.files['new_image']
                value = request.form['new_label']
                value = value.title()
                reply = ser.retrain(new_image, value)
                return render_template('index.html', reply=reply)
    return render_template('index.html')

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
