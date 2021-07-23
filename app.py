from flask import Flask, render_template, Response
# from camera import VideoCamera
import requests
import cv2
from face_recognition import FR_Services

app = Flask(__name__)
# cap = cv2.VideoCapture(0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/face_recognition/')
def recog():
    return render_template('face_recognition.html')

@app.route('/retrain/')
def retrain():
    return Response(FR_Services.get_image("http://127.0.0.1:3000/"), mimetype='multipart/x-mixed-replace; boundary=frame')
    # return render_template('retrain.html')

def gen(camera):
    while True:
        data = camera.get_frame()
        frame = data[0]
        yield (b'--frame\r\n'b'content-type: image/jpeg\r\n\r\n'+frame+b'\r\n')
            
    
# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route. Put this in the src attribute of an img tag."""
#     return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    app.run(debug=True)
