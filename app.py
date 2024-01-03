from flask import Flask,render_template,request,Response,jsonify
import cv2
import base64
import numpy as np
import os
from yolo_object_detection import video_detetcion_live,video_detetcion
app=Flask(__name__)

@app.route('/video_feed')
def video_feed():
    video_detetcion_live()
    #return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return 'success'

@app.route('/upload',methods=['POST'])
def upload():
    file = request.files['file']
    if file.content_type != 'video/mp4':
        return jsonify({'error': 'Invalid file format. Only MP4 videos are allowed.'}), 400

    try:
        # Save the file to the specified directory
        file.save('input_video.mp4')
        video_detetcion('input_video.mp4')
        return jsonify({'message': 'detection closed.'}), 200
    
    except Exception as e:
        return jsonify({'error': f'Error saving file: {str(e)}'}), 500


@app.route('/')
def index():
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)
