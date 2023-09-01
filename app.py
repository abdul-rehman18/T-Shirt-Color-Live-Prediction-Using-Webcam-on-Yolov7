import io
from operator import truediv
import os
import json
from PIL import Image
import cv2
import numpy as np

import torch
from flask import Flask, jsonify, url_for, render_template, request, redirect, Response

app = Flask(__name__)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# finds the model inside your directory automatically - works only if there is one model

    

model =torch.hub.load("WongKinYiu/yolov7", 'custom','best.pt')
print(type(model))

model.eval()

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images
# Inference
    results = model(imgs, size=640)  # includes NMS
    print(type(results))
    return results

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        
        img_bytes = file.read()
        print(type(img_bytes))
        # print("Img_Byte:",img_bytes)
        results = get_prediction(img_bytes)
        # print("Result: ",type(results))

        results.save(save_dir='static')
        filename = 'image0.jpg'
        
        return render_template('result.html',result_image = filename,model_name = 'best.pt')

    return render_template('index.html')
    
@app.route('/capture', methods=['GET', 'POST'])
def handle_video():
    cap = cv2.VideoCapture(0)
    
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Release the webcam
    cap.release()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, size=640)
    print(type(results))
    print(type(frame))

    results.save(save_dir='static')
    filename = 'image0.jpg'
        
    return render_template('result.html',result_image = filename,model_name = 'best.pt')

    

camera = cv2.VideoCapture(0)

class_names=['blue', 'green', 'red','yellow','purple']

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # ret, buffer = cv2.imencode('.jpg', frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # model(frame_rgb, size=640)
            results = model(frame_rgb)
            df = results.pandas().xyxy[0]
            print(df)
            for index, row in df.iterrows():
                if row['confidence'] > 0.2:
                    class_idx = int(row['class'])  # Get the class index
                    class_name = class_names[class_idx]  # Get the class name
                    label = f"{class_name}: {row['confidence']:.2f}"
                    print(class_idx,class_name,label)
                    cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (255, 155, 0), 2)
                    cv2.putText(frame, label, (int(row['xmin']), int(row['ymin']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 155, 0), 2)
        
            ret, buffer = cv2.imencode('.jpg', frame)
            # frame = buffer.tobytes()
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/webcam')
def webcam():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
