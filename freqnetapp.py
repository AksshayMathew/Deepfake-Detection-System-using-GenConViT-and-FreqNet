import os
import shutil
import subprocess
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import cv2
import numpy as np
from PIL import Image
from FreqNet.networks.freqnet import freqnet
import torchvision.transforms as T


# FreqNet\networks\freqnet.py
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SAMPLE_PREDICTION_FOLDER'] = 'sample_prediction_data/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif', 'mp4', 'mov', 'avi'}

# Ensure upload folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SAMPLE_PREDICTION_FOLDER'], exist_ok=True)

# Load FreqNet Model
class FreqNet:
    def __init__(self, model_path, num_classes=1):
        self.model = freqnet(num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.cuda()
        self.model.eval()

    def predict(self, image):
        image = self.transform_image(image)
        image = image.unsqueeze(0).cuda()
        with torch.no_grad():
            output = self.model(image)
            prediction = torch.sigmoid(output)
        return prediction.item()

    def transform_image(self, image):
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fake_frames = 0
        total_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            prediction = self.predict(image)
            if prediction > 0.5:
                fake_frames += 1
            total_frames += 1

        cap.release()
        result = 'Fake' if fake_frames / total_frames > 0.5 else 'Real'
        return {'result': result}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'media' not in request.files:
        return redirect(request.url)
    file = request.files['media']

    if file and file.filename.split('.')[-1].lower() in app.config['ALLOWED_EXTENSIONS']:
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        model = FreqNet('4-classes-freqnet-v2.pth')

        if filename.split('.')[-1].lower() in ['mp4', 'mov', 'avi']:
            result = model.process_video(upload_path)
        else:
            image = Image.open(upload_path)
            prediction = model.predict(image)
            result = 'Fake' if prediction > 0.5 else 'Real'

        return redirect(url_for('result', filename=filename, prediction=result))

@app.route('/result/<filename>')
def result(filename):
    prediction = request.args.get('prediction', 'Unknown')
    return render_template('result.html', filename=filename, detection_result=prediction)

if __name__ == '__main__':
    app.run(debug=True)