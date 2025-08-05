# import os
# import shutil
# import subprocess
# import torch
# import cv2
# import numpy as np
# from PIL import Image
# from flask import Flask, render_template, request, redirect, url_for
# from werkzeug.utils import secure_filename
# from FreqNet.networks.freqnet import freqnet
# import torchvision.transforms as T

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['SAMPLE_PREDICTION_FOLDER'] = 'sample_prediction_data/uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif', 'mp4', 'mov', 'avi'}
# python_path = os.path.join(os.getcwd(), 'venv39', 'Scripts', 'python.exe')  # ✅ adjust if needed

# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# os.makedirs(app.config['SAMPLE_PREDICTION_FOLDER'], exist_ok=True)

# # -------- FreqNet Wrapper --------
# class FreqNet:
#     def __init__(self, model_path, num_classes=1):
#         self.model = freqnet(num_classes=num_classes)
#         self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
#         self.model.cuda()
#         self.model.eval()

#     def transform_image(self, image):
#         transform = T.Compose([
#             T.Resize((224, 224)),
#             T.ToTensor(),
#             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#         return transform(image)

#     def predict(self, image):
#         image = self.transform_image(image)
#         image = image.unsqueeze(0).cuda()
#         with torch.no_grad():
#             output = self.model(image)
#             prediction = torch.sigmoid(output)
#         return prediction.item()

#     def process_video(self, video_path):
#         cap = cv2.VideoCapture(video_path)
#         fake_frames = 0
#         total_frames = 0
#         frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
#         sample_every_n = max(1, frame_rate // 3)

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             if total_frames % sample_every_n == 0:
#                 image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#                 if self.predict(image) > 0.5:
#                     fake_frames += 1
#             total_frames += 1

#         cap.release()
#         confidence = fake_frames / max(1, total_frames)
#         label = 'FAKE' if confidence > 0.5 else 'REAL'
#         return label, confidence


# # -------- GenConViT Prediction via Subprocess --------
# def run_genconvit_subprocess(model_flag, path):
#     try:
#         result = subprocess.run([
#             python_path, 'prediction.py',
#             '--p', path,
#             '--model', model_flag,
#             '--f', '10'
#         ], capture_output=True, text=True, check=True)
#         return parse_subprocess_output(result.stdout)
#     except subprocess.CalledProcessError as e:
#         print(f"Error in {model_flag} subprocess: {e.stderr}")
#         return 'Unknown', 0.0

# def parse_subprocess_output(output):
#     for line in output.splitlines():
#         if line.startswith("Prediction:"):
#             parts = line.strip().split()
#             if len(parts) >= 3:
#                 try:
#                     return parts[2].upper(), float(parts[1])
#                 except:
#                     return 'Unknown', 0.0
#     return 'Unknown', 0.0

# # -------- Flask Routes --------
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'media' not in request.files:
#         return redirect(request.url)

#     file = request.files['media']
#     if file and file.filename.split('.')[-1].lower() in app.config['ALLOWED_EXTENSIONS']:
#         filename = secure_filename(file.filename)
#         upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         sample_path = os.path.join(app.config['SAMPLE_PREDICTION_FOLDER'], filename)
#         file.save(upload_path)
#         shutil.copy(upload_path, sample_path)

#         # Run GenConViT AE & VAE
#         label_ed, conf_ed = run_genconvit_subprocess('genconvit_ed', sample_path)
#         label_vae, conf_vae = run_genconvit_subprocess('genconvit_vae', sample_path)

#         # Run FreqNet directly
#         model = FreqNet('4-classes-freqnet-v2.pth')
#         if filename.split('.')[-1].lower() in ['mp4', 'mov', 'avi']:
#             label_freq, conf_freq = model.process_video(upload_path)
#         else:
#             image = Image.open(upload_path)
#             conf_freq = model.predict(image)
#             label_freq = 'FAKE' if conf_freq > 0.5 else 'REAL'

#         # Voting
#         votes = [label_ed, label_vae, label_freq]
#         confidences = [conf_ed, conf_vae, conf_freq]
#         final_label = max(set(votes), key=votes.count)
#         avg_conf = sum(confidences) / len(confidences)

#         # Cleanup
#         os.remove(sample_path)

#         return redirect(url_for('result', filename=filename, prediction=final_label.title(), confidence=round(avg_conf, 4)))

#     return redirect(url_for('index'))

# @app.route('/result/<filename>')
# def result(filename):
#     prediction = request.args.get('prediction', 'Unknown')
#     confidence = request.args.get('confidence', 'N/A')
#     return render_template('result.html', filename=filename, detection_result=prediction, confidence=confidence)

# if __name__ == '__main__':
#     app.run(debug=True)


import os
import shutil
import subprocess
import torch
import cv2
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from FreqNet.networks.freqnet import freqnet
import torchvision.transforms as T


from transformers import ViTForImageClassification, ViTImageProcessor

class ViTDeepFakeModel:
    def __init__(self):
        self.model = ViTForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")
        self.processor = ViTImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")
        self.model.eval()

    def predict(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1)[0][predicted_class].item()
        label = self.model.config.id2label[predicted_class].upper()
        return label, confidence



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SAMPLE_PREDICTION_FOLDER'] = 'sample_prediction_data/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif', 'mp4', 'mov', 'avi'}
python_path = os.path.join(os.getcwd(), 'venv39', 'Scripts', 'python.exe')  # Change path if needed

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SAMPLE_PREDICTION_FOLDER'], exist_ok=True)

class FreqNet:
    def __init__(self, model_path, num_classes=1):
        self.model = freqnet(num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.cuda().eval()

    def transform_image(self, image):
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image)

    def predict(self, image):
        image = self.transform_image(image).unsqueeze(0).cuda()
        with torch.no_grad():
            prediction = torch.sigmoid(self.model(image))
        return prediction.item()

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fake_frames = 0
        total_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if total_frames % 5 == 0:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if self.predict(image) > 0.5:
                    fake_frames += 1
            total_frames += 1
        cap.release()
        confidence = fake_frames / max(total_frames, 1)
        label = 'FAKE' if confidence > 0.5 else 'REAL'
        return label, confidence

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def parse_subprocess_output(output):
    for line in output.splitlines():
        if line.startswith("Prediction:"):
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    return parts[2].upper(), float(parts[1])
                except:
                    return 'Unknown', 0.0
    return 'Unknown', 0.0

# def run_genconvit_subprocess(model_type, sample_path):
#     if model_type == 'genconvit_ed':
#         args = ['--model', model_type, '--e', 'genconvit_ed_inference']
#     else:
#         args = ['--model', model_type, '--v', 'genconvit_vae_inference']
#     try:
#         result = subprocess.run(
#             [python_path, 'prediction.py', '--p', sample_path, '--f', '10'] + args,
#             capture_output=True, text=True, check=True
#         )
#         return parse_subprocess_output(result.stdout)
#     except subprocess.CalledProcessError as e:
#         print(f"Error in {model_type} subprocess:\n", e.stderr)
#         return 'Unknown', 0.0

def run_genconvit_subprocess(model_type, sample_path):
    args = []
    if model_type == 'genconvit_ed':
        args = ['--e']  # triggers default: weight/genconvit_ed_inference.pth
    elif model_type == 'genconvit_vae':
        args = ['--v']  # triggers default: weight/genconvit_vae_inference.pth
    else:
        raise ValueError("model_type must be 'genconvit_ed' or 'genconvit_vae'")

    try:
        result = subprocess.run(
            [python_path, 'prediction.py', '--p', sample_path, '--f', '10'] + args,
            capture_output=True, text=True, check=True
        )
        return parse_subprocess_output(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error in {model_type} subprocess:\n", e.stderr)
        return 'Unknown', 0.0


@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'media' not in request.files:
#         return redirect(request.url)

#     file = request.files['media']
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         sample_path = os.path.join(app.config['SAMPLE_PREDICTION_FOLDER'], filename)
#         file.save(upload_path)
#         shutil.copy(upload_path, sample_path)

#         # GenConViT AE & VAE
#         label_ed, conf_ed = run_genconvit_subprocess('genconvit_ed', sample_path)
#         label_vae, conf_vae = run_genconvit_subprocess('genconvit_vae', sample_path)

#         # FreqNet prediction
#         freq_model = FreqNet('4-classes-freqnet-v2.pth')
#         if filename.lower().endswith(('mp4', 'mov', 'avi')):
#             label_freq, conf_freq = freq_model.process_video(upload_path)
#         else:
#             image = Image.open(upload_path).convert('RGB')
#             conf_freq = freq_model.predict(image)
#             label_freq = 'FAKE' if conf_freq > 0.5 else 'REAL'

#         # Voting
#         votes = [label_ed, label_vae, label_freq]
#         confidences = [conf_ed, conf_vae, conf_freq]
#         final_label = max(set(votes), key=votes.count)
#         avg_conf = sum(confidences) / len(confidences)

#         os.remove(sample_path)

#         return redirect(url_for('result', filename=filename, prediction=final_label, confidence=round(avg_conf, 4)))

#     return redirect(url_for('index'))


# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'media' not in request.files:
#         return redirect(request.url)

#     file = request.files['media']
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         sample_path = os.path.join(app.config['SAMPLE_PREDICTION_FOLDER'], filename)

#         # Save and copy for GenConViT
#         file.save(upload_path)
#         shutil.copy(upload_path, sample_path)

#         # -------- GenConViT ED Inference --------
#         label_ed, conf_ed = run_genconvit_subprocess('genconvit_ed', sample_path)
#         print(f"[GenConViT-ED] Prediction: {label_ed}, Confidence: {conf_ed:.4f}")

#         # -------- GenConViT VAE Inference --------
#         label_vae, conf_vae = run_genconvit_subprocess('genconvit_vae', sample_path)
#         print(f"[GenConViT-VAE] Prediction: {label_vae}, Confidence: {conf_vae:.4f}")

#         # -------- FreqNet Prediction --------
#         freq_model = FreqNet('4-classes-freqnet-v2.pth')
#         if filename.lower().endswith(('mp4', 'mov', 'avi')):
#             label_freq, conf_freq = freq_model.process_video(upload_path)
#         else:
#             image = Image.open(upload_path).convert('RGB')
#             conf_freq = freq_model.predict(image)
#             label_freq = 'FAKE' if conf_freq > 0.5 else 'REAL'
#         print(f"[FreqNet] Prediction: {label_freq}, Confidence: {conf_freq:.4f}")

#         # -------- Voting & Final Result --------
#         votes = [label_ed, label_vae, label_freq]
#         confidences = [conf_ed, conf_vae, conf_freq]
#         final_label = max(set(votes), key=votes.count)
#         avg_conf = sum(confidences) / len(confidences)
#         print(f"[Ensemble] Final Prediction: {final_label}, Avg Confidence: {avg_conf:.4f}")

#         # Cleanup
#         os.remove(sample_path)

#         return redirect(url_for(
#             'result',
#             filename=filename,
#             prediction=final_label,
#             confidence=round(avg_conf, 4)
#         ))

#     return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload():
    if 'media' not in request.files:
        return redirect(request.url)

    file = request.files['media']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        sample_path = os.path.join(app.config['SAMPLE_PREDICTION_FOLDER'], filename)

        # Save and copy
        file.save(upload_path)
        shutil.copy(upload_path, sample_path)

        # GenConViT predictions
        label_ed, conf_ed = run_genconvit_subprocess('genconvit_ed', sample_path)
        label_vae, conf_vae = run_genconvit_subprocess('genconvit_vae', sample_path)

        # FreqNet
        freq_model = FreqNet('4-classes-freqnet-v2.pth')
        if filename.lower().endswith(('mp4', 'mov', 'avi')):
            label_freq, conf_freq = freq_model.process_video(upload_path)
        else:
            image = Image.open(upload_path).convert('RGB')
            conf_freq = freq_model.predict(image)
            label_freq = 'FAKE' if conf_freq > 0.5 else 'REAL'

        # HuggingFace ViT model
        if filename.lower().endswith(('jpg', 'jpeg', 'png')):
            vit_model = ViTDeepFakeModel()
            label_vit, conf_vit = vit_model.predict(image)
        else:
            label_vit, conf_vit = 'REAL', 0.5  # Neutral fallback for video

        # Print all predictions
        print(f"[GenConViT-ED] Prediction: {label_ed}")
        print(f"[GenConViT-VAE] Prediction: {label_vae}")
        print(f"[FreqNet] Prediction: {label_freq}")
        print(f"[ViT HuggingFace] Prediction: {label_vit}")

        # Voting
        # votes = [label_ed, label_vae, label_freq, label_vit]
        # confidences = [conf_ed, conf_vae, conf_freq, conf_vit]
        # final_label = max(set(votes), key=votes.count)
        # avg_conf = sum(confidences) / len(confidences)

        # Assign weights
        weights = {
            'genconvit_ed': 0.325,
            'genconvit_vae': 0.325,
            'freqnet': 0.05,
            'vit': 0.3
        }

        # Map predictions to confidence * weight
        label_weights = {
            label_ed: weights['genconvit_ed'] * conf_ed,
            label_vae: weights['genconvit_vae'] * conf_vae,
            label_freq: weights['freqnet'] * conf_freq,
            label_vit: weights['vit'] * conf_vit
        }

        # Aggregate weights for each label
        from collections import defaultdict
        label_scores = defaultdict(float)
        for label, score in label_weights.items():
            label_scores[label] += score

        # Determine the label with highest weighted score
        final_label = max(label_scores.items(), key=lambda x: x[1])[0]
        avg_conf = sum(label_scores.values())  # total weighted confidence

        print(f"[Ensemble] Final Prediction: {final_label}, Avg Confidence: {avg_conf:.4f}")

        os.remove(sample_path)

        return redirect(url_for(
            'result',
            filename=filename,
            prediction=final_label,
            confidence=round(avg_conf, 4)
        ))

    return redirect(url_for('index'))


@app.route('/result/<filename>')
def result(filename):
    prediction = request.args.get('prediction', 'Unknown')
    confidence = request.args.get('confidence', 'N/A')
    return render_template('result.html', filename=filename, detection_result=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
