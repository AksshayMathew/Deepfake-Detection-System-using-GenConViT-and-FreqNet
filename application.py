# from flask import Flask, render_template, request, redirect, url_for
# import os
# from werkzeug.utils import secure_filename
# import shutil
# import torch

# # ---- GenConViT imports ----
# from model.pred_func import (
#     load_genconvit, pred_vid, df_face, df_face_from_image, real_or_fake
# )
# from model.config import load_config

# # ---- Flask App Setup ----
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['SAMPLE_PREDICTION_FOLDER'] = 'sample_prediction_data/uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif', 'mp4', 'mov', 'avi'}

# # ---- Create folders if they don't exist ----
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# os.makedirs(app.config['SAMPLE_PREDICTION_FOLDER'], exist_ok=True)

# # ---- Allowed file check ----
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# # ---- Load model once globally ----
# config = load_config()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = load_genconvit(config, net="genconvit", ed_weight="genconvit_ed_inference", vae_weight="genconvit_vae_inference", fp16=False)

# # ---- Media prediction function ----
# def predict_media(filepath, model, net="genconvit", num_frames=10, fp16=False):
#     # Handle image
#     if filepath.lower().endswith(('.jpg', '.jpeg', '.png')):
#         df = df_face_from_image(filepath, net)
#     else:
#         df = df_face(filepath, num_frames, net)

#     if fp16:
#         df = df.half()

#     if len(df) == 0:
#         return "Unknown"

#     y, y_val = pred_vid(df, model)
#     label = real_or_fake(y)
#     return "Fake" if label == "FAKE" else "Real"

# # ---- Routes ----
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'media' not in request.files:
#         return redirect(request.url)
#     file = request.files['media']

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         sample_path = os.path.join(app.config['SAMPLE_PREDICTION_FOLDER'], filename)

#         # Save and copy
#         file.save(upload_path)
#         shutil.copy(upload_path, sample_path)

#         # Run prediction directly
#         try:
#             prediction = predict_media(sample_path, model)
#         except Exception as e:
#             prediction = f"Error during prediction: {str(e)}"

#         # Clean up
#         if os.path.exists(sample_path):
#             os.remove(sample_path)

#         return redirect(url_for('result', filename=filename, prediction=prediction))

#     return redirect(url_for('index'))

# @app.route('/result/<filename>')
# def result(filename):
#     prediction = request.args.get('prediction', 'Unknown')
#     return render_template('result.html', filename=filename, detection_result=prediction)

# # ---- Run the app ----
# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import shutil
import torch

# GenConViT imports (update to your module path if needed)
from model.pred_func import (
    load_genconvit, pred_vid, df_face, df_face_from_image, real_or_fake
)
from model.config import load_config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SAMPLE_PREDICTION_FOLDER'] = 'sample_prediction_data/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif', 'mp4', 'mov', 'avi'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SAMPLE_PREDICTION_FOLDER'], exist_ok=True)

# Lazy model holder
model = None

# Check allowed file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Predict image or video
def predict_media(filepath, model, net="genconvit", num_frames=10, fp16=False):
    if filepath.lower().endswith(('.jpg', '.jpeg', '.png')):
        df = df_face_from_image(filepath, net)
    else:
        df = df_face(filepath, num_frames, net)

    if fp16:
        df = df.half()

    if len(df) == 0:
        return "Unknown"

    y, y_val = pred_vid(df, model)
    label = real_or_fake(y)
    return "Fake" if label == "FAKE" else "Real"

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Upload route with lazy model loading
@app.route('/upload', methods=['POST'])
def upload():
    global model  # use global to persist across calls

    if 'media' not in request.files:
        return redirect(request.url)
    file = request.files['media']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        sample_path = os.path.join(app.config['SAMPLE_PREDICTION_FOLDER'], filename)

        file.save(upload_path)
        shutil.copy(upload_path, sample_path)

        try:
            # Lazy load model on first request
            if model is None:
                print("[INFO] Loading GenConViT model...")
                device = torch.device("cpu")  # safe default
                config = load_config()
                config["model"]["backbone"] = "convnext_tiny"
                config["model"]["embedder"] = "swin_tiny_patch4_window7_224"
                config["model"]["type"] = "tiny"
                model = load_genconvit(config, net="genconvit", ed_weight="genconvit_ed_inference", vae_weight="genconvit_vae_inference", fp16=False, device=device)

            prediction = predict_media(sample_path, model)
        except Exception as e:
            prediction = f"Error during prediction: {str(e)}"

        # Cleanup
        if os.path.exists(sample_path):
            os.remove(sample_path)

        return redirect(url_for('result', filename=filename, prediction=prediction))

    return redirect(url_for('index'))

@app.route('/result/<filename>')
def result(filename):
    prediction = request.args.get('prediction', 'Unknown')
    return render_template('result.html', filename=filename, detection_result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
