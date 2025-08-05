import os
from flask import Flask, request, jsonify
import torch
from werkzeug.utils import secure_filename
from prediction import vids, faceforensics, timit, dfdc, celeb  # Import the prediction function

app = Flask(__name__)

# Set the folder for file uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4', 'avi'}

# Check allowed file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return "Welcome to the Deepfake Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process video or image based on the file type
        if filename.lower().endswith(('mp4', 'avi')):
            # Predict using the GenConViT model for video
            result = vids('genconvit_ed_inference', 'genconvit_vae_inference', filepath, num_frames=15, net='genconvit', fp16=False)
        else:
            # Process image using the GenConViT model (you would need to adapt this)
            result = "Image prediction result"  # Implement image handling logic here
        
        return jsonify(result)
    
    return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    app.run(debug=True)
