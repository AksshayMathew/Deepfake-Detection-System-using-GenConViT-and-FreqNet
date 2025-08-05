# from flask import Flask, render_template, request, redirect, url_for
# import os
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# # Set up upload folder and allowed file types
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif', 'mp4', 'mov', 'avi'}

# # Check if the file extension is allowed
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# # Home route
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Upload route to handle the file and redirect to the result page
# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'media' not in request.files:
#         return redirect(request.url)
#     file = request.files['media']
    
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         # Redirect to the result page after uploading
#         return redirect(url_for('result', filename=filename))

# # Result route to display the uploaded video or image and the result (real or fake)
# @app.route('/result/<filename>')
# def result(filename):
#     # For now, we assume the result is always "real". You can later change this.
#     detection_result = "Real"
    
#     return render_template('result.html', filename=filename, detection_result=detection_result)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import shutil
import subprocess

app = Flask(__name__)

# Set up upload folder and allowed file types
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SAMPLE_PREDICTION_FOLDER'] = 'sample_prediction_data/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif', 'mp4', 'mov', 'avi'}

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Home route
@app.route('/')
def index():
    return render_template('index.html')
def parse_prediction_output(output: str):
    result = {}
    print(output)

    # Extract filename
    for line in output.splitlines():
        if "Loading..." in line:
            result['filename'] = line.split('\\')[-1].strip()

    # Extract prediction and confidence
    for line in output.splitlines():
        if line.startswith("Prediction:"):
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    result['confidence'] = float(parts[1])  # score after 'Prediction:'
                except ValueError:
                    result['confidence'] = None

                # Set label based on next word
                if "REAL" in parts[2].upper():
                    result['label'] = "REAL"
                elif "FAKE" in parts[2].upper():
                    result['label'] = "FAKE"
                else:
                    result['label'] = "Unknown"

    # Extract duration
    for line in output.splitlines():
        if line.startswith("---") and "seconds" in line:
            result['time'] = line.replace("---", "").replace("seconds", "").strip() + " seconds"

    return result

# def parse_prediction_output(output: str):
#     result = {}
#     print(output)
#     # Extract filename
#     for line in output.splitlines():
#         if "Loading..." in line:
#             result['filename'] = line.split('\\')[-1].strip()
    
#     # Extract prediction
#     for line in output.splitlines():
#         if line.startswith("Prediction:"):
#             pred_line = line.strip()
#             if "FAKE" in pred_line:
#                 result['label'] = "FAKE"
#             elif "REAL" in pred_line:
#                 result['label'] = "REAL"
    
#     # Extract duration
#     for line in output.splitlines():
#         if line.startswith("---") and "seconds" in line:
#             result['time'] = line.replace("---", "").replace("seconds", "").strip() + " seconds"
    
#     return result
# Upload route to handle the file and redirect to the result page
@app.route('/upload', methods=['POST'])
def upload():
    if 'media' not in request.files:
        return redirect(request.url)
    file = request.files['media']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        sample_data_path = os.path.join(app.config['SAMPLE_PREDICTION_FOLDER'], filename)

        # Save file to upload directory
        file.save(upload_path)

        # # Copy the uploaded file to the sample_prediction_data directory
        shutil.copy(upload_path, sample_data_path)

        # Run the GenConViT prediction script using subprocess
        try:
            # result = subprocess.run(
            #     ['python', 'GenConViT/prediction.py', '--p', app.config['SAMPLE_PREDICTION_FOLDER'], '--e', '--v', '--f', '10'],
            #     capture_output=True, text=True, check=True
            # )
            python_path = os.path.join(os.getcwd(), 'venv39', 'Scripts', 'python.exe')
            result=subprocess.run([
                python_path,
                'prediction.py',
                '--p', app.config['SAMPLE_PREDICTION_FOLDER'],
                '--e', 'genconvit_ed_inference',
                '--v', 'genconvit_vae_inference',
                '--f', '10'
            ], capture_output=True, text=True, check=True)

            output = parse_prediction_output(result.stdout)
            print(output)  # Print the output for debugging
            # You can customize the logic here depending on how the prediction output is formatted
            if 'FAKE' in output.get('label', '').upper():
                prediction = 'Fake'
            elif 'REAL' in output.get('label', '').upper():
                prediction = 'Real'
            else:
                prediction = 'Unknown'
        except subprocess.CalledProcessError as e:
            prediction = f"Error during prediction: {e.stderr}"
        finally:
            # ✅ Delete the uploaded file after prediction
            if os.path.exists(sample_data_path):


                
                os.remove(sample_data_path)

        # Redirect to the result page
        return redirect(url_for('result', filename=filename, prediction=prediction))

# Result route to display the uploaded video or image and the result (real or fake)
@app.route('/result/<filename>')
def result(filename):
    prediction = request.args.get('prediction', 'Unknown')
    return render_template('result.html', filename=filename, detection_result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
