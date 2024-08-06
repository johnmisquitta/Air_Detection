from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import librosa
import tempfile
import cv2
import os
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plots
import matplotlib.pyplot as plt
# Load the model
model = tf.keras.models.load_model('./model.h5')

# Initialize FastAPI and Flask apps
app = FastAPI()
app = Flask(__name__)
CORS(app)  # Enable CORS for the entire Flask app

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
IMAGE_SIZE = (128, 128)  # Size of the spectrogram images

def ensure_one_second_clip(y, sr=SAMPLE_RATE):
    target_length = sr  # 1 second of audio
    if len(y) > target_length:
        y = y[:target_length]  # Trim to 1 second
    elif len(y) < target_length:
        padding = target_length - len(y)
        y = np.pad(y, (0, padding), mode='constant')  # Pad with zeros
    return y

def extract_spectrogram(y, sr=SAMPLE_RATE):
    y = ensure_one_second_clip(y, sr)  # Ensure the audio is 1 second long
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return librosa.power_to_db(S, ref=np.max)

def save_spectrogram(S, output_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S, sr=SAMPLE_RATE, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type == 'audio/wav':
        return JSONResponse(content={"success": True}, status_code=200)
    return JSONResponse(content={"success": False, "error": "Invalid file type"}, status_code=400)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    temp_dir = "./Temp"
    temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_dir, delete=False)
    file_data = file.read()

    with open(temp_wav_file.name, 'wb') as wav_file:
        wav_file.write(file_data)

    y, sr = librosa.load(temp_wav_file.name, sr=None)
    S = extract_spectrogram(y)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image_file:
            temp_image_path = temp_image_file.name
    save_spectrogram(S, temp_image_path)


    img = tf.keras.preprocessing.image.load_img(temp_image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
    img_array = img_array / 255.0  # Normalize to [0,1]
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    confidence_score = np.max(predictions, axis=1)

    if predicted_class ==[0]:
        predected_result="Air Detected"
    else :
        predected_result="No Air Detected"
    


    os.remove(temp_image_path)
    return(str(f"Prediction = {predected_result} confidence_score = {round(confidence_score[0],2)}"))
    #     'predicted_class': int(predicted_class[0]),
    #     'confidence_score': float(confidence_score[0])
    # }))

if __name__ == '__main__':
    app.run(debug=True)
