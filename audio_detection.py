import torch
import librosa
import numpy as np
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
from moviepy.editor import VideoFileClip
import os

# 1. Configuration
MODEL_NAME = "MelodyMachine/Deepfake-audio-detection-V2"
# Alternative model if the above is too heavy or unavailable:
# MODEL_NAME = "mo-thecreator/Deepfake-audio-detection"

def load_audio(file_path, target_sr=16000):
    """
    Loads audio from a file (audio or video).
    If it's a video, it extracts the audio first.
    """
    ext = os.path.splitext(file_path)[1].lower()

    # If it's a video file, extract audio using moviepy
    if ext in ['.mp4', '.avi', '.mov', '.mkv']:
        print(f"Extracting audio from video: {file_path}...")
        try:
            video = VideoFileClip(file_path)
            # Extract audio to a temporary file
            temp_audio = "temp_audio.wav"
            video.audio.write_audiofile(temp_audio, logger=None)
            video.close()
            # Load the temporary audio file
            audio, sr = librosa.load(temp_audio, sr=target_sr)
            os.remove(temp_audio) # Clean up
            return audio, sr
        except Exception as e:
            print(f"Error extracting audio from video: {e}")
            return None, None

    # If it's widely supported audio format
    elif ext in ['.wav', '.mp3', '.flac', '.m4a']:
        print(f"Loading audio file: {file_path}...")
        try:
            audio, sr = librosa.load(file_path, sr=target_sr)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    else:
        print(f"Unsupported file format: {ext}")
        return None, None

def predict_deepfake(file_path):
    # 2. Load Model and Feature Extractor
    print("Loading model...")
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Failed to load model from Hugging Face: {e}")
        return

    # 3. Preprocess Audio
    audio, sr = load_audio(file_path)
    if audio is None:
        return

    # Ensure audio is 16kHz (model requirement)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # The model expects a batch of inputs, so we process the audio
    # Truncate/Pad to a reasonable length if necessary, but Wav2Vec2 can handle varying lengths.
    # For very long files, you might want to chunk it, but here we take the first 10 seconds for speed.
    max_duration = 10 # seconds
    if len(audio) > 16000 * max_duration:
        audio = audio[:16000 * max_duration]

    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    # 4. Inference
    with torch.no_grad():
        logits = model(**inputs).logits

    # 5. Interpret Results
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    predicted_label = model.config.id2label[predicted_class_id]

    # Calculate confidence score (softmax)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidence = probs[0][predicted_class_id].item()

    print("\n" + "="*30)
    print(f"RESULT: {predicted_label.upper()}")
    print(f"Confidence: {confidence:.2%}")
    print("="*30 + "\n")

    # Print raw probabilities for debugging
    for id, label in model.config.id2label.items():
        print(f"{label}: {probs[0][id].item():.4f}")

if __name__ == "__main__":
    # REPLACE THIS WITH YOUR FILE PATH
    file_to_test = "real.mp4"

    if os.path.exists(file_to_test):
        predict_deepfake(file_to_test)
    else:
        print("File not found. Please check the path.")