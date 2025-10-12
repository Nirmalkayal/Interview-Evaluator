from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from pydub import AudioSegment
import cv2
from deepface import DeepFace
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import whisper
# --- NEW: IMPORTS FOR VOCAL ANALYSIS ---
import librosa
import numpy as np

# --- LOAD ALL AI MODELS ON STARTUP ---
print("Loading AI models, this may take a moment...")
relevance_model = SentenceTransformer('all-mpnet-base-v2')
whisper_model = whisper.load_model("base")
print("All models loaded successfully.")
# ------------------------------------

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')

app = Flask(__name__)
CORS(app)

if not os.path.exists("uploads"):
    os.makedirs("uploads")

FILLER_WORDS = [
    "um", "uh", "ah", "er", "like", "so", "you know", 
    "basically", "actually", "i mean", "right"
]

# --- NEW: VOCAL ANALYSIS FUNCTION ---
def analyze_vocal_characteristics(audio_path):
    try:
        y, sr = librosa.load(audio_path)
        
        # 1. Calculate Speaking Pace (WPM)
        duration_seconds = librosa.get_duration(y=y, sr=sr)
        # We'll get the word count from the transcription later
        
        # 2. Calculate Pitch Variation (Monotony Detection)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        # Get the pitches for frames with high confidence
        confident_pitches = pitches[magnitudes > np.median(magnitudes)]
        if len(confident_pitches) > 0:
            pitch_variation = np.std(confident_pitches[confident_pitches > 0])
        else:
            pitch_variation = 0

        return {
            "duration_seconds": duration_seconds,
            "pitch_variation": pitch_variation
        }
    except Exception as e:
        print(f"Error in vocal analysis: {e}")
        return { "duration_seconds": 0, "pitch_variation": 0 }
# ------------------------------------

def calculate_relevance_score(text, keywords):
    # This function is unchanged
    if not keywords or not text or text.startswith("Could not understand"): return 0
    try:
        embedding_text = relevance_model.encode(text, convert_to_tensor=True)
        embedding_keywords = relevance_model.encode(keywords, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embedding_text, embedding_keywords)
        relevance_score = (cosine_scores[0][0].item() + 1) / 2 * 100
        return max(0, min(100, relevance_score))
    except: return 0

def count_filler_words(text):
    # This function is unchanged
    words = text.lower().split()
    return sum(1 for word in words if word in FILLER_WORDS)

def analyze_facial_expressions(video_path):
    # This function is unchanged
    try:
        # ... (code for facial analysis is the same)
        emotions = []
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_count % int(frame_rate) == 0:
                try:
                    analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    if isinstance(analysis, list) and len(analysis) > 0:
                        emotions.append(analysis[0]['dominant_emotion'])
                except: pass
            frame_count += 1
        cap.release()
        if not emotions: return "Neutral"
        return Counter(emotions).most_common(1)[0][0].capitalize()
    except:
        return "Neutral"

# --- UPDATED: SCORING ENGINE ---
def calculate_scores(text, facial_emotion, sentiment_score, keywords, vocal_analysis):
    filler_count = count_filler_words(text)
    clarity_score = max(0, 100 - (filler_count * 5))
    
    confidence_score = 60
    if facial_emotion.lower() == 'happy': confidence_score += 20
    elif facial_emotion.lower() == 'neutral': confidence_score += 10
    else: confidence_score -= 25
    if sentiment_score > 0.5: confidence_score += 15
    elif sentiment_score < -0.4: confidence_score -= 20
    confidence_score = max(0, min(100, confidence_score))
    
    relevance_score = calculate_relevance_score(text, keywords)
    
    # NEW: Calculate Speaking Pace
    word_count = len(text.split())
    speaking_pace_wpm = 0
    if vocal_analysis['duration_seconds'] > 0:
        speaking_pace_wpm = int((word_count / vocal_analysis['duration_seconds']) * 60)

    # NEW: Calculate Engagement Score based on pitch
    # (This is a simple model; a higher std dev suggests more variation)
    engagement_score = min(100, int(vocal_analysis['pitch_variation'] * 10))
    
    return {
        "confidence_score": int(confidence_score),
        "clarity_score": int(clarity_score),
        "relevance_score": int(relevance_score),
        "engagement_score": int(engagement_score),
        "filler_word_count": filler_count,
        "speaking_pace_wpm": speaking_pace_wpm
    }

@app.route('/evaluate', methods=['POST'])
def evaluate_interview():
    if 'audio' not in request.files or 'video' not in request.files:
        return jsonify({"error": "Audio or video file not found"}), 400
    
    keywords = request.form.get('keywords', '')
    audio_file = request.files['audio']
    video_file = request.files['video']

    original_audio_path = os.path.join("uploads", "interview_audio.webm")
    video_path = os.path.join("uploads", "interview_video.webm")
    audio_file.save(original_audio_path)
    video_file.save(video_path)

    try:
        audio = AudioSegment.from_file(original_audio_path)
        wav_path = os.path.join("uploads", "interview_audio.wav")
        audio.export(wav_path, format="wav")
    except Exception as e:
        return jsonify({"error": f"Failed to process audio file: {e}"}), 500

    # --- ANALYSIS PIPELINE ---
    # 1. High-Accuracy Transcription with Whisper
    try:
        result = whisper_model.transcribe(wav_path, fp16=False)
        text = result.get('text', "Transcription failed.")
    except Exception as e:
        text = f"Could not understand the audio due to an error: {e}"

    # 2. NEW: Vocal Characteristics Analysis
    vocal_analysis_results = analyze_vocal_characteristics(wav_path)

    # 3. Analyze Sentiment & Facial Expressions
    sentiment_score = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
    facial_emotion = analyze_facial_expressions(video_path)
    
    # 4. Get All Scores from the Engine
    scores = calculate_scores(text, facial_emotion, sentiment_score, keywords, vocal_analysis_results)
    
    # 5. Build Final Report
    report = {
        "transcription": text,
        "facial_emotion": facial_emotion,
        "confidence_score": scores['confidence_score'],
        "clarity_score": scores['clarity_score'],
        "relevance_score": scores['relevance_score'],
        "engagement_score": scores['engagement_score'],
        "filler_word_count": scores['filler_word_count'],
        "speaking_pace_wpm": scores['speaking_pace_wpm'],
        "feedback": "Vocal analysis complete."
    }

    return jsonify(report)

if __name__ == '__main__':
    app.run(debug=True, port=5001)