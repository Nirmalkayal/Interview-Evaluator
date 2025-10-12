from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import nltk
from pydub import AudioSegment
import cv2
from deepface import DeepFace
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import whisper
import librosa
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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

# --- NEW: DYNAMIC FEEDBACK GENERATOR ---
def generate_dynamic_feedback(scores):
    feedback_tips = []

    # 1. Feedback on Speaking Pace
    pace = scores['speaking_pace_wpm']
    if pace < 130:
        feedback_tips.append({"type": "improvement", "tip": f"Your speaking pace of {pace} WPM is a bit slow. Try to speak more fluidly."})
    elif pace > 170:
        feedback_tips.append({"type": "improvement", "tip": f"Your speaking pace of {pace} WPM is quite fast. Remember to pause to let your key points sink in."})
    else:
        feedback_tips.append({"type": "positive", "tip": f"Your speaking pace of {pace} WPM is perfect. It's clear and easy for the listener to follow."})

    # 2. Feedback on Clarity (Filler Words)
    fillers = scores['filler_word_count']
    if fillers == 0:
        feedback_tips.append({"type": "positive", "tip": "Excellent clarity! You avoided using any filler words, which makes your speech sound confident and polished."})
    elif fillers > 3:
        feedback_tips.append({"type": "improvement", "tip": f"You used {fillers} filler words. To improve clarity, try pausing silently instead of using words like 'um' or 'like'."})

    # 3. Feedback on Engagement (Vocal Variety)
    engagement = scores['engagement_score']
    if engagement < 40:
        feedback_tips.append({"type": "improvement", "tip": "Your vocal tone was a bit monotone. Try to vary your pitch and energy to sound more engaging and dynamic."})
    else:
        feedback_tips.append({"type": "positive", "tip": "Great vocal engagement! Your varied tone helps keep the listener interested and conveys enthusiasm."})

    # 4. Feedback on Relevance
    relevance = scores['relevance_score']
    if relevance < 60:
        feedback_tips.append({"type": "improvement", "tip": "A good start, but try to connect your answer more directly to the keywords from the job description to show you're a strong fit."})
    else:
        feedback_tips.append({"type": "positive", "tip": "Fantastic answer! You did a great job of aligning your experience with the key skills for the role."})

    return feedback_tips
# ---------------------------------------

# (All other analysis functions like analyze_vocal_characteristics, calculate_relevance_score, etc. remain the same)
def analyze_vocal_characteristics(audio_path):
    try:
        y, sr = librosa.load(audio_path)
        duration_seconds = librosa.get_duration(y=y, sr=sr)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        confident_pitches = pitches[magnitudes > np.median(magnitudes)]
        pitch_variation = np.std(confident_pitches[confident_pitches > 0]) if len(confident_pitches) > 0 else 0
        return {"duration_seconds": duration_seconds, "pitch_variation": pitch_variation}
    except Exception as e:
        print(f"Error in vocal analysis: {e}")
        return {"duration_seconds": 0, "pitch_variation": 0}

def calculate_relevance_score(text, keywords):
    if not keywords or not text or text.startswith("Could not understand"): return 0
    try:
        embedding_text = relevance_model.encode(text, convert_to_tensor=True)
        embedding_keywords = relevance_model.encode(keywords, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embedding_text, embedding_keywords)
        relevance_score = (cosine_scores[0][0].item() + 1) / 2 * 100
        return max(0, min(100, relevance_score))
    except: return 0

def count_filler_words(text):
    words = text.lower().split()
    return sum(1 for word in words if word in FILLER_WORDS)

def analyze_facial_expressions(video_path):
    try:
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
    except: return "Neutral"

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
    word_count = len(text.split())
    speaking_pace_wpm = int((word_count / vocal_analysis['duration_seconds']) * 60) if vocal_analysis['duration_seconds'] > 0 else 0
    engagement_score = min(100, int(vocal_analysis['pitch_variation'] * 10))
    return {"confidence_score": confidence_score, "clarity_score": clarity_score, "relevance_score": relevance_score, "engagement_score": engagement_score, "filler_word_count": filler_count, "speaking_pace_wpm": speaking_pace_wpm}

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

    # Analysis Pipeline
    try:
        result = whisper_model.transcribe(wav_path, fp16=False)
        text = result.get('text', "Transcription failed.")
    except Exception as e:
        text = f"Could not understand the audio: {e}"

    vocal_analysis_results = analyze_vocal_characteristics(wav_path)
    sentiment_score = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
    facial_emotion = analyze_facial_expressions(video_path)
    scores = calculate_scores(text, facial_emotion, sentiment_score, keywords, vocal_analysis_results)
    
    # --- GET DYNAMIC FEEDBACK ---
    feedback_tips = generate_dynamic_feedback(scores)

    # Build Final Report
    report = {
        "transcription": text,
        "facial_emotion": facial_emotion,
        "scores": scores, # Send all scores nested in one object
        "feedback_tips": feedback_tips # Send the list of tips
    }

    return jsonify(report)

if __name__ == '__main__':
    app.run(debug=True, port=5001)