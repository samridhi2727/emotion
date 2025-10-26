import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from keras import backend as K
from PIL import Image
from utils.emotion_utils import detect_faces_and_emotions
from utils.suggestions import get_suggestion, get_joke
from utils.auth_utils import create_user_table, add_user, authenticate_user, user_exists
from gtts import gTTS
import io

from tensorflow import keras 
# --- Initialize required session variables ---
if "emotion_history" not in st.session_state:
    st.session_state["emotion_history"] = []
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None


import tensorflow as tf
from tensorflow import keras
import streamlit as st

# Safe model loading with proper deserialization handling
@st.cache_resource
def load_emotion_model():
    try:
        # Option 1 – Try loading with TensorFlow’s legacy-based deserializer
        model = keras.models.load_model("model/model.keras", compile=False)
        st.success("Emotion model loaded successfully ✅")
        return model
    except Exception as e:
        st.warning("Legacy Keras deserialization failed. Attempting fallback...")

        try:
            # Option 2 – Try using the H5 loader (for older model files)
            model = keras.models.load_model("model/model.h5", compile=False)
            st.success("Model loaded via H5 legacy loader ✅")
            return model
        except Exception as e2:
            st.error(f"❌ Error loading model: {e2}")
            return None

# Initialize model
model = load_emotion_model()

if model is None:
    st.stop()  # Stop app before model-dependent code executes

st.markdown("""
    <style>
    /* --- Global Header Styling --- */
    .vibe-header {
        background: linear-gradient(90deg,
            #f8a5c2 0%,      /* pastel pink */
            #fdd6bd 20%,     /* light orange */
            #fff5b7 40%,     /* soft yellow */
            #c7a9e9 70%,     /* light purple */
            #a1c4fd 100%);   /* sky blue */
        color: #2f3e46;
        font-family: 'Poppins', sans-serif;
        text-align: center;
        padding: 1.2em 0.5em;
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
    }

    .vibe-title {
        font-weight: 800;
        font-size: 32px;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 0;
    }

    .vibe-subtitle {
        font-weight: 500;
        font-size: 18px;     /* smaller than main title */
        margin-top: 4px;
        color: #2b6777;
        opacity: 0.9;
    }
    </style>

    <div class="vibe-header">
        <div class="vibe-title">🌈 VIBE CHECK AI</div>
        <div class="vibe-subtitle">By Samridhi</div>
    </div>
""", unsafe_allow_html=True)


# --- Global Style: Gradient, theme, hide Streamlit menu/header/footer ---
st.markdown("""
<style>
/* --- Page Background and Font --- */
body, .stApp {
  background: linear-gradient(135deg,
    #f8a5c2 0%,      /* pink */
    #fdd6bd 20%,     /* light orange */
    #fff5b7 40%,     /* soft yellow */
    #c7a9e9 60%,     /* light purple */
    #a1c4fd 80%,     /* sky blue */
    #c2e9fb 100%);   /* pastel blend */
  color: #2f3e46;
  font-family: 'Poppins', sans-serif;
}

/* --- Sidebar --- */
section[data-testid="stSidebar"] {
  background: linear-gradient(135deg, #e8fdf5 0%, #fdf2ff 100%);
  border-top-right-radius: 18px;
  border-bottom-right-radius: 18px;
  color: #1b4332;
  box-shadow: 0 0 15px rgba(100, 100, 100, 0.1);
}

/* --- Titles and Headings --- */
h1, h2, h3, h4, h5, h6 {
  color: #2b6777 !important;
  font-weight: 700 !important;
  letter-spacing: 0.8px;
  text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.5);
}

/* --- Buttons --- */
.stButton > button {
  background: linear-gradient(90deg, #ffafbd, #ffc3a0);
  color: white !important;
  font-weight: 700 !important;
  border-radius: 0.8em;
  box-shadow: 0 3px 10px rgba(255, 175, 189, 0.4);
  transition: all 0.3s ease;
}
.stButton > button:hover {
  transform: scale(1.05);
  background: linear-gradient(90deg, #84fab0, #8fd3f4);
}

/* --- Inputs & Labels --- */
label, .stTextInput, .stPasswordInput {
  color: #1b4332 !important;
  font-weight: 600 !important;
}

/* --- Sliders --- */
.stSlider > div[data-baseweb="slider"] > div {
  background: linear-gradient(90deg, #b7e4c7 0%, #95d5b2 100%) !important;
  border-radius: 10px;
}
.stSlider > div[data-baseweb="slider"] > div > div[role="slider"] {
  border: 2px solid #2b6777 !important;
  background-color: #ffffff !important;
  box-shadow: 0 0 10px 2px rgba(43, 103, 119, 0.3);
}

/* --- Alerts --- */
.stAlert-info, .stAlert-success, .stAlert-warning {
  background: rgba(255, 255, 255, 0.7);
  border-left: 5px solid #ffd6e3;
  font-weight: 600;
  color: #2f3e46;
}
/* --- Logout Button Styling --- */
div[data-testid="stSidebar"] button[kind="secondary"] {
  background: linear-gradient(90deg, #f8a5c2 0%, #fdd6bd 33%, #fff5b7 66%, #a1c4fd 100%) !important;
  color: white !important;
  font-weight: 700 !important;
  border-radius: 10px !important;
  padding: 0.6em 1em !important;
  border: none !important;
  box-shadow: 0 3px 10px rgba(255, 175, 189, 0.4);
  transition: all 0.3s ease-in-out;
}
div[data-testid="stSidebar"] button[kind="secondary"]:hover {
  transform: scale(1.05);
  background: linear-gradient(90deg, #84fab0, #8fd3f4) !important;
}


/* --- Hide Streamlit Menus --- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)



# --- Create SQLite user table for login/signup ---
create_user_table()

def login_signup_box():
    # Heading for login - no background box!
    st.markdown("<h2 style='text-align:center; color:#2b6777;'>Emotion Detection Portal</h2>", unsafe_allow_html=True)
    mode = st.radio("", ["Login", "Sign Up"], horizontal=True, label_visibility="collapsed")
    username = st.text_input("Username", max_chars=30)
    password = st.text_input("Password", type="password", max_chars=30)
    submitted = st.button(mode)

    msg = None
    success = False

    if submitted:
        if mode == "Sign Up":
            if user_exists(username):
                msg = "Username already exists. Please choose another."
            elif not username or not password:
                msg = "Username and password cannot be empty."
            else:
                add_user(username, password)
                msg = "Signed up successfully! Please login."
                success = True
        elif mode == "Login":
            if authenticate_user(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                msg = "Invalid username or password."

    if msg:
        col_msg = "⚠️" if not success else "✅"
        st.markdown(
            f"<div style='color:#7b4f85; font-weight:600; margin-top:10px;'>{col_msg} {msg}</div>",
            unsafe_allow_html=True
        )

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_signup_box()
    st.stop()

def logout():
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.rerun()


# --- Main App: Sidebar and emotion detector ---
st.set_page_config(page_title="Live Emotion Detector", layout="centered")
# --- Logout button styled to match theme ---
st.sidebar.markdown("<h3 style='color:#2b6777;'>👤 Logged in as: {}</h3>".format(st.session_state["username"]), unsafe_allow_html=True)

logout_clicked = st.sidebar.button("🚪 Logout")

if logout_clicked:
    logout()

st.sidebar.title("🎛️ Filters")

sensitivity = st.sidebar.slider("Detection Sensitivity", 0.01, 0.5, 0.3)
st.sidebar.subheader("Lower the sensitivity better is the result!")
selected_emotions = st.sidebar.multiselect(
    "Select Emotions to Display",
    ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"],
    default=["Happy", "Sad", "Angry"]
)
frame_rate = st.sidebar.slider("Frame Rate (FPS)", 1, 30, 10)



user_name = st.session_state.get("username", "User")
st.markdown(
    f"""
    <div style='
        margin-top: -10px;
        background: rgba(255, 255, 255, 0.7);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.1);
        font-size: 17px;
        color: #2b6777;
        font-family: "Poppins", sans-serif;
    '>
        👋 Hello <b>{user_name}</b>! Welcome to <b>Vibe Check AI</b> — your real-time emotion detection portal.<br>
        We combine computer vision and AI to analyze facial expressions.<br>
        Take a live photo to see which emotion you are experiencing right now.<br>
        Let's explore the science of emotions — one smile at a time!
    </div>
    """, unsafe_allow_html=True
)

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

st.subheader("📷 Capture your emotion!")

image_data = st.camera_input("Take a picture")

def speak_text(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

if image_data is not None:
    img = Image.open(image_data)

    results = detect_faces_and_emotions(
        img,
        face_cascade,
        model,
        emotion_labels,
        threshold=sensitivity,
        selected_emotions=selected_emotions,
    )

    if results:
        combined_results = {}
        for emotion, prob in results:
            if emotion in combined_results:
                combined_results[emotion] = max(combined_results[emotion], prob)
            else:
                combined_results[emotion] = prob

        filtered_results = [
            (e, p) for e, p in combined_results.items()
            if (not selected_emotions or e in selected_emotions) and p > 0
        ]
        filtered_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)

        if filtered_results:
            st.success("Emotion(s) Detected ✅")
            for emotion, prob in filtered_results:
                suggestion = get_suggestion(emotion)
                joke = get_joke(emotion)
                st.markdown(f"**{emotion}**: {prob:.2f}%")
                st.info(f"💡 Suggestion: {suggestion}")
                st.write(f"😂 Joke: {joke}")
                user_name = st.session_state.get("username", "dear")
                speak_str = (
                    f"Hie {user_name}! you look {emotion} today with a confidence of {prob:.0f} percent. I want to suggest you that: {suggestion}. "
                    f"Here is a joke to lift up your mood: {joke} hahahaheheheh. hope you Have a great day ahead"
                )
                audio_fp = speak_text(speak_str)
                st.audio(audio_fp, format="audio/mp3")
        else:
            st.warning("No emotion detected with the current settings. Change Your Emotion selection to get the appropriate results!")



# ---- LIVE EMOTION SUMMARY (Camera-Based) ----
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- FIX CRITICAL INITIALIZATION (DO THIS NEAR THE TOP OF YOUR APP) ---
if "emotion_history" not in st.session_state:
    st.session_state["emotion_history"] = []

st.markdown("<hr>", unsafe_allow_html=True)
st.header("📊 Live Emotion Summary Dashboard")
st.caption("Visual representation of your detected live facial emotions — updated dynamically after each capture.")

# --- STORE LATEST EMOTION ---
if image_data is not None and 'filtered_results' in locals() and filtered_results:
    top_emotion = filtered_results[0][0]
    st.session_state["emotion_history"].append(top_emotion)

if len(st.session_state["emotion_history"]) > 0:
    emotion_df = pd.DataFrame({
        "Capture": list(range(1, len(st.session_state["emotion_history"]) + 1)),
        "Emotion": st.session_state["emotion_history"]
    })
    freq = emotion_df["Emotion"].value_counts()

    # --- CONFIGURE STYLE ONCE ---
    sns.set_style("whitegrid")

    # Create pastel color scheme once
    pastel_colors = ["#f8a5c2", "#fdd6bd", "#fff5b7", "#a1c4fd", "#c7a9e9", "#b7e4c7"]

    # --- LINE PLOT (Emotion over captures) ---
    fig1, ax1 = plt.subplots(figsize=(5.5, 3))
    fig1.patch.set_alpha(0)
    ax1.set_facecolor((0, 0, 0, 0))

    sns.lineplot(data=emotion_df, x="Capture", y="Emotion", hue="Emotion",
                 marker="o", palette=pastel_colors[:len(emotion_df["Emotion"].unique())], ax=ax1)

    ax1.set_title("Emotion Trend Over Session", color="#2b6777", fontsize=13, weight="bold", pad=8)
    ax1.set_xlabel("Capture #", fontsize=11)
    ax1.set_ylabel("Emotion Label", fontsize=11)
    ax1.grid(alpha=0.2, linestyle="--")
    sns.despine()
    st.pyplot(fig1, transparent=True, clear_figure=True)

    # --- BAR PLOT (Emotion frequency) ---
    fig2, ax2 = plt.subplots(figsize=(5, 2.8))
    fig2.patch.set_alpha(0)
    ax2.set_facecolor((0, 0, 0, 0))

    sns.barplot(x=freq.index, y=freq.values, palette=pastel_colors[:len(freq)], ax=ax2)
    ax2.set_title("Emotion Occurrence Summary", color="#2b6777", fontsize=12, weight="semibold")
    ax2.set_xlabel("Emotion", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.grid(alpha=0.15, linestyle="--")
    sns.despine()
    st.pyplot(fig2, transparent=True, clear_figure=True)

    # --- Display Insights ---
    dominant = freq.idxmax()
    st.success(f"Dominant Emotion Detected: **{dominant}** 🧠")

    insights = {
        "Happy": "Consistently positive mood — reflects engagement and optimism.",
        "Sad": "Emotional fatigue observed — take short breaks to refresh your mindset.",
        "Neutral": "Steady and balanced mood — ideal for sustained focus.",
        "Angry": "Elevated stress — practice calm breathing or step away briefly.",
        "Fear": "Anxious signals detected — try confidence-building tasks.",
        "Surprise": "Spike in excitement detected — spontaneous emotional reaction.",
        "Disgust": "Possible aversion or discomfort — maybe due to content or environment.",
    }
    st.info(f"🧩 Insight: {insights.get(dominant, 'Maintain awareness of your current emotional state.')}")
else:
    st.warning("No emotions recorded yet. Capture an image to start building the emotion summary.")

# ---- EMOTION INSIGHTS ----
st.markdown("<hr>", unsafe_allow_html=True)
st.header("🧩 Live Emotion Insights")
st.caption("These insights are generated from your most recent camera-based emotion detection.")

insight_map = {
    "Happy": "High positivity and mental energy — ideal state for collaboration and focus.",
    "Sad": "Low mood detected — consider activities like music or fresh air to refresh yourself.",
    "Angry": "Elevated stress or frustration — deep breathing and mindfulness may help.",
    "Neutral": "Calm and balanced state — great for maintaining steady focus.",
    "Surprise": "Unexpected reaction captured — spontaneity brings joy, embrace it positively.",
    "Fear": "Slight anxiety noted — ground yourself through short breaks.",
    "Disgust": "Avoidance or discomfort emotion detected — shift focus to pleasant experiences.",
}

if st.session_state["emotion_history"]:
    current_emotion = st.session_state["emotion_history"][-1]
    st.success(f"Current Detected Emotion: **{current_emotion}**")
    st.info(f"📘 Insight: {insight_map.get(current_emotion, 'Stay emotionally aware and balanced!')}")
else:
    st.info("No emotion captured yet — take a live photo to generate insights.")

# ---- TEXT EMOTION DETECTOR ----
# --- Advanced Text Emotion Detector using Hugging Face ---
st.markdown("<hr>", unsafe_allow_html=True)
st.header("💬 Text Emotion Detector ")
st.caption("This module uses a fine-tuned BERT model to classify text into multiple emotional categories such as joy, sadness, anger, love, fear, and surprise. It provides precise emotion and confidence scores using Hugging Face Transformers.")

from transformers import pipeline
import torch

@st.cache_resource
def load_emotion_analyzer():
    try:
        # You can choose: "bhadresh-savani/distilbert-base-uncased-emotion"
        # or use "j-hartmann/emotion-english-distilroberta-base" for deeper results
        return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=False)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

emotion_analyzer = load_emotion_analyzer()

user_input = st.text_area("Type or paste text to analyze emotional tone:")

if st.button("🔍 Analyze Emotion"):
    if user_input.strip():
        try:
            result = emotion_analyzer(user_input)[0]
            label = result["label"].capitalize()
            score = result["score"] * 100  # Convert to %
            
            emoji_map = {
                "Joy": "😄",
                "Sadness": "😢",
                "Anger": "😠",
                "Fear": "😨",
                "Love": "❤️",
                "Surprise": "😲",
            }
            emoji = emoji_map.get(label, "🙂")
            
            st.success(f"Detected Emotion: **{label} {emoji}**")
            st.write(f"Confidence: **{score:.2f}%**")

            # Deeper insight interpretation for academic context
            insights = {
                "Joy": "Expresses happiness and contentment — strong positivity detected.",
                "Sadness": "Conveys low mood or distress — may indicate emotional fatigue.",
                "Anger": "Associated with frustration or injustice — emotional arousal detected.",
                "Fear": "Indicates anxiety or threat perception — cautious emotional state.",
                "Love": "Reflects strong affection or care — socially bonded emotion.",
                "Surprise": "Suggests unexpected content — cognitive spontaneity evident.",
            }
            st.info(f"🧠 NLP Insight: {insights.get(label, 'Balanced emotional tone detected.')}")
        
        except Exception as e:
            st.error(f"Error analyzing emotion: {e}")
    else:
        st.warning("Please enter some text to analyze its emotion.")

st.markdown("""
<hr style="border: 1px solid #ddd;">
<center>
    <p style='color:#2b6777; font-size:15px;'>
    © 2025 Vibe Check AI | Developed by <b>Samridhi</b> as a PG Project
    </p>
</center>
""", unsafe_allow_html=True)
