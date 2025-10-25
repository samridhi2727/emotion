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

# --- Global Style: Gradient, theme, hide Streamlit menu/header/footer ---
st.markdown("""
<style>
body, .stApp {
    background: linear-gradient(135deg,
        #ffb6c1 0%,
        #a8d0e6 25%,
        #fff9c4 50%,
        #d1c4e9 75%,
        #ffcc80 100%
    );
    color: #3d2a4d !important;
    font-family: 'Quicksand', sans-serif;
}
section[data-testid="stSidebar"] {
    background: rgba(255, 182, 193, 0.8);
    color: #3d2a4d !important;
    border-top-right-radius: 20px;
    border-bottom-right-radius: 20px;
}
.st-bf, label, .css-1n76uvr, .css-10trblm {
    color: #7b4f85 !important;
    font-weight: 600 !important;
    font-family: 'Quicksand', sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    color: #ff4081 !important;
    font-family: 'Pacifico', cursive;
    text-shadow:
        0 0 5px #ffb6c1,
        0 0 10px #7b4f85,
        0 0 15px #ffcc80;
    font-weight: 900 !important;
    letter-spacing: 1.1px;
    margin-bottom: 0.3em;
}
.stSlider > div[data-baseweb="slider"] > div {
    background: linear-gradient(90deg,
        #ffb6c1 0%,
        #a8d0e6 25%,
        #fff9c4 50%,
        #d1c4e9 75%,
        #ffcc80 100%
    ) !important;
    border-radius: 12px !important;
    height: 12px !important;
}
.stSlider > div[data-baseweb="slider"] > div > div[role="slider"] {
    border: 2.5px solid #ff4081 !important;
    background-color: #ffffff !important;
    box-shadow: 0 0 12px 3px #ff4081aa !important;
    width: 28px !important;
    height: 28px !important;
    cursor: pointer;
    margin-top: -9px !important;
}
.stSlider label,
.css-1p0e3fi {
    color: #7b4f85 !important;
    font-weight: 700 !important;
    font-family: 'Quicksand', sans-serif;
    text-shadow: 0 0 5px #d1c4e9;
}
.css-1hwfws3 {
    background-color: #ffb6c1 !important;
    color: #3d2a4d !important;
    border-radius: 10px !important;
    font-weight: 700;
}
.css-79elbk {
    background-color: #f3e5f5 !important;
    color: #7b4f85 !important;
}
.stButton > button {
    background-color: #ff4081 !important;
    color: white !important;
    font-family: 'Quicksand', sans-serif;
    font-weight: 700 !important;
    border-radius: 1em !important;
    box-shadow: 0 4px 15px #ffb6c133 !important;
    transition: background-color 0.3s ease;
}
.stButton > button:hover {
    background-color: #e040fb !important;
}
.stAlert-info, .stAlert-success, .stAlert-warning {
    background: rgba(255, 203, 112, 0.18) !important;
    border-left: 5px solid #ffcc80 !important;
    color: #3d2a4d !important;
    font-family: 'Quicksand', sans-serif;
    font-weight: 600;
    box-shadow: none !important;
}
.stAlert-info svg, .stAlert-success svg, .stAlert-warning svg {
    color: #ffcc80 !important;
}
.stCameraInput label {
    color: #ff4081 !important;
    font-family: 'Quicksand', sans-serif;
    font-weight: 600 !important;
}
::-webkit-scrollbar {
  width: 8px;
}
::-webkit-scrollbar-thumb {
  background: #ff4081aa;
  border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
  background: #ff4081dd;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.css-18e3th9 {padding-top: 0rem;}   
.css-1d391kg {padding-top: 0rem;}
</style>
""", unsafe_allow_html=True)

# --- Create SQLite user table for login/signup ---
create_user_table()

def login_signup_box():
    # Heading for login - no background box!
    st.markdown(
        "<h2 style='text-align:center; color:#ff4081; font-family:Roboto, cursive;'>Login Page for Emotion Detector</h2>",
        unsafe_allow_html=True
    )
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

# --- Main App: Sidebar and emotion detector ---
st.set_page_config(page_title="Live Emotion Detector", layout="centered")
st.sidebar.title("🎛️ Filters")

sensitivity = st.sidebar.slider("Detection Sensitivity", 0.01, 0.5, 0.3)
st.sidebar.subheader("💗 Lower the sensitivity better is the result!")
selected_emotions = st.sidebar.multiselect(
    "Select Emotions to Display",
    ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"],
    default=["Happy", "Sad", "Angry"]
)
frame_rate = st.sidebar.slider("Frame Rate (FPS)", 1, 30, 10)

st.title("🧠 Real-Time Emotion Detector")

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

try:
    model = load_model("model/model.keras", compile=False)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

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
                    f"Hello {user_name}! you look {emotion} today with a confidence of {prob:.0f} percent. I want to suggest you that: {suggestion}. "
                    f"Here is a joke to lift up your mood: {joke} hahahah. hope you Have a great day ahead"
                )
                audio_fp = speak_text(speak_str)
                st.audio(audio_fp, format="audio/mp3")
        else:
            st.warning("No emotion detected with the current settings. Change Your Emotion selection to get the appropriate results!")
