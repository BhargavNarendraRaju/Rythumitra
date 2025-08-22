import streamlit as st
import requests
import speech_recognition as sr
import json
from datetime import datetime
from geopy.geocoders import Nominatim
from gtts import gTTS
import base64
import io
from streamlit_js_eval import streamlit_js_eval, get_geolocation
import pyaudio
import wave
import threading
import time
import os
from PIL import Image
import base64
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import TensorFlow with error handling
TENSORFLOW_AVAILABLE = False
PLANT_DISEASE_MODEL = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow imported successfully")
except ImportError as e:
    logger.warning(f"TensorFlow not available: {e}. Plant disease detection will be disabled.")
except Exception as e:
    logger.warning(f"Error loading TensorFlow: {e}. Plant disease detection will be disabled.")

# ----------------------------
# Configuration
# ----------------------------
MODEL_REGISTRY = {
    "ASA – Policy Specialist": {
        "role": "specialist-policy",
        "color": "#27AE60",
    },
    "ASB – Agronomy Specialist": {
        "role": "specialist-agronomy",
        "color": "#E67E22",
    },
    "ASC – Fact Checker": {
        "role": "specialist-crosscheck",
        "color": "#C0392B"
    }
}

ROUTER_OPTIONS = ["Auto (Router decides)"] + list(MODEL_REGISTRY.keys())
WEATHER_API_KEY = "b10a43e49ad59f27140d077c8f1a6bfd"  # Replace with your actual API key

# Plant disease detection model configuration (only if TensorFlow is available)
if TENSORFLOW_AVAILABLE:
    from huggingface_hub import hf_hub_download
    REPO_ID = "Bhargavnarendraraju/Plant_Disease_Detection_model"
    FILENAME = "plant_disease_efficientnetb4.h5"

    # Define PlantVillage classes
    CLASS_NAMES = [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry_(including_sour)___Powdery_mildew",
        "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Peach___Bacterial_spot",
        "Peach___healthy",
        "Pepper,_bell___Bacterial_spot",
        "Pepper,_bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch",
        "Strawberry___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]
else:
    # Fallback configuration if TensorFlow is not available
    CLASS_NAMES = []
    logger.warning("Plant disease detection disabled due to TensorFlow import error")

# ----------------------------
# Language detection
# ----------------------------
def detect_lang_from_text(text: str) -> str:
    """Detect language from text with Telugu priority"""
    t = (text or "").lower()
    if "telugu" in t or "తెలుగు" in t or any(char in t for char in ["ఆ", "ఇ", "ఈ", "ఉ", "ఊ"]):
        return "Telugu"
    return "English"

# ----------------------------
# Geocoding Helper
# ----------------------------
@st.cache_resource(show_spinner=False, max_entries=10)
def reverse_geocode(lat: float, lon: float) -> tuple:
    try:
        geolocator = Nominatim(user_agent="agrobot_geolocator")
        location = geolocator.reverse((lat, lon), language="en", exactly_one=True)
        if location and location.raw and "address" in location.raw:
            addr = location.raw["address"]
            return (
                addr.get("state", ""),
                addr.get("county", addr.get("state_district", "")),
                addr.get("village", addr.get("town", addr.get("city", ""))))
        return ("", "", "")
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        return ("", "", "")

# ----------------------------
# Weather Data
# ----------------------------
def get_current_weather(lat: float, lon: float) -> dict:
    """Get current weather data from OpenWeatherMap"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "temp": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "conditions": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"],
                "icon": data["weather"][0]["icon"]
            }
        return {}
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return {}

def get_weather_icon(icon_code: str) -> str:
    """Get weather icon from code"""
    icon_map = {
        "01d": "☀️", "01n": "🌙", "02d": "⛅", "02n": "⛅",
        "03d": "☁️", "03n": "☁️", "04d": "☁️", "04n": "☁️",
        "09d": "🌧️", "09n": "🌧️", "10d": "🌦️", "10n": "🌦️",
        "11d": "⛈️", "11n": "⛈️", "13d": "❄️", "13n": "❄️",
        "50d": "🌫️", "50n": "🌫️"
    }
    return icon_map.get(icon_code, "🌡️")

# ----------------------------
# Voice Functions
# ----------------------------
def text_to_speech(text, lang='en'):
    try:
        # Use gTTS for better multilingual support
        if lang == 'te':  # Telugu
            tts = gTTS(text=text, lang='te', slow=False)
        else:  # English
            tts = gTTS(text=text, lang='en', slow=False)
            
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        # Convert to base64 for embedding in HTML
        audio_base64 = base64.b64encode(audio_bytes.read()).decode()
        audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
        """
        st.components.v1.html(audio_html, height=0)
        
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")

def record_audio(filename, duration=5):
    """Record audio using PyAudio"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    frames = []
    
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def speech_to_text_from_file(filename):
    """Convert speech to text from a WAV file"""
    try:
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(filename) as source:
            audio = recognizer.record(source)
        
        # Recognize speech using Google's speech recognition
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Error with speech recognition service: {e}"
    except Exception as e:
        return f"Error: {e}"

# ----------------------------
# Plant Disease Detection using Local Model (if available)
# ----------------------------
@st.cache_resource
def load_plant_disease_model():
    """Load the plant disease detection model from Hugging Face Hub"""
    if not TENSORFLOW_AVAILABLE:
        return None
        
    try:
        # Download the model from Hugging Face Hub
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            cache_dir="models"
        )
        
        # Load the model
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(img, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    try:
        # Resize image
        img = img.resize(target_size)
        # Convert to array and normalize
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def predict_plant_disease(img):
    """Predict plant disease using the local model"""
    if not TENSORFLOW_AVAILABLE:
        return "TensorFlow not available - Plant disease detection disabled", 0.0
    
    try:
        # Load the model
        model = load_plant_disease_model()
        if model is None:
            return "Model not available", 0.0
        
        # Preprocess the image
        img_array = preprocess_image(img)
        if img_array is None:
            return "Error processing image", 0.0
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Get the class name
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Format the class name for display
        if "healthy" in predicted_class.lower():
            return "Plant is healthy", confidence
        else:
            # Extract plant type and disease name
            parts = predicted_class.split("___")
            if len(parts) >= 2:
                plant_type = parts[0].replace("_", " ")
                disease_name = parts[1].replace("_", " ")
                return f"{plant_type} with {disease_name}", confidence
            else:
                return predicted_class.replace("_", " "), confidence
                
    except Exception as e:
        return f"Error in prediction: {str(e)}", 0.0

def get_treatment_recommendation(disease):
    """Get treatment recommendation based on detected disease"""
    disease_lower = disease.lower()
    
    if "healthy" in disease_lower:
        return "Your plant appears to be healthy! Continue with your current care routine including proper watering, sunlight, and nutrient management."
    
    treatment_map = {
        "blight": "Remove affected leaves, apply copper-based fungicide, ensure proper air circulation, and avoid overhead watering.",
        "rot": "Remove affected parts, improve drainage, apply appropriate fungicide, and avoid overwatering.",
        "mildew": "Apply sulfur or potassium bicarbonate-based fungicide, reduce humidity, and ensure good air circulation.",
        "spot": "Remove affected leaves, apply copper fungicide, avoid overhead watering, and ensure proper spacing between plants.",
        "rust": "Remove infected leaves, apply fungicide containing myclobutanil or tebuconazole, and ensure good air circulation.",
        "virus": "Remove and destroy infected plants to prevent spread. Use virus-free planting material and control insect vectors.",
        "scab": "Apply fungicides containing captan or thiophanate-methyl, and remove fallen leaves in autumn.",
        "canker": "Prune affected branches, apply fungicide, and avoid wounding trees.",
        "wilting": "Improve soil drainage, avoid overwatering, and apply appropriate fungicide if fungal infection is suspected.",
        "mold": "Improve air circulation, reduce humidity, and apply appropriate fungicide.",
        "anthracnose": "Apply fungicides containing chlorothalonil or thiophanate-methyl, and remove infected plant debris."
    }
    
    # Find the best matching treatment
    for key, treatment in treatment_map.items():
        if key in disease_lower:
            return f"**Treatment:** {treatment}"
    
    # Default treatment if no specific match
    return "For specific treatment recommendations, consult with a local agricultural expert or extension service. General recommendations include removing affected plant parts, improving air circulation, and applying appropriate fungicides if fungal infection is suspected."

# ----------------------------
# Simplified Routing
# ----------------------------
def enhanced_router(query: str) -> str:
    """Optimized routing with keyword matching"""
    q = (query or "").lower()

    # Policy/subsidy related queries
    policy_keywords = ["policy", "subsidy", "pm-kisan", "scheme", "mandi",
                      "insurance", "price", "loan", "government", "yojana", "benefit"]
    if any(k in q for k in policy_keywords):
        return "ASA – Policy Specialist"

    # Verification/fact-checking
    verification_keywords = ["verify", "cross-check", "double check", "correct",
                            "fact check", "source", "accurate", "truth", "confirm"]
    if any(k in q for k in verification_keywords):
        return "ASC – Fact Checker"

    # Default to agronomy specialist
    return "ASB – Agronomy Specialist"

def route_query(query: str, router_pref: str) -> str:
    if router_pref != "Auto (Router decides)":
        return router_pref
    return enhanced_router(query)

# ----------------------------
# API Call Functions
# ----------------------------
def call_backend(query: str, specialist: str, history: list, weather_info: dict, language: str) -> dict:
    """Call the backend API with the query"""
    try:
        payload = {
            "query": query,
            "specialist": specialist,
            "history": history,
            "weather_info": weather_info,
            "language": language
        }
        
        resp = requests.post("http://localhost:8000/query", json=payload, timeout=30)
        
        if resp.status_code == 200:
            return resp.json()
        else:
            # Fallback response if backend is not available
            fallback_responses = {
                "ASA – Policy Specialist": "I specialize in agricultural policies. For detailed policy information, please visit the official PM-KISAN website or contact your local agriculture office.",
                "ASB – Agronomy Specialist": "As an agronomy specialist, I can help with crop management. For immediate assistance, consider consulting with your local Krishi Vigyan Kendra (KVK).",
                "ASC – Fact Checker": "I verify agricultural information. For accurate information, please check with certified agricultural experts or government sources."
            }
            return {"answer": fallback_responses.get(specialist, "I'm currently unable to process your request. Please try again later."), "sources": []}
    except Exception as e:
        fallback_responses = {
            "ASA – Policy Specialist": "For policy-related queries, please visit the official PM-KISAN portal at https://pmkisan.gov.in",
            "ASB – Agronomy Specialist": "For agronomy advice, contact your local agriculture extension officer or visit https://farmer.gov.in",
            "ASC – Fact Checker": "To verify agricultural information, please consult certified sources like ICAR or your state agriculture department."
        }
        return {"answer": fallback_responses.get(specialist, "Connection error. Please check your internet connection and try again."), "sources": []}

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(
    page_title="Rythumitra – AI Farming Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Agriculture-themed color scheme with higher contrast for disease detection
AGRICULTURE_THEME = {
    "primary": "#2e7d32",        # Dark green
    "secondary": "#4caf50",      # Medium green
    "accent": "#8bc34a",         # Light green
    "background": "#e8f5e9",     # Very light green
    "text_dark": "#1b5e20",      # Dark green text
    "text_black": "#212121",     # Black text
    "card_bg": "#ffffff",        # White cards
    "border": "#c8e6c9",         # Light green border
    "highlight": "#FFD54F",      # Highlight color
    "sidebar_bg": "#1b5e20",     # Dark green sidebar
    "sidebar_text": "#ffffff",   # White sidebar text
    "weather_bg": "#dcedc8",     # Weather card background
    "rag_step_bg": "#1b5e20",    # Dark green for RAG steps
    "rag_step_text": "#ffffff",  # White text for RAG steps
    "disease_text": "#000000",   # Black text for disease detection (high contrast)
    "disease_bg": "#ffffff",     # White background for disease detection
    "uploader_text": "#1b5e20",  # Dark green text for uploader (high contrast)
}

# Apply theme with high contrast for disease detection
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {AGRICULTURE_THEME['background']};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {AGRICULTURE_THEME['primary']} !important;
    }}
    .stButton>button {{
        background: linear-gradient(to right, {AGRICULTURE_THEME['primary']}, {AGRICULTURE_THEME['secondary']}) !important;
        color: white !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {AGRICULTURE_THEME['sidebar_bg']} !important;
    }}
    .terminal-output {{
        color:black;
        background: rgba(255, 255, 255, 0.95);
        border-left: 5px solid {AGRICULTURE_THEME['primary']};
        padding: 15px;
        border-radius: 0 10px 10px 0;
    }}
    .weather-card {{
        background: {AGRICULTURE_THEME['weather_bg']};
        border-radius: 12px;
        padding: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .location-card {{
        background: linear-gradient(135deg, #e8f5e9, #dcedc8);
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 25px;
    }}
    .metric-value {{
        font-size: 1.2em;
        font-weight: bold;
        color: {AGRICULTURE_THEME['primary']};
    }}
    .metric-label {{
        font-size: 0.9em;
        color: {AGRICULTURE_THEME['text_dark']};
    }}
    .rag-step {{
        background-color: {AGRICULTURE_THEME['rag_step_bg']};
        color: {AGRICULTURE_THEME['rag_step_text']};
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        font-weight: bold;
    }}
    .disease-result {{
        color: {AGRICULTURE_THEME['disease_text']} !important;
        background: {AGRICULTURE_THEME['disease_bg']};
        border-left: 5px solid #2e7d32;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
        font-weight: bold;
        font-size: 1.1em;
    }}
    .treatment-info {{
        color: {AGRICULTURE_THEME['disease_text']} !important;
        background: {AGRICULTURE_THEME['disease_bg']};
        border-left: 5px solid #FFD54F;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
        font-size: 1em;
    }}
    .confidence-badge {{
        background: #006400;
        color: white !important;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.9em;
    }}
    .uploader-text {{
        color: {AGRICULTURE_THEME['uploader_text']} !important;
        font-weight: bold;
        font-size: 1.1em;
    }}
    .disease-tab-title {{
        color: {AGRICULTURE_THEME['primary']} !important;
        font-weight: bold;
        font-size: 1.5em;
        margin-bottom: 20px;
    }}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "user_lat" not in st.session_state:
    # Initialize with None to indicate location not yet set
    st.session_state.user_lat = None
    st.session_state.user_lon = None
    st.session_state.use_browser_location = False

if "selected_route" not in st.session_state:
    st.session_state.selected_route = "Auto (Router decides)"
if "enable_critic" not in st.session_state:
    st.session_state.enable_critic = True
if "response_lang" not in st.session_state:
    st.session_state.response_lang = "Auto"
if "weather_data" not in st.session_state:
    st.session_state.weather_data = {}
if "use_browser_location" not in st.session_state:
    st.session_state.use_browser_location = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "recording" not in st.session_state:
    st.session_state.recording = False
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "audio_filename" not in st.session_state:
    st.session_state.audio_filename = f"recorded_audio_{int(time.time())}.wav"
if "recording_thread" not in st.session_state:
    st.session_state.recording_thread = None
if "recording_status" not in st.session_state:
    st.session_state.recording_status = "idle"
if "mic_permission" not in st.session_state:
    st.session_state.mic_permission = False
if "location_requested" not in st.session_state:
    st.session_state.location_requested = False
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Chat"
if "location_attempted" not in st.session_state:
    st.session_state.location_attempted = False
if "processing_audio" not in st.session_state:
    st.session_state.processing_audio = False

# Get current location if not already set
if not st.session_state.location_attempted:
    try:
        # Try to get current location first
        loc = get_geolocation()
        if loc and 'coords' in loc:
            st.session_state.user_lat = loc['coords']['latitude']
            st.session_state.user_lon = loc['coords']['longitude']
            st.session_state.use_browser_location = True
            st.session_state.location_attempted = True
        else:
            # If browser location fails, use IP-based location
            try:
                ip_response = requests.get('https://ipapi.co/json/', timeout=5)
                if ip_response.status_code == 200:
                    ip_data = ip_response.json()
                    st.session_state.user_lat = ip_data.get('latitude', 17.7271)
                    st.session_state.user_lon = ip_data.get('longitude', 83.3013)
                    st.session_state.location_attempted = True
                else:
                    # Fallback to default if IP location fails
                    st.session_state.user_lat = 17.7271
                    st.session_state.user_lon = 83.3013
                    st.session_state.location_attempted = True
            except:
                # Final fallback to default
                st.session_state.user_lat = 17.7271
                st.session_state.user_lon = 83.3013
                st.session_state.location_attempted = True
    except:
        # If all location methods fail, use default
        st.session_state.user_lat = 17.7271
        st.session_state.user_lon = 83.3013
        st.session_state.location_attempted = True

# Get location info
state, district, village = reverse_geocode(st.session_state.user_lat, st.session_state.user_lon)
location_str = f"{village}, {district}, {state}" if village or district or state else "your location"

# Get weather data
if WEATHER_API_KEY:
    st.session_state.weather_data = get_current_weather(st.session_state.user_lat, st.session_state.user_lon)

# Main UI
st.title("🌾 Rythumitra — AI Farming Assistant")
st.markdown(f"""
    <div class="location-card">
        <p style="font-size: 18px; margin: 0; color: #1b5e20; font-weight: 500;">
            తెలుగు & English AI Farming Assistant | Location: {location_str}
        </p>
        {"" if st.session_state.use_browser_location else "<p style='color: #d32f2f; margin: 5px 0 0 0; font-size: 14px;'>Using approximate location. Click 'Use My Current Location' for more accurate results.</p>"}
    </div>
""", unsafe_allow_html=True)

# Weather display
if st.session_state.weather_data:
    weather = st.session_state.weather_data
    icon = get_weather_icon(weather.get("icon", ""))
    st.markdown(f"""
        <div class="weather-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; color: {AGRICULTURE_THEME['primary']};">Current Weather</h3>
                    <div style="font-size: 1.5em; margin: 5px 0;">{icon} {weather.get('conditions', 'N/A').title()}</div>
                </div>
                <div style="font-size: 2.5em; font-weight: bold;">{weather.get('temp', 'N/A')}°C</div>
                <div>
                    <div class="metric-label">Feels Like</div>
                    <div class="metric-value">{weather.get('feels_like', 'N/A')}°C</div>
                </div>
                <div>
                    <div class="metric-label">Humidity</div>
                    <div class="metric-value">{weather.get('humidity', 'N/A')}%</div>
                </div>
                <div>
                    <div class="metric-label">Wind Speed</div>
                    <div class="metric-value">{weather.get('wind_speed', 'N/A')} km/h</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Sidebar with organized sections
with st.sidebar:
    st.sidebar.title("⚙️ Control Panel")
    
    # Navigation
    st.sidebar.header("🧭 Navigation")
    tab_options = ["Chat", "Plant Disease Detection"]
    
    st.session_state.current_tab = st.sidebar.radio(
        "Select Mode:",
        options=tab_options,
        index=0 if st.session_state.current_tab == "Chat" else 1
    )

    # Model Selection Section
    st.sidebar.header("🧠 Model Selection")
    st.session_state.selected_route = st.sidebar.selectbox(
        "Routing Method:",
        options=ROUTER_OPTIONS,
        index=ROUTER_OPTIONS.index(st.session_state.selected_route),
        help="Auto routing selects the best model, or choose a specific model"
    )

    # Show model info
    if st.session_state.selected_route != "Auto (Router decides)":
        model_info = MODEL_REGISTRY[st.session_state.selected_route]
        st.sidebar.markdown(f"""
            <div style="background: {AGRICULTURE_THEME['card_bg']};
                        border-radius: 10px; padding: 10px; margin: 10px 0;
                        border-left: 4px solid {model_info['color']};">
                <div style="color: {model_info['color']}; font-weight: bold;">
                    {st.session_state.selected_route}
                </div>
                <div><strong>Role:</strong> {model_info["role"].replace('specialist-', '').title()}</div>
            </div>
        """, unsafe_allow_html=True)

    # Language Settings Section
    st.sidebar.header("🌐 Language Settings")
    st.session_state.response_lang = st.sidebar.selectbox(
        "Response Language:",
        options=["Auto", "English", "Telugu"],
        index=["Auto", "English", "Telugu"].index(st.session_state.response_lang),
        help="Choose response language (Auto detects from question)"
    )

    # Processing Options Section
    st.sidebar.header("⚙️ Processing Options")
    st.session_state.enable_critic = st.sidebar.toggle(
        "Enable Fact Checker",
        value=st.session_state.enable_critic,
        help="Verify response accuracy with fact checker"
    )

    # Voice Settings Section
    st.sidebar.header("🎤 Voice Settings")
    st.session_state.auto_voice = st.sidebar.toggle(
        "Auto Voice Response",
        value=False,
        help="Automatically speak responses"
    )

    # Location Section
    st.sidebar.header("📍 Location Setup")

    # Browser location button
    if st.sidebar.button("📍 Use My Current Location", use_container_width=True):
        st.session_state.location_requested = True

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.session_state.user_lat = st.number_input(
            "Latitude",
            value=st.session_state.user_lat,
            format="%.6f",
            key="lat_input"
        )
    with col2:
        st.session_state.user_lon = st.number_input(
            "Longitude",
            value=st.session_state.user_lon,
            format="%.6f",
            key="lon_input"
        )

    # Weather refresh button
    if WEATHER_API_KEY and st.sidebar.button("🔄 Refresh Weather Data", use_container_width=True):
        st.session_state.weather_data = get_current_weather(st.session_state.user_lat, st.session_state.user_lon)
        st.rerun()

# Handle location request
if st.session_state.location_requested:
    try:
        # Use streamlit_js_eval to get geolocation
        loc = get_geolocation()
        
        if loc and 'coords' in loc:
            st.session_state.user_lat = loc['coords']['latitude']
            st.session_state.user_lon = loc['coords']['longitude']
            st.session_state.use_browser_location = True
            # Update weather data with new location
            st.session_state.weather_data = get_current_weather(st.session_state.user_lat, st.session_state.user_lon)
            st.sidebar.success("Location updated successfully!")
            st.session_state.location_requested = False
            st.rerun()
        else:
            st.sidebar.error("Could not access your location. Please check your browser permissions.")
            st.session_state.location_requested = False
    except Exception as e:
        st.sidebar.error(f"Failed to get location: {e}")
        st.session_state.location_requested = False

# Main content area based on selected tab
if st.session_state.current_tab == "Chat":
    st.subheader("💬 Ask Your Agriculture Question")

    # Voice input section
    if st.session_state.recording_status == "recording":
        st.warning("Recording in progress... Speak now (5 seconds max)")
        
        # Stop recording button
        if st.button("⏹️ Stop Recording", key="stop_recording", use_container_width=True):
            st.session_state.recording_status = "processing"
            st.session_state.processing_audio = True
            st.rerun()
            
    elif st.session_state.recording_status == "processing" and st.session_state.processing_audio:
        with st.spinner("Processing your voice..."):
            # Wait for recording thread to finish if it's still running
            if st.session_state.recording_thread and st.session_state.recording_thread.is_alive():
                st.session_state.recording_thread.join()
            
            # Convert speech to text
            voice_text = speech_to_text_from_file(st.session_state.audio_filename)
            
            if voice_text and not voice_text.startswith("Could not") and not voice_text.startswith("Error"):
                st.session_state.user_query = voice_text
                st.success("Voice input captured!")
            elif voice_text.startswith("Could not understand audio"):
                st.info("Could not understand the audio. Please try again.")
            else:
                st.error(f"Voice recognition failed: {voice_text}")
            
            # Clean up audio file
            try:
                if os.path.exists(st.session_state.audio_filename):
                    os.remove(st.session_state.audio_filename)
            except:
                pass
                
            st.session_state.recording_status = "idle"
            st.session_state.processing_audio = False
            st.session_state.audio_filename = f"recorded_audio_{int(time.time())}.wav"
            st.rerun()
    else:
        # Text input and voice button
        col1, col2 = st.columns([3, 1])
        with col1:
            user_query = st.text_area(
                "Enter your question in English or తెలుగు",
                value=st.session_state.user_query,
                placeholder="e.g., When should I irrigate my paddy crop? / వరి పంటకు నేను ఎప్పుడు నీటి పారుదల చేయాలి?",
                height=120,
                key="user_query_input",
                label_visibility="collapsed"
            )
            st.session_state.user_query = user_query
        with col2:
            st.write("")
            st.write("")
            if st.button("🎤 Start Recording", key="start_recording", use_container_width=True):
                # Check microphone permission
                try:
                    # Test microphone access
                    p = pyaudio.PyAudio()
                    p.terminate()
                    st.session_state.mic_permission = True
                    
                    # Start recording in a thread
                    st.session_state.recording_status = "recording"
                    st.session_state.audio_filename = f"recorded_audio_{int(time.time())}.wav"
                    st.session_state.recording_thread = threading.Thread(
                        target=record_audio, 
                        args=(st.session_state.audio_filename, 20)
                    )
                    st.session_state.recording_thread.start()
                    st.rerun()
                except Exception as e:
                    st.error(f"Microphone access denied: {e}. Please allow microphone access in your browser.")
                    st.session_state.mic_permission = False

    # Display chat history
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")
            # Add text-to-speech button for each response
            if st.button("🔊 Play", key=f"speak_{i}"):
                lang = 'te' if detect_lang_from_text(msg['content']) == "Telugu" else 'en'
                text_to_speech(msg['content'], lang)

    # Submit button
    if st.button("🚀 Get AI-Powered Answer", type="primary", use_container_width=True):
        if not st.session_state.user_query.strip():
            st.warning("Please enter a question")
        else:
            # Get location info
            state, district, village = reverse_geocode(st.session_state.user_lat, st.session_state.user_lon)

            # Initialize progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Routing
            status_text.text("Routing to specialist...")
            progress_bar.progress(15)
            route_code = route_query(st.session_state.user_query, st.session_state.selected_route)
            model_info = MODEL_REGISTRY[route_code]
            model_color = model_info["color"]

            # Display routing info
            st.markdown(f"""
                <div style="text-align: center; padding: 15px; border-radius: 12px;
                            background: {AGRICULTURE_THEME['card_bg']};
                            margin-bottom: 20px;">
                    <div style="font-weight: bold; color: {model_color}; font-size: 1.1em; margin-bottom: 10px;">
                        {route_code}
                    </div>
                    <div style="color: {AGRICULTURE_THEME['text_black']};">
                        <strong>Response Language:</strong> {st.session_state.response_lang}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Step 2: RAG processing
            status_text.text("Searching knowledge base...")
            progress_bar.progress(30)
            
            # Determine final language
            final_lang = detect_lang_from_text(st.session_state.user_query) if st.session_state.response_lang == "Auto" else st.session_state.response_lang
            
            # Call backend for specialist answer
            status_text.text("Processing query with RAG...")
            progress_bar.progress(45)
            
            # Show RAG processing steps with high contrast styling
            with st.expander("RAG Processing Steps", expanded=True):
                st.markdown('<div class="rag-step">1. 📚 Chunking and indexing documents</div>', unsafe_allow_html=True)
                st.markdown('<div class="rag-step">2. 🔍 Generating embeddings for query</div>', unsafe_allow_html=True)
                st.markdown('<div class="rag-step">3. 🤔 Searching vector database for relevant context</div>', unsafe_allow_html=True)
                st.markdown('<div class="rag-step">4. 🧠 Retrieving top matches from knowledge base</div>', unsafe_allow_html=True)
                st.markdown('<div class="rag-step">5. 📋 Preparing context for LLM</div>', unsafe_allow_html=True)
            
            status_text.text("Generating answer...")
            progress_bar.progress(60)
            
            specialist_response = call_backend(
                st.session_state.user_query, 
                route_code, 
                st.session_state.chat_history,
                st.session_state.weather_data,
                final_lang
            )
            
            answer = specialist_response.get("answer", "No answer received")
            sources = specialist_response.get("sources", [])

            # Step 3: Critic processing
            if st.session_state.enable_critic:
                status_text.text("Verifying with fact checker...")
                progress_bar.progress(80)
                
                # Call backend for critic refinement
                critic_response = call_backend(
                    f"Original answer: {answer}\n\nPlease fact-check this answer for the question: {st.session_state.user_query}",
                    "ASC – Fact Checker",
                    st.session_state.chat_history,
                    st.session_state.weather_data,
                    final_lang
                )
                
                answer = critic_response.get("answer", answer)

            progress_bar.progress(100)
            status_text.empty()

            # Update chat history
            st.session_state.chat_history.append({"role": "user", "content": st.session_state.user_query})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            # Display result
            st.markdown("---")
            st.subheader("🌱 AI Recommendation" if final_lang != "Telugu" else "🌱 AI సిఫార్సు")
            st.markdown(f'<div class="terminal-output">{answer}</div>', unsafe_allow_html=True)
            
            # Auto voice response if enabled
            if st.session_state.auto_voice:
                lang = 'te' if final_lang == "Telugu" else 'en'
                text_to_speech(answer, lang)
            
            # Show sources
            if sources:
                with st.expander("Sources"):
                    st.json(sources)

elif st.session_state.current_tab == "Plant Disease Detection":
    st.markdown('<div class="disease-tab-title">🌿 Plant Disease Detection</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="uploader-text">Upload an image of a plant leaf to detect potential diseases</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a plant leaf for disease detection",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Detect Disease", type="primary", use_container_width=True):
            with st.spinner("Analyzing image for diseases..."):
                # Predict disease using the local model
                disease, confidence = predict_plant_disease(img)
                
                if "Error" not in disease:
                    st.markdown(f"""
                        <div class="disease-result">
                            Detection Result: {disease}
                        </div>
                        <div style="margin: 10px 0;">
                            Confidence: <span class="confidence-badge">{confidence:.2%}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Provide additional information based on the detection
                    st.markdown("---")
                    st.markdown('<div class="uploader-text">📋 Recommended Treatment</div>', unsafe_allow_html=True)
                    
                    # Get treatment recommendation
                    treatment_text = get_treatment_recommendation(disease)
                    
                    st.markdown(f'<div class="treatment-info">{treatment_text}</div>', unsafe_allow_html=True)
                else:
                    st.error(f"Error in disease detection: {disease}")

# Footer
st.markdown("---")
st.markdown(f"""
    <div style="text-align: center; color: {AGRICULTURE_THEME['text_dark']}; font-size: 0.9em; padding: 20px;">
        <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 10px; margin-top: 20px;">
            <span style="display: flex; align-items: center;">🌐 తెలుగు & English Support</span>
            <span style="display: flex; align-items: center;">🌦️ Live Weather Integration</span>
            <span style="display: flex; align-items: center;">📍 Location-aware Advice</span>
            <span style="display: flex; align-items: center;">🎤 Voice Input & Output</span>
            <span style="display: flex; align-items: center;">🌿 Plant Disease Detection</span>
        </div>
        <div>Rythumitra AI Assistant v9.0 • Powered by KissanAI</div>
    </div>
""", unsafe_allow_html=True)
