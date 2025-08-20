%%writefile capital_man.py

import json
import time
import torch
import logging
import requests
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from geopy.geocoders import Nominatim
import streamlit as st
from streamlit_js_eval import get_geolocation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------
# Configuration
# ----------------------------
MODEL_REGISTRY = {
    "ASA – Policy Specialist": {
        "id": "KissanAI/ThinkingDhenu1-CRSA-India-preview",
        "role": "specialist-policy",
        "color": "#27AE60",
        "requires_trust": True,
        "format": "mistral"
    },
    "ASB – Agronomy Specialist": {
        "id": "KissanAI/Dhenu2-In-Llama3.2-3B-Instruct",
        "role": "specialist-agronomy",
        "color": "#E67E22",
        "requires_trust": True,
        "format": "llama3"
    },
    "ASC – Fact Checker": {
        "id": "bharatgenai/AgriParam",
        "role": "specialist-crosscheck",
        "color": "#C0392B",
        "requires_trust": True,
        "format": "mistral"
    }
}

ROUTER_OPTIONS = ["Auto (Router decides)"] + list(MODEL_REGISTRY.keys())
WEATHER_API_KEY = "b10a43e49ad59f27140d077c8f1a6bfd"  # Replace with your actual API key

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
# Model Loading and Inference
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_id: str, requires_trust: bool):
    """Load model efficiently with quantization"""
    try:
        logger.info(f"Loading model: {model_id}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=requires_trust)

        # Load model with 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=requires_trust
        )

        logger.info(f"Successfully loaded model: {model_id}")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Model loading error for {model_id}: {e}")
        # Fallback without quantization
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=requires_trust
            )
            return tokenizer, model
        except Exception as e2:
            logger.error(f"Fallback loading failed: {e2}")
            return None, None

def generate_text(tokenizer, model, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
    """Generate text using the model (optimized)"""
    try:
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(model.device)

        # Generate response
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode only the new tokens (after input length)
        input_length = inputs.input_ids.shape[1]
        response = tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )

        # Clean up common artifacts
        stop_phrases = ["<|end|>", "<|eot_id|>", "###", "Human:", "Assistant:", "\nUser:", "\nSystem:"]
        for phrase in stop_phrases:
            response = response.replace(phrase, "").strip()

        return response
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return f"Generation error: {str(e)}"

def format_prompt(model_format: str, system_prompt: str, user_query: str, weather_info: dict) -> str:
    """Format prompt according to model requirements with weather context"""
    weather_context = ""
    if weather_info:
        temp = weather_info.get("temp", "N/A")
        conditions = weather_info.get("conditions", "N/A")
        humidity = weather_info.get("humidity", "N/A")
        wind = weather_info.get("wind_speed", "N/A")
        weather_context = f"\nCurrent Weather: {conditions}, Temp: {temp}°C, Humidity: {humidity}%, Wind: {wind} km/h"

    full_system_prompt = system_prompt + weather_context

    if model_format == "llama3":
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {full_system_prompt}
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        {user_query}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
    elif model_format == "mistral":
        return f"<s>[INST] {full_system_prompt}\n\n{user_query} [/INST]"
    else:
        return f"System: {full_system_prompt}\n\nUser: {user_query}\n\nAssistant:"

def call_model_chat(model_info: dict, system_prompt: str, user_query: str, weather_info: dict, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
    """Call actual LLM for response generation"""
    tokenizer, model = load_model(model_info["id"], model_info["requires_trust"])
    if tokenizer is None or model is None:
        return "Model loading failed"

    # Format prompt according to model requirements
    prompt = format_prompt(model_info["format"], system_prompt, user_query, weather_info)

    return generate_text(tokenizer, model, prompt, max_new_tokens, temperature)

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
# Specialist Functions
# ----------------------------
def specialist_answer(route_code: str, query: str, state: str, district: str, language: str, weather_info: dict, max_new: int, temperature: float) -> str:
    model_info = MODEL_REGISTRY[route_code]

    # Determine final language
    final_lang = detect_lang_from_text(query) if language == "Auto" else language

    # System prompt based on role and language
    if final_lang == "Telugu":
        sys_prompt = (
            f"మీరు భారతీయ రైతులకు నిపుణ వ్యవసాయ సలహాదారు.  \n"
            f"వినియోగదారు {district}, {state} వద్ద ఉన్నారు. \n"
            "పూర్తి వాక్యాలలో స్పష్టమైన, ఆచరణాత్మక సలహా ఇవ్వండి."
        )
    else:
        sys_prompt = (
            f"You are an expert agriculture advisor for Indian farmers. \n"
            f"The user is located in {district}, {state}.\n"
            "Give clear, practical advice in complete sentences."
        )

    return call_model_chat(model_info, sys_prompt, query, weather_info, max_new_tokens=max_new, temperature=temperature)

def critic_refine(original_answer: str, query: str, language: str, weather_info: dict, max_new: int, temperature: float) -> str:
    model_info = MODEL_REGISTRY["ASC – Fact Checker"]

    # Determine final language
    final_lang = detect_lang_from_text(query) if language == "Auto" else language

    # System prompt for critic
    if final_lang == "Telugu":
        sys_prompt = (
            "మీరు వ్యవసాయ సత్యాసత్యత తనిఖీదారు. ఇచ్చిన సమాధానాన్ని వాస్తవికత, సురక్షితత్వం మరియు "
            "భారత సందర్భానికి అనుగుణంగా ధృవీకరించండి. "
            "సరిదిద్దడం అవసరమైతే, సరిదిద్దబడిన సమాధానం ఇవ్వండి."
        )
        user_prompt = f"అసలు సమాధానం: {original_answer}\n\nప్రశ్న: {query}"
    else:
        sys_prompt = (
            "You are an agriculture fact-checker. Verify the answer for factuality, safety, and "
            "Indian context. If corrections are needed, provide a corrected answer."
        )
        user_prompt = f"Original Answer: {original_answer}\n\nQuestion: {query}"

    return call_model_chat(model_info, sys_prompt, user_prompt, weather_info, max_new_tokens=max_new, temperature=temperature)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(
    page_title="Rythumitra – AI Farming Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Agriculture-themed color scheme
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
    "weather_bg": "#dcedc8"      # Weather card background
}

# Apply theme
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
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "user_lat" not in st.session_state:
    st.session_state.user_lat = 17.7271  # Default: Visakhapatnam
    st.session_state.user_lon = 83.3013
if "selected_route" not in st.session_state:
    st.session_state.selected_route = "Auto (Router decides)"
if "enable_critic" not in st.session_state:
    st.session_state.enable_critic = True
if "response_lang" not in st.session_state:
    st.session_state.response_lang = "Auto"
if "max_new" not in st.session_state:
    st.session_state.max_new = 256
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "weather_data" not in st.session_state:
    st.session_state.weather_data = {}
if "use_browser_location" not in st.session_state:
    st.session_state.use_browser_location = False

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

    st.session_state.max_new = st.sidebar.slider(
        "Response Length",
        128, 512, st.session_state.max_new, step=32,
        help="Number of tokens in the response"
    )

    st.session_state.temperature = st.sidebar.slider(
        "Temperature",
        0.1, 1.0, st.session_state.temperature, step=0.1,
        help="Controls randomness (lower = more deterministic)"
    )

    # Location Section
    st.sidebar.header("📍 Location Setup")

    # Browser location button
    if st.sidebar.button("📍 Use My Current Location", use_container_width=True):
        loc = get_geolocation()
        if loc and "coords" in loc:
            st.session_state.user_lat = loc["coords"]["latitude"]
            st.session_state.user_lon = loc["coords"]["longitude"]
            st.session_state.use_browser_location = True
            st.rerun()

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

# Main content area
st.subheader("💬 Ask Your Agriculture Question")
user_query = st.text_area(
    "Enter your question in English or తెలుగు",
    placeholder="e.g., When should I irrigate my paddy crop? / వరి పంటకు నేను ఎప్పుడు నీటి పారుదల చేయాలి?",
    height=120,
    key="user_query",
    label_visibility="collapsed"
)


# Submit button
if st.button("🚀 Get AI-Powered Answer", type="primary", use_container_width=True):
    if not user_query.strip():
        st.warning("Please enter a question")
    else:
        # Get location info
        state, district, village = reverse_geocode(st.session_state.user_lat, st.session_state.user_lon)

        # Initialize progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Routing
        status_text.text("Routing to specialist...")
        progress_bar.progress(30)
        route_code = route_query(user_query, st.session_state.selected_route)
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

        # Step 2: Specialist processing
        status_text.text("Generating answer...")
        progress_bar.progress(60)
        answer = specialist_answer(
            route_code,
            user_query,
            state,
            district,
            st.session_state.response_lang,
            st.session_state.weather_data,
            st.session_state.max_new,
            st.session_state.temperature
        )

        # Step 3: Critic processing
        if st.session_state.enable_critic:
            status_text.text("Verifying with fact checker...")
            progress_bar.progress(80)
            answer = critic_refine(
                answer,
                user_query,
                st.session_state.response_lang,
                st.session_state.weather_data,
                min(256, st.session_state.max_new),
                st.session_state.temperature
            )

        progress_bar.progress(100)
        status_text.empty()

        # Display result
        st.markdown("---")
        st.subheader("🌱 AI Recommendation" if st.session_state.response_lang != "Telugu" else "🌱 AI సిఫార్సు")
        st.markdown(f'<div class="terminal-output">{answer}</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
    <div style="text-align: center; color: {AGRICULTURE_THEME['text_dark']}; font-size: 0.9em; padding: 20px;">
        <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 10px; margin-top: 20px;">
            <span style="display: flex; align-items: center;">🌐 తెలుగు & English Support</span>
            <span style="display: flex; align-items: center;">🌦️ Live Weather Integration</span>
            <span style="display: flex; align-items: center;">📍 Location-aware Advice</span>
            <span style="display: flex; align-items: center;">⚡ Optimized Performance</span>
        </div>
        <div>Rythumitra AI Assistant v8.0 • Powered by KissanAI</div>
    </div>
""", unsafe_allow_html=True)
