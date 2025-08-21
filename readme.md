# Rythumitra – AI Farming Assistant 🌾

A multilingual AI-powered agricultural assistant that provides farmers with expert advice, plant disease detection, and location-aware recommendations in both English and Telugu.

## Features

- **🤖 AI-Powered Agriculture Advice**: Get expert recommendations from specialized AI models
- **🌐 Multilingual Support**: Full support for both English and Telugu languages
- **📍 Location-Aware Responses**: Contextual advice based on your geographical location
- **🌦️ Live Weather Integration**: Real-time weather data for personalized recommendations
- **🎤 Voice Input & Output**: Speak your questions and hear responses
- **🌿 Plant Disease Detection**: Upload plant images for AI-powered disease identification
- **🔍 Fact-Checking**: Built-in verification system for accurate information

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **AI Models**: Groq API with Llama 4
- **Vector Database**: Qdrant
- **Speech Recognition**: Google Speech Recognition
- **Text-to-Speech**: Google Text-to-Speech (gTTS)
- **Geolocation**: OpenStreetMap Nominatim
- **Weather Data**: OpenWeatherMap API
- **Plant Disease Detection**: Hugging Face Inference API

## Installation

1. Clone the repository:
bash
git clone https://github.com/your-username/rythumitra-ai-farming-assistant.git
cd rythumitra-ai-farming-assistant

# Usage

## Chat Mode
Ask agricultural questions in English or Telugu using the chat interface. The AI will route your question to the appropriate specialist and provide tailored advice.

## Plant Disease Detection
Upload images of plant leaves for AI-powered disease diagnosis. The system will analyze the image and provide treatment recommendations.

## Voice Input
Use the microphone button to ask questions verbally. The system supports speech recognition in both English and Telugu.

## Location Settings
Allow location access for precise weather and location-specific advice, or manually set coordinates in the sidebar.

## Model Selection
Choose between automatic routing (AI selects the best specialist) or manually select from:
- ASA – Policy Specialist
- ASB – Agronomy Specialist  
- ASC – Fact Checker

