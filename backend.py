import os, json
from datetime import datetime
from qdrant_client import QdrantClient, models
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from langchain_groq import ChatGroq
import requests
import numpy as np

# API Keys - Use environment variables for security
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.snRi6tx5MD_PDpWJHQ3JY68MB1gbHWb8TV8c03ROrYE")
QDRANT_URL = os.getenv("QDRANT_URL", "https://2d8f5f32-7119-4c89-8423-dca5151be814.us-west-1-0.aws.cloud.qdrant.io")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
COLLECTION_NAME = "multimodal_rag_data"
HISTORY_DIRECTORY = "history"

# Use Hugging Face Inference API for embeddings
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "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")  # Use environment variable

# Init clients
try:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="meta-llama/llama-4-scout-17b-16e-instruct")
except Exception as e:
    print(f"Error initializing clients: {e}")
    # Fallback: Create mock clients for basic functionality
    qdrant_client = None
    llm = None

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create history directory if it doesn't exist
os.makedirs(HISTORY_DIRECTORY, exist_ok=True)

# Models
class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    specialist: str
    history: List[ChatMessage] = Field(default_factory=list)
    weather_info: Optional[Dict[str, Any]] = None
    language: str = "English"

# Get embeddings using Hugging Face API with fallback
def get_embeddings(texts):
    if not HF_API_TOKEN or HF_API_TOKEN == "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":
        # Return a simple TF-IDF like embedding as fallback
        print("Using fallback embeddings - HF token not configured")
        words = " ".join(texts).lower().split()
        vocab = list(set(words))
        embedding = [words.count(word) for word in vocab]
        # Pad or truncate to 384 dimensions
        if len(embedding) < 384:
            embedding += [0] * (384 - len(embedding))
        else:
            embedding = embedding[:384]
        return [embedding]
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error from HF API: {response.status_code} - {response.text}")
            # Return a simple fallback embedding
            return [[0.1] * 384]
    except Exception as e:
        print(f"Exception in get_embeddings: {e}")
        # Return a simple fallback embedding
        return [[0.1] * 384]

# Search Qdrant with fallback
def search_text(query, top_k=5):
    if not qdrant_client:
        print("Qdrant client not available, using fallback search")
        # Return some mock results
        return [
            type('obj', (object,), {
                'payload': {
                    'source': 'Fallback Knowledge Base', 
                    'content': 'Agricultural best practices recommend consulting local experts for region-specific advice.'
                }
            })()
        ]
    
    # Get embedding
    embeddings = get_embeddings([query])
    
    # Ensure we have a valid embedding vector
    if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list):
        query_embedding = embeddings[0]
    else:
        # Fallback to zero vector
        query_embedding = [0.0] * 384

    try:
        # For multimodal collections, we need to specify the vector name
        return qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=("text", query_embedding),  # Specify vector name for multimodal collection
            limit=top_k,
            with_payload=True
        )
    except Exception as e:
        print(f"Qdrant search error: {e}")
        try:
            # Fallback to regular search if named vector search fails
            return qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )
        except Exception as e2:
            print(f"Qdrant fallback search also failed: {e2}")
            # Return mock results
            return [
                type('obj', (object,), {
                    'payload': {
                        'source': 'Fallback Knowledge Base', 
                        'content': 'Agricultural best practices recommend consulting local experts for region-specific advice.'
                    }
                })()
            ]

# Format results
def format_context(text_results):
    context, source_info = "", []
    for result in text_results:
        source = result.payload.get("source", "unknown")
        content = result.payload.get("content", "")
        context += f"- {content}\n"
        if source not in [s['source'] for s in source_info]:
            source_info.append({"source": source, "content": content[:200] + "..."})
    return context, source_info

# Generate with Groq with fallback
def generate_answer(query: str, context: str, history: List[ChatMessage], specialist: str, weather_info: dict, language: str):
    if not llm:
        # Fallback response if LLM is not available
        fallback_responses = {
            "ASA – Policy Specialist": "I specialize in agricultural policies. For detailed policy information, please visit the official PM-KISAN website or contact your local agriculture office.",
            "ASB – Agronomy Specialist": "As an agronomy specialist, I can help with crop management. For immediate assistance, consider consulting with your local Krishi Vigyan Kendra (KVK).",
            "ASC – Fact Checker": "I verify agricultural information. For accurate information, please check with certified agricultural experts or government sources."
        }
        return fallback_responses.get(specialist, "I'm currently unable to process your request. Please try again later.")
    
    conversation_history = "".join(
        f"{m.role}: {m.content}\n"
        for m in history
    )
    
    # Create system prompt based on specialist and language
    if language == "Telugu":
        if specialist == "ASA – Policy Specialist":
            system_prompt = "మీరు భారతీయ వ్యవసాయ విధానాల నిపుణుడు. ప్రభుత్వ విధానాలు, సబ్సిడీలు మరియు పథకాల గురించి స్పష్టమైన, ఆచరణాత్మక సలహా ఇవ్వండి."
        elif specialist == "ASC – Fact Checker":
            system_prompt = "మీరు వ్యవసాయ సత్యాసత్యత తనిఖీదారు. సమాచారం యొక్క ఖచ్చితత్వాన్ని ధృవీకరించండి మరియు అవసరమైతే సరిదిద్దబడిన సమాచారాన్ని అందించండి."
        else:  # Agronomy specialist
            system_prompt = "మీరు భారతీయ రైతులకు నిపుణ వ్యవసాయ సలహాదారు. పంట నిర్వహణ, నీటిపారుదల, ఎరువులు మరియు కీటకాల గురించి స్పష్టమైన, ఆచరణాత్మక సలహా ఇవ్వండి."
    else:
        if specialist == "ASA – Policy Specialist":
            system_prompt = "You are a policy specialist for Indian agriculture. Provide clear, practical advice about government policies, subsidies, and schemes."
        elif specialist == "ASC – Fact Checker":
            system_prompt = "You are a fact checker for Indian agriculture. Verify the information for accuracy and provide corrected information if needed."
        else:  # Agronomy specialist
            system_prompt = "You are an agronomy specialist for Indian agriculture. Provide clear, practical advice about crop management, irrigation, fertilizers, and pests."
    
    # Add weather context if available
    weather_context = ""
    if weather_info:
        temp = weather_info.get("temp", "N/A")
        conditions = weather_info.get("conditions", "N/A")
        humidity = weather_info.get("humidity", "N/A")
        wind = weather_info.get("wind_speed", "N/A")
        weather_context = f"\nCurrent Weather: {conditions}, Temp: {temp}°C, Humidity: {humidity}%, Wind: {wind} km/h"
    
    prompt = f"""
{system_prompt}
{weather_context}

Conversation so far:
{conversation_history}

Context from documents:
{context}

User question: {query}

Answer clearly and concisely without hallucination, staying in agricultural context:
"""
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        print(f"Error generating answer with LLM: {e}")
        # Fallback response
        fallback_responses = {
            "ASA – Policy Specialist": "For policy-related queries, please visit the official PM-KISAN portal at https://pmkisan.gov.in",
            "ASB – Agronomy Specialist": "For agronomy advice, contact your local agriculture extension officer or visit https://farmer.gov.in",
            "ASC – Fact Checker": "To verify agricultural information, please consult certified sources like ICAR or your state agriculture department."
        }
        return fallback_responses.get(specialist, "I'm currently unable to process your request. Please try again later.")

# Endpoint
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        # Log the RAG process
        print(f"RAG Process: Starting query processing for: {request.query}")
        
        # Step 1: Generate embeddings
        print("RAG Process: Generating embeddings for query...")
        results = search_text(request.query)
        
        if not results:
            print("RAG Process: No relevant documents found in knowledge base")
            return {"answer": "I couldn't find relevant information in my knowledge base.", "sources": []}

        # Step 2: Format context
        print("RAG Process: Formatting context from retrieved documents")
        context, source_info = format_context(results)
        
        # Step 3: Generate answer
        print("RAG Process: Generating answer with LLM")
        answer = generate_answer(
            request.query, 
            context, 
            request.history,
            request.specialist,
            request.weather_info,
            request.language
        )
        
        print("RAG Process: Completed successfully")
        return {"answer": answer, "sources": source_info}
    except Exception as e:
        print(f"RAG Process: Error occurred - {str(e)}")
        # Fallback response if RAG fails
        fallback_responses = {
            "ASA – Policy Specialist": "For policy-related queries, please visit the official PM-KISAN portal at https://pmkisan.gov.in",
            "ASB – Agronomy Specialist": "For agronomy advice, contact your local agriculture extension officer or visit https://farmer.gov.in",
            "ASC – Fact Checker": "To verify agricultural information, please consult certified sources like ICAR or your state agriculture department."
        }
        fallback_response = fallback_responses.get(request.specialist, "I'm sorry, I encountered an error while processing your request. Please try again later.")
        return {"answer": fallback_response, "sources": []}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Agricultural Assistant API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
