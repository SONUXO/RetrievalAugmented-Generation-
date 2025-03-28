import os
from dotenv import load_dotenv
import google.generativeai as genai
from llama_index.llms.gemini import Gemini

# Load API key
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configure Gemini API
genai.configure(api_key=google_api_key)

def load_model():
    """Loads the Gemini-Pro model."""
    try:
        model = Gemini(model_name="models/gemini-1.5-pro-latest", api_key=google_api_key)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
