import os
from dotenv import load_dotenv
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.gemini import Gemini 

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("Google API Key is missing. Please check your .env file.")

# Ensure Gemini embedding is used
gemini_embed_model = GeminiEmbedding(model_name="models/text-embedding-004")

def query_index(query):
    try:
        # Load index correctly with Gemini embeddings
        storage_context = StorageContext.from_defaults(persist_dir="Data/index_store")
        index = load_index_from_storage(storage_context, embed_model=gemini_embed_model)
        
        # Ensure LLM is NOT OpenAI (disable by setting llm=None)
        query_engine = index.as_query_engine(llm=Gemini(model_name="models/gemini-1.5-pro-latest"))
        
        return query_engine.query(query)
    
    except Exception as e:
        print("Error querying index:", e)
        return None
