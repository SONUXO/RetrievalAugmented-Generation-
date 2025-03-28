from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding
from modules.data_loader import load_documents
from modules.model_api import load_model

def create_embedding():
    """Creates vector embeddings from uploaded documents (PDF, TXT, CSV)."""
    try:
        documents = load_documents()
        model = load_model()

        if documents is None or model is None:
            print("Error: Missing documents or model.")
            return None

        # Create embedding model
        gemini_embed_model = GeminiEmbedding(model_name="models/text-embedding-004")
        
        # Chunk documents for better embeddings
        node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=20)

        # Create index
        index = VectorStoreIndex.from_documents(
            documents,
            llm=model,
            embed_model=gemini_embed_model,
            node_parser=node_parser
        )

        # Save index for querying
        index.storage_context.persist(persist_dir="Data/index_store")

        print("Embeddings created successfully!")
        return index
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None
