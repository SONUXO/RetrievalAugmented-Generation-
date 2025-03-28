import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

# Load API key from .env
load_dotenv()

def load_documents():
    """Loads all documents from the 'Data' directory, including PDFs, TXTs, and CSVs."""
    try:
        data_path = os.path.join(os.getcwd(), "Data")

        # Ensure Data folder exists
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        # Read TXT and PDF files
        documents = SimpleDirectoryReader(data_path).load_data()

        # Read CSV files
        for file in os.listdir(data_path):
            if file.endswith(".csv"):
                csv_path = os.path.join(data_path, file)
                df = pd.read_csv(csv_path)
                text = df.to_string()
                documents.append(Document(text=text))

        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return None
