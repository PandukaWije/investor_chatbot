from gemini_parser import DocumentProcessor
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize processor with API key
processor = DocumentProcessor(api_key=os.getenv("GEMINI_API_KEY"))


result = processor.process_folder(Path("startup nation decks"))

print(result)