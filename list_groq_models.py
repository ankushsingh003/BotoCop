import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
try:
    models = client.models.list()
    print("Available Groq Models:")
    for model in models.data:
        if "vision" in model.id.lower():
            print(f" - {model.id} (Vision Supported)")
        else:
            print(f" - {model.id}")
except Exception as e:
    print(f"Error listing models: {e}")
