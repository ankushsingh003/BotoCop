import os
from google import genai
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("GEMINI_API_KEY is not set in environment or .env file.")
else:
    try:
        client = genai.Client(api_key=api_key)
        print("Available Gemini Models:")
        for model in client.models.list():
            print(f" - {model.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
