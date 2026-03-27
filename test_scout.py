import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QBwADogHCwjSxcgAAAABJRU5ErkJggg=="

try:
    print("Testing llama-4-scout for vision...")
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{pixel}",
                        },
                    },
                ],
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    print("SUCCESS: llama-4-scout supports vision!")
except Exception as e:
    print(f"FAILED: llama-4-scout does not support vision. Error: {e}")
