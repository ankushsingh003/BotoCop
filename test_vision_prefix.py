import os
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QBwADogHCwjSxcgAAAABJRU5ErkJggg=="

try:
    print("Testing meta-llama/llama-3.2-11b-vision-preview...")
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
        model="meta-llama/llama-3.2-11b-vision-preview",
    )
    print("SUCCESS: meta-llama/llama-3.2-11b-vision-preview supports vision!")
except Exception as e:
    print(f"FAILED: meta-llama/llama-3.2-11b-vision-preview does not support vision. Error: {e}")
