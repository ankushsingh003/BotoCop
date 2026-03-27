import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QBwADogHCwjSxcgAAAABJRU5ErkJggg=="

try:
    print("Testing groq/compound for vision...")
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
        model="groq/compound",
    )
    print("SUCCESS: groq/compound supports vision!")
except Exception as e:
    print(f"FAILED: groq/compound does not support vision. Error: {e}")
