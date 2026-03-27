import os
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Small 1x1 black pixel base64
pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QBwADogHCwjSxcgAAAABJRU5ErkJggg=="

try:
    print("Testing llama-3.3-70b-versatile for vision...")
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
        model="llama-3.3-70b-versatile",
    )
    print("SUCCESS: llama-3.3-70b-versatile supports vision!")
    print(chat_completion.choices[0].message.content)
except Exception as e:
    print(f"FAILED: llama-3.3-70b-versatile does not support vision. Error: {e}")

try:
    print("\nListing all vision models found in internal list search...")
    # I'll try a brute force of common names
    for m in ["llama-3.2-11b-vision", "llama-3.2-90b-vision", "llama-3.2-11b-vision-p"]:
         try:
             client.chat.completions.create(messages=[{"role":"user","content":"test"}], model=m)
             print(f"FOUND: {m}")
         except:
             pass
except:
    pass
