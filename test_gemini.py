from google import genai
from dotenv import load_dotenv
import os
load_dotenv()

client = genai.Client(
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location="us-central1"
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say: Chatmaxxing is live"
)
print(response.text)
