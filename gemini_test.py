from google import genai
from google.genai import types
import os
import time

# ==========================================
# CONFIGURATION
# ==========================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API KEY NOT FOUND!")
    raise SystemExit(1)

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=30000)
)

# ==========================================
# GEMINI TEST
# ==========================================

try:

    print("Connecting to Gemini...")
    start = time.time()

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Define database in one sentence.",
        config=types.GenerateContentConfig(
            max_output_tokens=100
        )
    )

    elapsed = time.time() - start

    print("\n========== GEMINI RESPONSE ==========")
    print(response.text)
    print("===================================")
    print(f"Response Time: {elapsed:.2f} seconds")

except Exception as e:

    print("\nAPI ERROR:")
    print(e)

finally:

    client.close()