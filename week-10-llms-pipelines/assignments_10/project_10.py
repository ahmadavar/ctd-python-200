# Video walkthrough: https://youtu.be/sUhgQUkVJxE?si=ibOxFkIg0hSRD-NJ
#
# Reflection (Step 6):
# Classifying weather conditions for outdoor running is a borderline LLM use case.
# A rule-based approach (e.g. temperature > 10 and precipitation < 1 → good) would
# actually handle this task more reliably — the rules are simple, deterministic, and
# free. The LLM adds cost and latency without adding much value here, since there is
# no language ambiguity to resolve. What you lose with rule-based: flexibility if the
# classification logic becomes more nuanced (e.g. "feels like" temperature, wind speed,
# humidity). What you gain: zero API cost, instant results, and 100% predictable output.
# The real value of LLMs in pipelines is for tasks like classifying free-text support
# tickets or extracting fields from unstructured documents — tasks where rules break down.
#
# NOTE ON MODEL CHOICE — INTENTIONAL DEVIATION FROM SPEC:
# The assignment asks for the OpenAI API (gpt-4o-mini).
# I do not have an OpenAI API key; I am using the Anthropic key set up in Weeks 5–7.
# The anthropic client provides identical functionality: structured system prompt,
# single-word response, fallback handling. Results are equivalent in practice.
# To switch to OpenAI: replace the anthropic client block with:
#   from openai import OpenAI
#   client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#   response = client.chat.completions.create(model="gpt-4o-mini", messages=[...])
#   label = response.choices[0].message.content.strip().lower()

import os
import json
import requests
import pandas as pd
import anthropic
from datetime import date
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCOUNT_URL = os.getenv("AZURE_ACCOUNT_URL")
CONTAINER = os.getenv("AZURE_CONTAINER")
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
TODAY = date.today().isoformat()

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}

OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=37.7749&longitude=-122.4194"
    "&hourly=temperature_2m,precipitation"
    "&forecast_days=7"
    "&timezone=America%2FLos_Angeles"
)

# ---------------------------------------------------------------------------
# Azure client
# ---------------------------------------------------------------------------

blob_service = BlobServiceClient.from_connection_string(CONNECTION_STRING)
container_client = blob_service.get_container_client(CONTAINER)

# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ---------------------------------------------------------------------------
# Step 1: Read
# Since Week 9 blob data is not available for today's date, we fetch fresh
# data from the Open-Meteo API and upload it to raw/<today>/weather.json,
# then read it back — completing both the extract and load steps inline.
# ---------------------------------------------------------------------------

print(f"Fetching weather data from Open-Meteo API...")
response = requests.get(OPEN_METEO_URL)
response.raise_for_status()
raw_data = response.json()

raw_blob_path = f"raw/{TODAY}/weather.json"
raw_bytes = json.dumps(raw_data).encode("utf-8")
container_client.upload_blob(name=raw_blob_path, data=raw_bytes, overwrite=True)
print(f"Uploaded raw data to {raw_blob_path}")

downloaded = container_client.download_blob(raw_blob_path).readall()
raw_data = json.loads(downloaded.decode("utf-8"))

hourly = raw_data["hourly"]
records = [
    {
        "time": hourly["time"][i],
        "temperature_2m": hourly["temperature_2m"][i],
        "precipitation": hourly["precipitation"][i],
    }
    for i in range(len(hourly["time"]))
]

print(f"Reshaped {len(records)} hourly records")

# ---------------------------------------------------------------------------
# Step 2: Transform — classify first 24 records with Anthropic
# ---------------------------------------------------------------------------

print("Classifying weather conditions with Claude...")

for i, record in enumerate(records[:24]):
    if i % 6 == 0:
        print(f"  Processing record {i}...")

    user_message = (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )

    message = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    label = message.content[0].text.strip().lower()
    record["conditions"] = label if label in VALID_LABELS else "unknown"

enriched = records[:24]
print(f"Classification complete — {len(enriched)} records enriched")

# ---------------------------------------------------------------------------
# Step 3: Write enriched records back to blob
# ---------------------------------------------------------------------------

processed_blob_path = f"processed/{TODAY}/weather_classified.json"
processed_bytes = json.dumps(enriched).encode("utf-8")
container_client.upload_blob(name=processed_blob_path, data=processed_bytes, overwrite=True)
print(f"Uploaded enriched data to {processed_blob_path}")

# ---------------------------------------------------------------------------
# Step 4: Spot-check
# ---------------------------------------------------------------------------

downloaded_processed = container_client.download_blob(processed_blob_path).readall()
df = pd.DataFrame(json.loads(downloaded_processed.decode("utf-8")))

print("\n--- Conditions value counts ---")
print(df["conditions"].value_counts())
print("\n--- First 5 rows ---")
print(df.head())

# ---------------------------------------------------------------------------
# Step 5: Save first 10 records to outputs/
# ---------------------------------------------------------------------------

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

output_path = os.path.join(OUTPUTS_DIR, "first_10_records.json")
with open(output_path, "w") as f:
    json.dump(enriched[:10], f, indent=2)

print(f"\nSaved first 10 records to {output_path}")
print(f"\nDone. Processed blob: {processed_blob_path}")
