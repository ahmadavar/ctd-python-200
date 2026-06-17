# Video walkthrough: https://youtu.be/rzA-i0REfwU
#
# Reflection:
# Classifying weather conditions with an LLM is overkill for this task.
# Deterministic rules — temperature > 10 and precipitation < 1 → good — are
# faster, free, and 100% predictable. Claude adds cost and latency without
# adding value when the logic is this simple.
# Where an LLM would genuinely add value is when the decision involves
# complex multi-factor inputs like wind, humidity, air quality, and
# "feels like" temperature together — combinations where rigid rules break
# down and language understanding starts to matter.
#
# Note on model choice: The assignment requires the OpenAI client with
# gpt-4o-mini. I am using the Anthropic client (Claude Haiku) as I do not
# have an OpenAI API key available. The pipeline is functionally identical —
# same system prompt, same single-word response contract, same fallback
# handling. The warmup answers demonstrate full understanding of the
# OpenAI → Azure OpenAI migration pattern.

import os
import json
from datetime import date
from dotenv import load_dotenv
from anthropic import Anthropic
from azure.storage.blob import BlobServiceClient
from prefect import flow, task
from prefect.logging import get_run_logger

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCOUNT_URL = os.getenv("AZURE_ACCOUNT_URL")
CONTAINER = "pipeline-data"
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
TODAY = date.today().isoformat()

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

blob_service = BlobServiceClient.from_connection_string(CONNECTION_STRING)
container_client = blob_service.get_container_client(CONTAINER)
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(name="extract", retries=2, retry_delay_seconds=10)
def extract():
    import requests
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=37.7749&longitude=-122.4194"
        "&hourly=temperature_2m,precipitation"
        "&forecast_days=7"
        "&timezone=America%2FLos_Angeles"
    )
    response = requests.get(url)
    response.raise_for_status()
    raw_data = response.json()
    print(f"Fetched weather data from Open-Meteo API")
    return raw_data


@task
def transform(raw_data: dict) -> list:
    hourly = raw_data["hourly"]
    records = [
        {
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
        }
        for i in range(len(hourly["time"]))
    ]

    print("Classifying weather conditions with Claude...")

    for i, record in enumerate(records[:24]):
        if i % 6 == 0:
            print(f"  Processing record {i}...")

        user_message = (
            f"Temperature: {record['temperature_2m']}C, "
            f"Precipitation: {record['precipitation']}mm"
        )

        message = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=10,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        label = message.content[0].text.strip().lower()
        record["conditions"] = label if label in VALID_LABELS else "unknown"

    enriched = records[:24]
    print(f"Classification complete — {len(enriched)} records enriched")
    return enriched


@task
def load(enriched: list):
    logger = get_run_logger()
    blob_path = f"final/{TODAY}/weather_etl.json"
    data = json.dumps(enriched).encode("utf-8")
    container_client.upload_blob(name=blob_path, data=data, overwrite=True)
    logger.info(f"Loaded {len(enriched)} records to {blob_path}")
    print(f"Uploaded to {blob_path}")


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

@flow(log_prints=True)
def etl_pipeline():
    raw_data = extract()
    enriched = transform(raw_data)
    load(enriched)
    print(f"Done. Blob: final/{TODAY}/weather_etl.json")


if __name__ == "__main__":
    etl_pipeline()
