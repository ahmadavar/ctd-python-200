# project_09.py — Week 9: Extract + Load Pipeline
#
# Video Walkthrough (Written):
# 1. Terminal: run `python project_09.py` — output shows bytes uploaded to
#    raw/2026-05-30/weather.json, lists blobs in the container, prints first 5
#    rows of the hourly weather DataFrame, and confirms file saved to outputs/.
# 2. Portal: navigate to Storage account p200ahmadavar → Containers →
#    pipeline-data → raw/2026-05-30/ → click weather.json to confirm upload.
# 3. Terminal: show the DataFrame printout with time, temperature_2m, precipitation columns.
#
# Pipeline: Open-Meteo API → JSON bytes → Azure Blob Storage
# Run: az login   (must be done before executing this script)

import json
import requests
import pandas as pd
from datetime import date
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

# =============================================================================
# Config — fill in your storage account name
# =============================================================================
ACCOUNT_URL = "https://p200ahmadavar.blob.core.windows.net"
CONTAINER = "pipeline-data"

# Charlotte, NC coordinates
LATITUDE = 35.2271
LONGITUDE = -80.8431


# =============================================================================
# Step 1: Extract
# =============================================================================
def extract() -> dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&hourly=temperature_2m,precipitation&forecast_days=7"
    )
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


# =============================================================================
# Step 2: Serialize
# =============================================================================
def serialize(data: dict) -> bytes:
    return json.dumps(data).encode("utf-8")


# =============================================================================
# Step 3: Load
# =============================================================================
def load(container_client, payload: bytes) -> str:
    blob_path = f"raw/{date.today().isoformat()}/weather.json"
    blob_client = container_client.get_blob_client(blob_path)
    blob_client.upload_blob(payload, overwrite=True)
    print(f"Uploaded {len(payload)} bytes → {blob_path}")
    return blob_path


# =============================================================================
# Step 4: Verify
# =============================================================================
def verify(container_client) -> None:
    print("\nBlobs in container:")
    for blob in container_client.list_blobs():
        print(f"  {blob.name}  ({blob.size} bytes)")


# =============================================================================
# Step 5: Read Back
# =============================================================================
def read_back(container_client, blob_path: str) -> None:
    blob_client = container_client.get_blob_client(blob_path)
    raw_bytes = blob_client.download_blob().readall()

    # Save locally for mentor inspection
    output_path = "outputs/weather_raw.json"
    with open(output_path, "wb") as f:
        f.write(raw_bytes)
    print(f"\nSaved to {output_path}")

    # Parse and print first 5 rows
    data = json.loads(raw_bytes)
    df = pd.DataFrame(data["hourly"])
    print("\nFirst 5 rows of hourly weather data:")
    print(df.head())


# =============================================================================
# Main
# =============================================================================
def main():
    credential = DefaultAzureCredential()
    service_client = BlobServiceClient(account_url=ACCOUNT_URL, credential=credential)
    container_client = service_client.get_container_client(CONTAINER)

    data = extract()
    payload = serialize(data)
    blob_path = load(container_client, payload)
    verify(container_client)
    read_back(container_client, blob_path)


if __name__ == "__main__":
    main()
