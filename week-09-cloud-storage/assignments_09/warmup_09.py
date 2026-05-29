# warmup_09.py — Week 9: Cloud Storage
# Azure Authentication + Blob Storage exercises

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient

# =============================================================================
# --- Azure Authentication ---
# =============================================================================

# Q1
# When a Python script runs locally and uses DefaultAzureCredential, it cannot
# type a password like a human. Instead, DefaultAzureCredential tries a chain
# of authentication methods automatically, in order, until one succeeds.
#
# For local development, the relevant method is the Azure CLI credential.
# When you run `az login` in your terminal, Azure saves a short-lived token
# to your local machine (~/.azure/). DefaultAzureCredential finds that token
# and uses it — no credentials are hardcoded in the script.
#
# The command you must run first: az login
#
# DefaultAzureCredential "knows" to use it because the CLI credential is one
# of the built-in steps in the chain. It checks whether a valid az login
# session exists; if yes, it uses it. If not, it moves on to the next method
# in the chain (environment variables, managed identity, etc.).

# Q2
# A deployed pipeline running on an Azure VM or container cannot use `az login`
# because there is no human present to open a browser and complete the login
# flow. `az login` is an interactive command — it requires a person.
#
# Instead, the resource is assigned a Managed Identity by Azure. This is an
# identity attached to the resource itself (the VM, container, function app),
# not to a human. Azure handles the credential exchange internally — there is
# no password or token to manage or rotate.
#
# The same Python code works without changes because DefaultAzureCredential
# tries Managed Identity automatically when it runs in an Azure-hosted
# environment. In local dev, the CLI credential step succeeds first. In
# production, that step fails (no az login session) and the chain moves on
# to Managed Identity, which succeeds. The script never needs to know which
# environment it is in.

# Q3
# Two most likely causes of AuthenticationError immediately after creating
# DefaultAzureCredential:
#
# Cause 1: You never ran `az login` (or the session expired).
#   DefaultAzureCredential tries the CLI credential and finds no valid token.
#   Diagnosis: run `az account show` in your terminal. If it fails or returns
#   an error, run `az login` and try again. Tokens typically expire in ~1 hour.
#
# Cause 2: The azure-identity package is not installed, or the wrong version
#   is installed, so the credential class cannot load properly.
#   Diagnosis: run `pip show azure-identity` to confirm it is installed and
#   check the version. If missing, run:
#   uv pip install azure-identity azure-mgmt-resource azure-storage-blob


# =============================================================================
# --- Blob Storage ---
# =============================================================================

# Q1
# Azure Blob Storage has three levels:
#
# 1. Storage Account — the top-level container, like an entire filing cabinet.
#    It has a globally unique name and holds everything below it.
#    Analogy: the filing cabinet itself.
#
# 2. Container — a named grouping inside the account, like a drawer in the
#    filing cabinet. You might have one drawer for raw data, one for processed
#    outputs, one for model artifacts.
#    Analogy: a drawer in the filing cabinet.
#
# 3. Blob — the actual file stored inside a container, like a folder or
#    document inside a drawer. It can be any file type: CSV, JSON, image, etc.
#    Analogy: a single document in the drawer.
#
# Full path: storage-account → container → blob
# (like: filing-cabinet → drawer → document)

# Q2
# Scenario A: A REST API returns a JSON payload each hour. Store raw responses
#   for reprocessing later.
#   → Blob Storage. Raw files with no structured query needs are a perfect fit
#     for blob storage — cheap, durable, and simple to append new files hourly.
#
# Scenario B: 50 million customer transactions queried daily by date range and
#   customer ID.
#   → Relational database (Azure SQL). The data is structured and queried with
#     filters — a database with indexes will be orders of magnitude faster than
#     scanning blobs.
#
# Scenario C: Image embeddings as NumPy arrays saved between pipeline runs.
#   → Blob Storage. Binary array files (e.g., .npy) are unstructured file
#     artifacts — blob storage is the right place to persist them cheaply
#     between runs.


def list_container(container_client: ContainerClient) -> None:
    # Q3: Print name and size of every blob in the container
    for blob in container_client.list_blobs():
        print(f"{blob.name}  {blob.size} bytes")


def upload_text(container_client: ContainerClient, blob_name: str, text: str) -> None:
    # Q4: Encode string as UTF-8 and upload, overwriting any existing blob
    data = text.encode("utf-8")
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(data, overwrite=True)
