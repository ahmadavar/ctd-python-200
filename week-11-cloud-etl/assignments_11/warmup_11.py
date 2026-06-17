# --- Prefect Orchestration ---

# Q1
# @task is for operations that can fail independently — API calls, file reads,
# database writes, blob uploads. Prefect tracks their state, retries them on
# failure, and logs them individually.
#
# A pure in-memory calculation like Celsius → Fahrenheit has no I/O and no
# failure modes worth tracking. Wrapping it in @task adds Prefect overhead
# (state tracking, API writes, serialization) for zero benefit. Call it as a
# plain Python function from inside a task instead.

# Q2
# Decorator for a task named call_api that retries 3 times with 30s delay:
#
# @task(name="call_api", retries=3, retry_delay_seconds=30)

# Q3
# extract = Completed, transform = Failed, load = never ran.
# Where to look: click the failed transform task in the Prefect UI, open its Logs tab.
# What to expect: the Python traceback — error type, message, and the exact line
# number where it failed. load never ran because Prefect blocks downstream tasks
# when an upstream task fails.

# --- Production Patterns ---

# Q1
# raise_for_status() raises an HTTPError exception immediately if the response
# is 4xx or 5xx, stopping execution right at the API call.
#
# print("error") does nothing to stop the pipeline — execution continues with
# a bad or empty response, either crashing later with a confusing error or
# silently producing wrong output.
#
# In a Prefect task, raise_for_status() causes the task to fail immediately,
# which blocks all downstream tasks from running. The error points directly
# at the API call that broke — not somewhere downstream.

# Q2
# Without overwrite=True, re-running after a crash throws a ResourceExistsError —
# the blob from the failed run is still there and Azure refuses to overwrite it.
# You would have to manually delete the corrupted blob before retrying.
#
# overwrite=True lets the re-run replace whatever is there with the correct
# complete output. The pipeline is idempotent — safe to re-run any number of
# times without manual cleanup.

# Q3
from prefect import task
from prefect.logging import get_run_logger

@task
def load_records(records: list, blob_path: str):
    logger = get_run_logger()
    logger.info(f"Loaded {len(records)} records to {blob_path}")
