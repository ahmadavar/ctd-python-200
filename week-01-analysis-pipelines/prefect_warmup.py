# prefect_warmup.py — Pipeline Q2 (Prefect)
# Python 200 — Code the Dream

from prefect import task, flow, get_run_logger
import numpy as np
import pandas as pd

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task
def create_series(arr):
    logger = get_run_logger()
    series = pd.Series(arr, name="values")
    logger.info("Series created")
    return series

@task
def clean_data(series):
    logger = get_run_logger()
    cleaned = series.dropna()
    logger.info(f"Cleaned data: {len(cleaned)} values remaining")
    return cleaned

@task
def summarize_data(series):
    logger = get_run_logger()
    summary = {
        "mean":   series.mean(),
        "median": series.median(),
        "std":    series.std(),
        "mode":   series.mode()[0]
    }
    for key, value in summary.items():
        logger.info(f"{key}: {value}")
    return summary

@flow
def pipeline_flow():
    series  = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)
    return summary

if __name__ == "__main__":
    pipeline_flow()


# ── Reflection ───────────────────────────────────────────────────
# Using prefect for this task might seem pointless but for the larger projects
# this is an amazing tool for engineers to see error before stakeholders get wrong data
