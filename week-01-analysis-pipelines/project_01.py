# project_01.py — World Happiness Pipeline
# Python 200 — Code the Dream

from prefect import task, flow, get_run_logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

data_dir = os.path.expanduser("~/ctd-school-repo/assignments/resources/happiness_project")
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)


# ── Task 1: Load Data ────────────────────────────────────────────
@task(retries=3, retry_delay_seconds=2)
def load_data(data_dir):
    logger = get_run_logger()
    dfs = []
    for year in range(2015, 2025):
        path = f"{data_dir}/world_happiness_{year}.csv"
        df = pd.read_csv(path, sep=";", decimal=",")
        df["year"] = year
        dfs.append(df)
        logger.info(f"Loaded {year}: {len(df)} rows")

    combined = pd.concat(dfs, ignore_index=True)
    combined.columns = [c.strip().lower().replace(" ", "_") for c in combined.columns]
    combined.to_csv(f"{output_dir}/merged_happiness.csv", index=False)
    logger.info(f"Merged dataset shape: {combined.shape}")
    return combined


# ── Task 2: Descriptive Stats ────────────────────────────────────
@task
def descriptive_stats(df):
    logger = get_run_logger()
    col = "happiness_score"

    logger.info(f"Mean:   {df[col].mean():.3f}")
    logger.info(f"Median: {df[col].median():.3f}")
    logger.info(f"Std:    {df[col].std():.3f}")

    by_year = df.groupby("year")[col].mean()
    logger.info(f"By year:\n{by_year.to_string()}")

    by_region = df.groupby("regional_indicator")[col].mean().sort_values(ascending=False)
    logger.info(f"By region:\n{by_region.to_string()}")

    return by_region


# ── Task 3: Visualizations ───────────────────────────────────────
@task
def visual_exploration(df):
    logger = get_run_logger()
    col = "happiness_score"

    plt.figure()
    df[col].hist(bins=20)
    plt.title("Happiness Score Distribution")
    plt.xlabel("Happiness Score")
    plt.ylabel("Count")
    plt.savefig(f"{output_dir}/happiness_histogram.png")
    plt.close()
    logger.info("Saved happiness_histogram.png")

    plt.figure(figsize=(12, 6))
    df.boxplot(column=col, by="year")
    plt.title("Happiness by Year")
    plt.suptitle("")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    plt.savefig(f"{output_dir}/happiness_by_year.png")
    plt.close()
    logger.info("Saved happiness_by_year.png")

    plt.figure()
    plt.scatter(df["gdp_per_capita"], df[col], alpha=0.3)
    plt.title("GDP vs Happiness")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")
    plt.savefig(f"{output_dir}/gdp_vs_happiness.png")
    plt.close()
    logger.info("Saved gdp_vs_happiness.png")

    numeric = df.select_dtypes(include=np.number).drop(columns=["ranking", "year"], errors="ignore")
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()
    logger.info("Saved correlation_heatmap.png")


# ── Task 4: Hypothesis Testing ───────────────────────────────────
@task
def hypothesis_testing(df):
    logger = get_run_logger()
    col = "happiness_score"

    pre  = df[df["year"] == 2019][col].dropna()
    post = df[df["year"] == 2020][col].dropna()
    t, p = stats.ttest_ind(pre, post)

    logger.info(f"2019 vs 2020 — t={t:.4f}, p={p:.4f}")
    logger.info(f"2019 mean: {pre.mean():.3f} | 2020 mean: {post.mean():.3f}")
    if p < 0.05:
        logger.info("Significant: happiness dropped meaningfully at pandemic onset")
    else:
        logger.info("Not significant: no meaningful change between 2019 and 2020")

    r1 = df[df["regional_indicator"] == "Western Europe"][col].dropna()
    r2 = df[df["regional_indicator"] == "Sub-Saharan Africa"][col].dropna()
    t2, p2 = stats.ttest_ind(r1, r2)
    logger.info(f"Western Europe vs Sub-Saharan Africa — t={t2:.4f}, p={p2:.6f}")
    if p2 < 0.05:
        logger.info("Significant: Western Europe scores significantly higher than Sub-Saharan Africa")


# ── Task 5: Correlation + Bonferroni ────────────────────────────
@task
def correlation_analysis(df):
    logger = get_run_logger()
    col = "happiness_score"

    explanatory = [
        "gdp_per_capita", "social_support", "healthy_life_expectancy",
        "freedom_to_make_life_choices", "generosity", "perceptions_of_corruption"
    ]

    results = []
    for var in explanatory:
        clean = df[[col, var]].dropna()
        r, p = stats.pearsonr(clean[col], clean[var])
        results.append((var, r, p))
        logger.info(f"{var}: r={r:.3f}, p={p:.6f}")

    adjusted_alpha = 0.05 / len(results)
    logger.info(f"Bonferroni adjusted alpha: {adjusted_alpha:.4f}")

    logger.info("--- Significant at original alpha = 0.05 ---")
    for var, r, p in results:
        if p < 0.05:
            logger.info(f"  {var}: r={r:.3f} (p={p:.6f})")

    logger.info("--- Significant after Bonferroni correction ---")
    bonferroni_significant = []
    for var, r, p in results:
        if p < adjusted_alpha:
            logger.info(f"  {var}: r={r:.3f} (p={p:.6f})")
            bonferroni_significant.append((var, r, p))
        else:
            logger.info(f"  {var}: NOT significant after correction")

    strongest = max(bonferroni_significant, key=lambda x: abs(x[1]))
    return strongest[0]


# ── Task 6: Summary Report ───────────────────────────────────────
@task
def summary_report(df, by_region, strongest_var):
    logger = get_run_logger()
    col = "happiness_score"

    logger.info(f"Total countries: {df['country'].nunique()}")
    logger.info(f"Total years: {df['year'].nunique()}")
    logger.info(f"Top 3 regions:    {list(by_region.head(3).index)}")
    logger.info(f"Bottom 3 regions: {list(by_region.tail(3).index)}")

    pre  = df[df["year"] == 2019][col].dropna()
    post = df[df["year"] == 2020][col].dropna()
    _, p = stats.ttest_ind(pre, post)
    if p < 0.05:
        logger.info("2019 vs 2020: happiness dropped significantly at pandemic onset")
    else:
        logger.info("2019 vs 2020: no statistically significant change at pandemic onset")

    logger.info(f"Strongest predictor of happiness (Bonferroni): {strongest_var}")


# ── Flow ─────────────────────────────────────────────────────────
@flow
def happiness_pipeline():
    df        = load_data(data_dir)
    by_region = descriptive_stats(df)
    visual_exploration(df)
    hypothesis_testing(df)
    strongest = correlation_analysis(df)
    summary_report(df, by_region, strongest)


if __name__ == "__main__":
    happiness_pipeline()
