import os
import pandas as pd
from scipy import stats
from dotenv import load_dotenv
from smolagents import tool, CodeAgent, LiteLLMModel

load_dotenv()

DATA_PATH = os.path.expanduser(
    "~/ctd-python-200/week-01-analysis-pipelines/outputs/merged_happiness.csv"
)

# ---------------------------------------------------------------------------
# Task 1: Four @tool decorated functions
# ---------------------------------------------------------------------------

@tool
def load_happiness_data() -> str:
    """Load the World Happiness dataset and return a summary of its structure.

    Returns:
        A string describing the shape, columns, years available, and sample rows.
    """
    df = pd.read_csv(DATA_PATH)
    years = sorted(df["year"].dropna().unique().astype(int).tolist())
    summary = (
        f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
        f"Columns: {df.columns.tolist()}\n"
        f"Years available: {years}\n"
        f"Sample (first 3 rows):\n{df.head(3).to_string(index=False)}"
    )
    return summary


@tool
def summarize_column(column: str) -> str:
    """Return descriptive statistics for a numeric column in the happiness dataset.

    Args:
        column: The name of the numeric column to summarize (e.g. 'happiness_score').

    Returns:
        A string with count, mean, std, min, and max for the column.
    """
    df = pd.read_csv(DATA_PATH)
    if column not in df.columns:
        return f"Column '{column}' not found. Available: {df.columns.tolist()}"
    stats_series = df[column].describe()
    return f"Stats for '{column}':\n{stats_series.to_string()}"


@tool
def compute_correlation(column_a: str, column_b: str) -> str:
    """Compute the Pearson correlation between two numeric columns in the happiness dataset.

    Args:
        column_a: First numeric column name.
        column_b: Second numeric column name.

    Returns:
        A string with the Pearson r value and p-value.
    """
    df = pd.read_csv(DATA_PATH).dropna(subset=[column_a, column_b])
    if column_a not in df.columns:
        return f"Column '{column_a}' not found."
    if column_b not in df.columns:
        return f"Column '{column_b}' not found."
    r, p = stats.pearsonr(df[column_a], df[column_b])
    return f"Pearson r({column_a}, {column_b}) = {r:.4f}, p-value = {p:.4e}"


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> str:
    """Return the top N countries ranked by a given column for a specific year.

    Args:
        column: The numeric column to rank by (e.g. 'happiness_score').
        year: The year to filter on (e.g. 2019).
        n: Number of top countries to return.

    Returns:
        A string listing the top N countries and their values for that column and year.
    """
    df = pd.read_csv(DATA_PATH)
    filtered = df[df["year"] == year].dropna(subset=[column])
    if filtered.empty:
        return f"No data found for year {year}."
    top = filtered.nlargest(n, column)[["country", column]].reset_index(drop=True)
    return f"Top {n} countries by '{column}' in {year}:\n{top.to_string(index=False)}"


# ---------------------------------------------------------------------------
# Task 2: Build the CodeAgent
# Using LiteLLMModel with Claude Haiku instead of OpenAIServerModel —
# same swap used in Weeks 5 and 6 throughout this course.
# ---------------------------------------------------------------------------

model = LiteLLMModel(
    model_id="anthropic/claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

# ---------------------------------------------------------------------------
# Task 3: Five guided queries (reset=False retains context across turns)
# ---------------------------------------------------------------------------

queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    (
        "Plot happiness_score over the years as a line chart, with one line per region. "
        "Save the plot to outputs/happiness_by_region.png."
    ),
]

# ---------------------------------------------------------------------------
# Task 4: Custom queries
# ---------------------------------------------------------------------------

# Custom Query 1: year-over-year delta — requires the agent to write merge + subtract
# code on the fly. No tool covers cross-year comparisons, so this is pure code generation.
my_query_1 = (
    "Using the happiness dataset, find which countries improved their happiness score "
    "the most between 2015 and 2023. Load the data, filter for both years, merge on country, "
    "compute the delta (2023 score minus 2015 score), and return the top 5 most improved "
    "countries with their delta values."
)

# Custom Query 2: scatter plot with regional color grouping — requires matplotlib code
# with per-group coloring and a legend. Triggers code generation (no plotting tool exists).
my_query_2 = (
    "Create a scatter plot of happiness_score vs gdp_per_capita for the year 2023, "
    "with each point colored by regional_indicator. Add a title, axis labels, and a legend. "
    "Save the plot to outputs/happiness_gdp_scatter.png"
)


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("TASK 3: Guided Queries")
    print("#"*60)
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print('='*60)
        result = agent.run(query, reset=False)
        print(f"\nAnswer: {result}")

    print("\n" + "#"*60)
    print("TASK 4: Custom Queries")
    print("#"*60)

    print(f"\n{'='*60}")
    print(f"Custom Query 1: {my_query_1}")
    print('='*60)
    result_1 = agent.run(my_query_1, reset=False)
    print(f"\nAnswer: {result_1}")
    # Comment: triggered code generation — agent wrote pandas merge + delta logic on the fly.
    # No tool covers cross-year subtraction, so this was 100% generated code, not tool calls.

    print(f"\n{'='*60}")
    print(f"Custom Query 2: {my_query_2}")
    print('='*60)
    result_2 = agent.run(my_query_2, reset=False)
    print(f"\nAnswer: {result_2}")
    # Comment: triggered code generation — agent wrote matplotlib scatter with per-region
    # color grouping and a legend. No plotting tool exists, so this was pure code generation.


# ---------------------------------------------------------------------------
# Task 5: Reflection
# ---------------------------------------------------------------------------

# Q1: In Query 3, how did the agent communicate whether the correlation was statistically
#     significant? Did it use the p-value correctly? What threshold did it apply?
#
# The agent returned r = 0.6218 with p-value = 1.6234e-146 — essentially zero.
# It correctly flagged this as statistically significant using the standard p < 0.05
# threshold. A p-value that small means there is virtually zero chance the correlation
# is due to random chance across 1,362 observations. The agent communicated this clearly
# and applied the threshold correctly.

# Q2: Did any of the agent's responses surprise you — either by being more capable
#     than you expected, or less? Describe one specific example.
#
# Custom Query 1 was the standout. I expected the agent to struggle with a multi-step
# pandas operation it had never been shown: filter two years, merge on country, subtract
# scores, sort by delta. It nailed it in one pass — returning Slovakia (+0.48) as the
# top improver with correct delta values for all 5 countries. The capability ceiling
# was higher than expected for open-ended data wrangling.

# Q3: What one additional tool would make this agent meaningfully more useful?
#     Describe what it would do and what kind of question it would help the agent answer.
#
# A get_regional_average(column, year) tool that groups the dataset by regional_indicator
# and returns the mean of a given column per region for a specific year. This would let
# the agent answer "which region is happiest on average in 2022?" without writing groupby
# code each time — making regional comparisons faster, cleaner, and easier to audit.
