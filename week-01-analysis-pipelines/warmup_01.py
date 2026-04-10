# warmup_01.py — Week 1 Warmup Exercises
# Python 200 — Code the Dream

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ── Pandas ───────────────────────────────────────────────────────

# Q1: Create a DataFrame
data = {
    "name":  ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "age":   [25, 30, 35, 28, 22],
    "score": [88, 92, 79, 95, 85]
}
df = pd.DataFrame(data)
print(df)

# Q2: Select a column
print(df["name"])

# Q3: Filter rows where score > 85
print(df[df["score"] > 85])

# Q4: Add a new column
df["passed"] = df["score"] >= 80
print(df)

# Q5: Sort by age descending
print(df.sort_values("age", ascending=False))

# Q6: Compute mean of score
print(df["score"].mean())

# Q7: Group by passed and compute mean score
print(df.groupby("passed")["score"].mean())


# ── NumPy ────────────────────────────────────────────────────────

# Q1: Create a 1D array of 10 evenly spaced values between 0 and 1
arr = np.linspace(0, 1, 10)
print(arr)

# Q2: Create a 3x3 matrix of zeros
matrix = np.zeros((3, 3))
print(matrix)

# Q3: Compute mean, sum, and standard deviation
data_arr = np.array([4, 7, 13, 2, 1])
print(np.mean(data_arr))
print(np.sum(data_arr))
print(np.std(data_arr))

# Q4: Multiply every element by 2
print(data_arr * 2)

# Q5: Boolean mask — elements greater than 5
print(data_arr[data_arr > 5])

# Q6: Reshape a 1D array of 12 elements into a 3x4 matrix
arr12 = np.arange(12)
print(arr12.reshape(3, 4))


# ── Matplotlib ───────────────────────────────────────────────────

# Q1: Line plot
x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 8, 5]
plt.plot(x, y)
plt.title("Line Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Q2: Bar chart
categories = ["A", "B", "C", "D"]
values = [3, 7, 5, 9]
plt.bar(categories, values)
plt.title("Bar Chart")
plt.xlabel("Category")
plt.ylabel("Value")
plt.show()

# Q3: Scatter plot
x3 = [1, 2, 3, 4, 5]
y3 = [5, 3, 8, 1, 7]
plt.scatter(x3, y3)
plt.title("Scatter Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Q4: Histogram
data_hist = np.random.normal(0, 1, 1000)
plt.hist(data_hist, bins=30)
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


# ── Descriptive Statistics ───────────────────────────────────────

data_stats = [23, 45, 67, 12, 89, 34, 56, 78, 90, 11,
              23, 45, 67, 34, 56, 78, 45, 67, 89, 23]

# Q1: Mean, median, mode
print("Mean:",   np.mean(data_stats))
print("Median:", np.median(data_stats))
print("Mode:",   stats.mode(data_stats)[0])

# Q2: Variance and standard deviation
print("Variance:", np.var(data_stats))
print("Std Dev:",  np.std(data_stats))

# Q3: Min, max, range
print("Min:",   np.min(data_stats))
print("Max:",   np.max(data_stats))
print("Range:", np.max(data_stats) - np.min(data_stats))

# Q4: Percentiles
print("25th percentile:", np.percentile(data_stats, 25))
print("75th percentile:", np.percentile(data_stats, 75))
print("IQR:", np.percentile(data_stats, 75) - np.percentile(data_stats, 25))

# Q5: Histogram of data_stats
plt.hist(data_stats, bins=10)
plt.title("Distribution of Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


# ── Hypothesis Testing ───────────────────────────────────────────

group_a = [78, 82, 79, 85, 83, 80, 77, 81, 84, 76]
group_b = [88, 91, 87, 93, 90, 89, 86, 92, 94, 85]

# Q1: Independent samples t-test
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print("t-statistic:", t_stat)
print("p-value:", p_value)

# Q2: Paired t-test
before = [70, 75, 68, 80, 72, 74, 71, 73, 69, 77]
after  = [75, 80, 74, 85, 78, 79, 76, 78, 74, 82]
t_stat2, p_value2 = stats.ttest_rel(before, after)
print("t-statistic:", t_stat2)
print("p-value:", p_value2)

# Q3: One-sample t-test vs benchmark
scores = [72, 68, 75, 80, 65, 70, 78, 82, 71, 69]
t_stat3, p_value3 = stats.ttest_1samp(scores, 65)
print("t-statistic:", t_stat3)
print("p-value:", p_value3)

# Q4: One-sample t-test vs benchmark 70
t_stat4, p_value4 = stats.ttest_1samp(scores, 70)
print("p-value:", p_value4)
# p = 0.127 > 0.05 — not statistically significant
# Scores are not meaningfully different from the national benchmark of 70

# Q5: One-tailed test — is Group A lower than Group B?
t_stat5, p_value5 = stats.ttest_ind(group_a, group_b, alternative='less')
print("t-statistic:", t_stat5)
print("p-value:", p_value5)
# p << 0.05 — Group A scores significantly lower than Group B

# Q6: Plain language conclusion
conclusion = """
- The class average is not significantly different from the national benchmark of 70 (p = 0.127).
- Group A scores significantly lower than Group B (p = 0.0000008).
- The difference between groups is real and unlikely due to chance.
"""
print(conclusion)


# ── Correlation ──────────────────────────────────────────────────

# Q1: corrcoef
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
corr_matrix = np.corrcoef(x, y)
print("Correlation matrix:\n", corr_matrix)
print("Correlation coefficient:", corr_matrix[0, 1])
# Expected: 1.0 — perfect positive correlation because y = 2x exactly

# Q2: pearsonr
from scipy.stats import pearsonr
x2 = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y2 = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]
corr, p_val = pearsonr(x2, y2)
print("Correlation coefficient:", corr)
print("p-value:", p_val)

# Q3: df.corr()
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df_people = pd.DataFrame(people)
print("Correlation matrix:\n", df_people.corr())

# Q4: Scatter plot — negative relationship
x4 = [10, 20, 30, 40, 50]
y4 = [90, 75, 60, 45, 30]
plt.scatter(x4, y4)
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Q5: Heatmap
import seaborn as sns
sns.heatmap(df_people.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# ── Pipeline Q1 ──────────────────────────────────────────────────

arr_pipe = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    return pd.Series(arr, name="values")

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    return {
        "mean":   series.mean(),
        "median": series.median(),
        "std":    series.std(),
        "mode":   series.mode()[0]
    }

def data_pipeline(arr):
    series  = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)
    return summary

result = data_pipeline(arr_pipe)
for key, value in result.items():
    print(key, value)
