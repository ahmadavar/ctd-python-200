import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

os.makedirs("outputs", exist_ok=True)

# the file uses semicolons as separators not commas, need sep=";" in read_csv

# Task 1 - Load and Explore

df = pd.read_csv("student_performance_math.csv", sep=";")
print("Shape:", df.shape)
print(df.head())
print(df.dtypes)

plt.figure()
plt.hist(df["G3"], bins=21, edgecolor="black")
plt.title("Distribution of Final Math Grades")
plt.xlabel("G3 (Final Grade)")
plt.ylabel("Count")
plt.savefig("outputs/g3_distribution.png")
plt.close()

# Task 2 - Preprocess

print("\nShape before filtering:", df.shape)
df_clean = df[df["G3"] > 0].copy()
print("Shape after filtering:", df_clean.shape)
# G3=0 means the student didn't show up for the final, not that they actually scored 0
# keeping those rows would confuse the model since they're missing data not real grades

binary_cols = ["schoolsup", "internet", "higher", "activities"]
for col in binary_cols:
    df_clean[col] = (df_clean[col] == "yes").astype(int)

df_clean["sex"] = (df_clean["sex"] == "M").astype(int)

corr_before = df["absences"].corr(df["G3"])
corr_after = df_clean["absences"].corr(df_clean["G3"])
print("\nabsences-G3 correlation (original):", corr_before)
print("absences-G3 correlation (filtered):", corr_after)
# interesting - the correlation changes a lot after filtering
# students with G3=0 had high absences so they were dragging the correlation down
# once we remove them the real relationship shows up

# Task 3 - EDA

numeric_features = ["age", "Medu", "Fedu", "traveltime", "studytime",
                    "failures", "absences", "freetime", "goout", "Walc"]

correlations = df_clean[numeric_features + ["G3"]].corr()["G3"].drop("G3").sort_values()
print("\nCorrelations with G3:")
print(correlations)

# plot 1: failures vs G3
plt.figure()
plt.scatter(df_clean["failures"], df_clean["G3"], alpha=0.4)
plt.title("Past Failures vs Final Grade")
plt.xlabel("failures")
plt.ylabel("G3")
plt.savefig("outputs/failures_vs_g3.png")
plt.close()
# students with more past failures tend to score lower, makes sense

# plot 2: studytime vs G3
plt.figure()
df_clean.boxplot(column="G3", by="studytime")
plt.title("G3 by Study Time")
plt.suptitle("")
plt.xlabel("Study Time (1=low, 4=high)")
plt.ylabel("G3")
plt.savefig("outputs/studytime_vs_g3.png")
plt.close()
# slight upward trend but lots of variance - study time helps but isn't the whole story

# Task 4 - Baseline Model

X_base = df_clean[["failures"]].values
y = df_clean["G3"].values

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_base, y, test_size=0.2, random_state=42)

model_base = LinearRegression()
model_base.fit(X_train_b, y_train_b)
y_pred_b = model_base.predict(X_test_b)

rmse_b = np.sqrt(mean_squared_error(y_test_b, y_pred_b))
r2_b = model_base.score(X_test_b, y_test_b)

print("\nBaseline (failures only)")
print("Slope:", model_base.coef_[0])
print("RMSE:", rmse_b)
print("R2:", r2_b)
# RMSE around 3 on a 0-20 scale means we're off by about 3 grade points on average
# R2 is low which isn't surprising - failures alone can't explain everything

# Task 5 - Full Model

feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]

X_full = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_full, y, test_size=0.2, random_state=42)

model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)
y_pred_f = model_full.predict(X_test_f)

rmse_f = np.sqrt(mean_squared_error(y_test_f, y_pred_f))
r2_train = model_full.score(X_train_f, y_train_f)
r2_test = model_full.score(X_test_f, y_test_f)

print("\nFull Model")
print("Train R2:", r2_train)
print("Test R2:", r2_test)
print("RMSE:", rmse_f)
print("baseline R2:", r2_b, "full model R2:", r2_test)

print("\nCoefficients:")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name}: {coef:.3f}")

# train and test R2 are close so no overfitting
# schoolsup is negative which looks weird but makes sense - school gives support to struggling students
# so it's picking up "this student was already behind" not "support is bad"
# for a real deployment i'd probably drop freetime, activities, traveltime - very low signal

# Task 6 - Evaluate

plt.figure()
plt.scatter(y_pred_f, y_test_f, alpha=0.6)
min_val = min(y_pred_f.min(), y_test_f.min())
max_val = max(y_pred_f.max(), y_test_f.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect fit")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")
plt.legend()
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()
# points above the line = underpredicted, below = overpredicted
# errors look roughly even across grade levels, no obvious pattern at the extremes

# Bonus - Adding G1

X_g1 = df_clean[feature_cols + ["G1"]].values
y = df_clean["G3"].values

X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_g1, y, test_size=0.2, random_state=42)

model_g1 = LinearRegression()
model_g1.fit(X_train_g, y_train_g)
print("\nWith G1 - Test R2:", model_g1.score(X_test_g, y_test_g))
# R2 jumps a lot when we add G1 - makes sense since G1 and G3 are from the same student
# but this isn't useful for early intervention since G1 only exists after the first exam
# if you wanted to catch struggling students early you'd have to rely on the other features
