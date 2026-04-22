import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

os.makedirs("outputs", exist_ok=True)

# scikit-learn API

# Q1
years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)

pred_4 = model.predict([[4]])[0]
pred_8 = model.predict([[8]])[0]

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Salary at 4yrs:", pred_4)
print("Salary at 8yrs:", pred_8)

# Q2
x = np.array([10, 20, 30, 40, 50])
print("\n1D shape:", x.shape)
x_2d = x.reshape(-1, 1)
print("2D shape:", x_2d.shape)
# sklearn needs X to be 2D because models can have multiple features - rows are samples, columns are features
# even with one feature it still expects that (n_samples, n_features) format

# Q3
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)

print("\nCluster centers:\n", kmeans.cluster_centers_)
print("Points per cluster:", np.bincount(labels))

plt.figure()
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap="viridis", alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c="black", marker="X", s=200, label="Centers")
plt.title("KMeans Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.savefig("outputs/kmeans_clusters.png")
plt.close()

# Linear Regression

np.random.seed(42)
num_patients = 100
age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Q1
plt.figure()
plt.scatter(age, cost, c=smoker, cmap="coolwarm", alpha=0.7)
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost ($)")
plt.savefig("outputs/cost_vs_age.png")
plt.close()
# two clear bands - smokers sit way higher at every age, so smoker status matters a lot

# Q2
X_age = age.reshape(-1, 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_age, cost, test_size=0.2, random_state=42)
print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q3
model_age = LinearRegression()
model_age.fit(X_train, y_train)
y_pred = model_age.predict(X_test)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model_age.score(X_test, y_test)

print("\nSlope:", model_age.coef_[0])
print("Intercept:", model_age.intercept_)
print("RMSE:", rmse)
print("R2:", r2)
# slope is ~200 meaning each extra year of age adds about $200 to medical cost

# Q4
X_full = np.column_stack([age, smoker])
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_full, cost, test_size=0.2, random_state=42)

model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)

print("\nAge-only R2:", r2)
print("Full model R2:", model_full.score(X_test_f, y_test_f))
print("age coefficient:", model_full.coef_[0])
print("smoker coefficient:", model_full.coef_[1])
# smoker adds ~$15k to cost which makes sense given how the data was generated
# R2 jumps a lot when we add smoker - it was clearly missing from the first model

# Q5
y_pred_f = model_full.predict(X_test_f)

plt.figure()
plt.scatter(y_pred_f, y_test_f, alpha=0.7)
min_val = min(y_pred_f.min(), y_test_f.min())
max_val = max(y_pred_f.max(), y_test_f.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect fit")
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost ($)")
plt.ylabel("Actual Cost ($)")
plt.legend()
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()
# points above the line = model underpredicted, below = overpredicted
