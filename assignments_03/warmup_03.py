import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# =============================================================================
# --- Preprocessing ---
# =============================================================================

# Q1
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,       # randomize order before splitting
    stratify=y          # preserve class proportions in both splits
)

print("X_train:", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train:", y_train.shape)
print("y_test: ", y_test.shape)

# Q2
# Fit on training data only — fitting on the full dataset would leak test-set
# statistics into the model, making evaluation dishonestly optimistic
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\nColumn means of X_train_scaled:", X_train_scaled.mean(axis=0).round(10))

# =============================================================================
# --- KNN ---
# =============================================================================

# Q1 — unscaled
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)

print("\n--- KNN Q1 (unscaled) ---")
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds, target_names=iris.target_names))

# Q2 — scaled
knn.fit(X_train_scaled, y_train)
preds_scaled = knn.predict(X_test_scaled)

print("--- KNN Q2 (scaled) ---")
print("Accuracy (scaled):", accuracy_score(y_test, preds_scaled))
# Scaling slightly reduced accuracy on Iris (1.0 → 0.93) because all features
# are already in centimeters and similar ranges — distances were not distorted
# without scaling. On datasets with mixed scales, scaling would reliably help KNN.

# Q3 — cross-validation on unscaled
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn_unscaled, X_train, y_train, cv=5)

print("\n--- KNN Q3 (cross-validation, unscaled) ---")
print("Fold scores:", cv_scores)
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std:  {cv_scores.std():.3f}")
# Cross-validation is more trustworthy than a single split — it evaluates the
# model on 5 different held-out subsets and averages them, so one lucky or
# unlucky split doesn't skew the result. Low std (0.033) confirms stability.

# Q4 — k tuning
print("\n--- KNN Q4 (k tuning) ---")
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_k, X_train, y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}  std={scores.std():.3f}")
# k=7 is the best choice — same mean accuracy as k=5 (0.975) but lower std
# (0.020 vs 0.033), meaning it's more stable across different data splits.
# When two k values tie on accuracy, always prefer the more stable one.

# =============================================================================
# --- Classifier Evaluation ---
# =============================================================================

# Q1 — confusion matrix using predictions from KNN Q1 (unscaled, k=5)
knn_eval = KNeighborsClassifier(n_neighbors=5)
knn_eval.fit(X_train, y_train)
preds_eval = knn_eval.predict(X_test)

cm = confusion_matrix(y_test, preds_eval)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(colorbar=False)
plt.title("KNN Confusion Matrix (k=5, unscaled)")
plt.tight_layout()
plt.savefig("outputs/knn_confusion_matrix.png")
plt.close()
print("\n--- Classifier Evaluation Q1 ---")
print("Confusion matrix saved to outputs/knn_confusion_matrix.png")
print(cm)
# All 30 predictions correct on this split — no species are confused.
# In general, versicolor and virginica are the hardest pair to separate
# (their petal measurements overlap), but k=5 handles it cleanly here.

# =============================================================================
# --- Decision Trees ---
# =============================================================================

# Q1
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("\n--- Decision Tree Q1 ---")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(classification_report(y_test, y_pred_dt, target_names=iris.target_names))
# Decision Tree (0.967) is slightly below KNN unscaled (1.0) on this split;
# cross-validation would give a fairer comparison between the two.

# Scaling does not affect Decision Trees — they split on feature thresholds
# (e.g. petal_length > 2.45?), not distances, so rescaling changes nothing.

# =============================================================================
# --- Logistic Regression and Regularization ---
# =============================================================================

# Q1 — three C values
# Note: solver='liblinear' only supports binary classification.
# Iris has 3 classes so we use 'lbfgs', which supports multiclass natively.
print("\n--- Logistic Regression Q1 ---")
for C in [0.01, 1.0, 100]:
    model = LogisticRegression(C=C, max_iter=1000, solver='lbfgs')
    model.fit(X_train_scaled, y_train)
    total = np.abs(model.coef_).sum()
    print(f"C={C:6}: total coefficient magnitude = {total:.4f}")
# As C increases, total coefficient magnitude grows dramatically (1.7 → 13 → 41).
# Regularization (small C) keeps all weights small, preventing over-reliance on
# any single feature. Large C removes this constraint — coefficients grow freely
# and the model risks overfitting to training noise.

# =============================================================================
# --- PCA ---
# =============================================================================

digits  = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

# Q1 — shapes and sample grid
print("\n--- PCA Q1 ---")
print("X_digits shape:", X_digits.shape)
print("images shape:  ", images.shape)

fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for digit in range(10):
    idx = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[idx], cmap='gray_r')
    axes[digit].set_title(str(digit))
    axes[digit].axis('off')
plt.suptitle("One example of each digit (0-9)")
plt.tight_layout()
plt.savefig("outputs/sample_digits.png")
plt.close()
print("Saved outputs/sample_digits.png")

# Q2 — PCA 2D scatter
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)
plt.colorbar(scatter, label='Digit')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D Projection of Digits Dataset")
plt.tight_layout()
plt.savefig("outputs/pca_2d_projection.png")
plt.close()
print("Saved outputs/pca_2d_projection.png")
# Same-digit images do tend to cluster together in this 2D space, though clusters
# overlap — 2 components don't capture enough variance to fully separate all 10 digits.

# Q3 — cumulative explained variance
cumvar = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumvar) + 1), cumvar)
plt.axhline(0.80, linestyle='--', color='red', label='80% variance')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance — Digits")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/pca_variance_explained.png")
plt.close()
n_80 = np.argmax(cumvar >= 0.80) + 1
print(f"\nComponents needed for 80% variance: {n_80}")
print("Saved outputs/pca_variance_explained.png")
# Approximately 20 components are needed to explain 80% of the variance —
# a dramatic reduction from the original 64 dimensions.

# Q4 — digit reconstruction across n_components
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

n_values   = [2, 5, 15, 40]
n_samples  = 5
n_rows     = len(n_values) + 1  # +1 for original row

fig, axes = plt.subplots(n_rows, n_samples, figsize=(10, 10))

for col in range(n_samples):
    axes[0, col].imshow(images[col], cmap='gray_r')
    axes[0, col].axis('off')
axes[0, 0].set_ylabel("Original", rotation=90, size=10, labelpad=40)

for row, n in enumerate(n_values, start=1):
    for col in range(n_samples):
        recon = reconstruct_digit(col, scores, pca, n)
        axes[row, col].imshow(recon, cmap='gray_r')
        axes[row, col].axis('off')
    axes[row, 0].set_ylabel(f"n={n}", rotation=90, size=10, labelpad=40)

plt.suptitle("PCA Digit Reconstructions")
plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png")
plt.close()
print("Saved outputs/pca_reconstructions.png")
# Digits become clearly recognizable around n=15, which matches where the
# cumulative variance curve begins to level off (~80%). Beyond n=15, each
# additional component adds diminishing visual improvement.
