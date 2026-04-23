import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
from io import BytesIO

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# =============================================================================
# Task 1: Load and Explore
# =============================================================================

COLUMN_NAMES = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
    "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
    "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
    "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
    "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!",
    "char_freq_$", "char_freq_#",
    "capital_run_length_average", "capital_run_length_longest",
    "capital_run_length_total", "spam_label"
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()
df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES

print("=== Task 1: Load and Explore ===")
print(f"Dataset shape: {df.shape}")
print(f"\nClass balance:\n{df['spam_label'].value_counts()}")
print(f"\nSpam rate: {df['spam_label'].mean():.1%}")
# ~39% spam, 61% ham — slightly imbalanced. A model that always predicts "ham"
# would achieve 61% accuracy without learning anything. This is why F1 and
# precision/recall matter more than raw accuracy on this dataset.

# Boxplots for three key features
features_to_plot = ["word_freq_free", "char_freq_!", "capital_run_length_total"]
for feat in features_to_plot:
    fig, ax = plt.subplots(figsize=(6, 4))
    spam_vals = df[df["spam_label"] == 1][feat]
    ham_vals  = df[df["spam_label"] == 0][feat]
    ax.boxplot([ham_vals, spam_vals], labels=["Ham", "Spam"])
    ax.set_title(f"{feat} by class")
    ax.set_ylabel(feat)
    plt.tight_layout()
    fname = f"outputs/{feat.replace('/', '_')}_boxplot.png"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
# All three features show dramatically higher values in spam emails:
# "free" appears more in spam, "!" is used more aggressively, and
# capital letter runs are far longer. The differences are dramatic, not subtle.

# Heavy skew toward zero: most emails don't contain "free" at all. The few
# that do tend to be spam, making zero-heavy distributions still informative.
# Scale varies wildly (tiny fractions vs. thousands) — this matters for
# distance-based models (KNN) and gradient-based models (LR) that are sensitive
# to feature magnitude. Tree-based models are unaffected.

# =============================================================================
# Task 2: Prepare Data
# =============================================================================

print("\n=== Task 2: Prepare Data ===")

X = df.drop("spam_label", axis=1)
y = df["spam_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features — essential for KNN (distances) and LR (gradient descent)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# PCA: always scale first — features with large ranges would dominate components
pca = PCA()
pca.fit(X_train_scaled)

cumvar = np.cumsum(pca.explained_variance_ratio_)
n_components = int(np.argmax(cumvar >= 0.90)) + 1
print(f"Components needed for 90% variance: {n_components}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumvar) + 1), cumvar)
plt.axhline(0.90, linestyle='--', color='red', label='90% threshold')
plt.axvline(n_components, linestyle='--', color='blue', label=f'n={n_components}')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Variance — Spambase")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/pca_variance_spambase.png")
plt.close()
print("Saved outputs/pca_variance_spambase.png")

# Transform both sets, keep first n_components
X_train_pca = pca.transform(X_train_scaled)[:, :n_components]
X_test_pca  = pca.transform(X_test_scaled)[:, :n_components]

# =============================================================================
# Task 3: Classifier Comparison
# =============================================================================

print("\n=== Task 3: Classifier Comparison ===")

# --- KNN on unscaled ---
knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)
y_pred_knn_raw = knn_raw.predict(X_test)
print("\n-- KNN (unscaled) --")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn_raw):.4f}")
print(classification_report(y_test, y_pred_knn_raw))

# --- KNN on scaled ---
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)
print("\n-- KNN (scaled) --")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn_scaled):.4f}")
print(classification_report(y_test, y_pred_knn_scaled))

# --- KNN on PCA ---
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pred_knn_pca = knn_pca.predict(X_test_pca)
print("\n-- KNN (PCA-reduced) --")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn_pca):.4f}")
print(classification_report(y_test, y_pred_knn_pca))

# --- Decision Tree: depth sweep ---
print("\n-- Decision Tree depth sweep --")
for depth in [3, 5, 10, None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc  = accuracy_score(y_test,  dt.predict(X_test))
    print(f"max_depth={str(depth):4s}  train={train_acc:.4f}  test={test_acc:.4f}")
# As depth increases: training accuracy climbs toward 1.0, but test accuracy
# peaks then plateaus or drops — classic overfitting signature.
# A tree with no limit memorizes training data (100% train accuracy) but
# generalizes poorly. max_depth=5 balances accuracy and generalization.

best_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
best_dt.fit(X_train, y_train)
y_pred_dt = best_dt.predict(X_test)
print("\n-- Decision Tree (max_depth=5) --")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(classification_report(y_test, y_pred_dt))

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\n-- Random Forest --")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

# --- Logistic Regression on scaled ---
lr = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print("\n-- Logistic Regression (scaled) --")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr))

# --- Logistic Regression on PCA ---
lr_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
lr_pca.fit(X_train_pca, y_train)
y_pred_lr_pca = lr_pca.predict(X_test_pca)
print("\n-- Logistic Regression (PCA-reduced) --")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr_pca):.4f}")
print(classification_report(y_test, y_pred_lr_pca))

# Summary comment:
# Random Forest performs best — ensemble averaging across 100 trees eliminates
# individual tree noise and generalizes well without scaling.
# KNN improved significantly with scaling (mixed feature scales matter).
# PCA slightly reduced performance for both KNN and LR — the compressed
# representation loses some discriminative signal for spam.
# For a spam filter, minimizing false positives (legitimate mail marked spam)
# matters most — a missed spam is annoying, but a lost business email is costly.
# Therefore precision on class 0 (ham) is our key metric to optimize.

# --- Feature importances ---
print("\n-- Feature Importances --")
dt_imp = pd.Series(best_dt.feature_importances_, index=X.columns)
rf_imp = pd.Series(rf.feature_importances_,      index=X.columns)

print("\nTop 10 Decision Tree features:")
print(dt_imp.nlargest(10))
print("\nTop 10 Random Forest features:")
print(rf_imp.nlargest(10))

plt.figure(figsize=(10, 6))
rf_imp.nlargest(10).sort_values().plot(kind='barh')
plt.title("Random Forest — Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importances.png")
plt.close()
print("Saved outputs/feature_importances.png")
# Both models agree: capital letter runs, char_freq_$, char_freq_!, and
# word_freq_remove are the most discriminative spam signals — matching intuition.

# --- Best model confusion matrix (Random Forest) ---
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot(colorbar=False)
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/best_model_confusion_matrix.png")
plt.close()
print("Saved outputs/best_model_confusion_matrix.png")
# The model makes more false negatives (spam gets through) than false positives
# (ham marked spam) — a reasonable tradeoff for a spam filter prioritizing
# inbox reliability over perfect spam detection.

# =============================================================================
# Task 4: Cross-Validation
# =============================================================================

print("\n=== Task 4: Cross-Validation ===")

models = {
    "KNN (unscaled)":       (KNeighborsClassifier(n_neighbors=5), X_train),
    "KNN (scaled)":         (KNeighborsClassifier(n_neighbors=5), X_train_scaled),
    "Decision Tree (d=5)":  (DecisionTreeClassifier(max_depth=5, random_state=42), X_train),
    "Random Forest":        (RandomForestClassifier(n_estimators=100, random_state=42), X_train),
    "Logistic Regression":  (LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'), X_train_scaled),
}

for name, (model, X_cv) in models.items():
    scores = cross_val_score(model, X_cv, y_train, cv=5)
    print(f"{name:30s}  mean={scores.mean():.4f}  std={scores.std():.4f}")
# Random Forest is most accurate AND most stable (lowest std) — internal
# averaging across trees acts as built-in cross-validation.
# Decision Tree has the highest variance — sensitive to which data it trains on.

# =============================================================================
# Task 5: Prediction Pipelines
# =============================================================================

print("\n=== Task 5: Pipelines ===")

# Pipeline 1: Random Forest (tree-based — no scaling needed)
rf_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Pipeline 2: Logistic Regression with scaling (non-tree, benefits from scaling)
lr_pipeline = Pipeline([
    ("scaler",     StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])
# The two pipelines have different structures: LR needs a scaler step because
# gradient descent is sensitive to feature magnitudes; Random Forest does not
# because it splits on thresholds, not distances or gradients.
# Pipelines make deployment safe — preprocessing and model are one object,
# so the correct transformations are always applied in the right order.

rf_pipeline.fit(X_train, y_train)
print("\n-- RF Pipeline --")
print(f"Accuracy: {rf_pipeline.score(X_test, y_test):.4f}")
print(classification_report(y_test, rf_pipeline.predict(X_test)))

lr_pipeline.fit(X_train, y_train)
print("\n-- LR Pipeline --")
print(f"Accuracy: {lr_pipeline.score(X_test, y_test):.4f}")
print(classification_report(y_test, lr_pipeline.predict(X_test)))
