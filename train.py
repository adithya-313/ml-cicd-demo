from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ==========================
# 1. Load Dataset
# ==========================

iris = load_iris()

X = iris.data
y = iris.target

# ==========================
# 2. Train/Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# 3. Train Model
# ==========================

model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# 4. Evaluate Model
# ==========================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"Accuracy: {accuracy:.4f}")

# ==========================
# 5. Model Validation Gate
# ==========================

MIN_ACCURACY = 0.90

if accuracy < MIN_ACCURACY:
    raise Exception(
        f"Model accuracy {accuracy:.4f} is below required threshold {MIN_ACCURACY}"
    )

print("Model passed validation gate.")

# ==========================
# 6. Save Model Artifact
# ==========================

joblib.dump(
    model,
    "model/model.joblib"
)

print("Model saved successfully.")