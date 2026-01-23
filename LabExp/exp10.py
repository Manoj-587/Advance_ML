from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 1. Load inbuilt dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Define models and hyperparameters (AutoML-style search space)
models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=500),
        "params": {
            "model__C": [0.01, 0.1, 1, 10]
        }
    },
    "Support Vector Machine": {
        "model": SVC(),
        "params": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {
            "model__n_estimators": [50, 100],
            "model__max_depth": [None, 5, 10]
        }
    }
}

best_model = None
best_score = 0
best_name = ""

# 4. Auto model selection + hyperparameter tuning
for name, config in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", config["model"])
    ])

    grid = GridSearchCV(
        pipeline,
        config["params"],
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model = grid.best_estimator_
        best_name = name

# 5. Evaluate best model
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 6. Print results
print("Best Selected Model:", best_name)
print("\nModel Performance on Test Set:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")