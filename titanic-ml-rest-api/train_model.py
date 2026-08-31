import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/titanic.csv")

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# --------------------------------------------------
# 2. Select features and target
# --------------------------------------------------

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

target = "survived"

X = df[features]
y = df[target]


# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. Define feature types
# --------------------------------------------------

numeric_features = [
    "age",
    "sibsp",
    "parch",
    "fare",
    "pclass"
]

categorical_features = [
    "sex",
    "embarked"
]


# --------------------------------------------------
# 5. Numerical preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# --------------------------------------------------
# 6. Categorical preprocessing
# --------------------------------------------------

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# --------------------------------------------------
# 7. Combine preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# --------------------------------------------------
# 8. Complete ML pipeline
# --------------------------------------------------

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


# --------------------------------------------------
# 9. Train model
# --------------------------------------------------

model_pipeline.fit(X_train, y_train)

print("Model training completed.")


# --------------------------------------------------
# 10. Evaluate
# --------------------------------------------------

predictions = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# --------------------------------------------------
# 11. Save complete pipeline
# --------------------------------------------------

joblib.dump(
    model_pipeline,
    "model/titanic_pipeline.joblib"
)

print("\nModel saved successfully.")