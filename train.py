from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

csv_files = list(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise SystemExit("ERROR: No CSV found. Put your dataset inside the data folder.")

csv_path = max(csv_files, key=lambda p: p.stat().st_size)
print(f"Dataset: {csv_path}")
print("Loading a sample so a large CSV can be processed with less memory...")

# Use up to 150,000 rows for the initial training run. Increase later if your computer has enough RAM.
try:
    df = pd.read_csv(csv_path, nrows=150000, low_memory=False)
except Exception as e:
    raise SystemExit(f"Could not read CSV: {e}")

df.columns = [str(c).strip() for c in df.columns]
print(f"Rows loaded: {len(df):,}")
print("Columns:", list(df.columns))

def find_col(candidates):
    lookup = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for name in candidates:
        key = name.lower().replace(" ", "").replace("_", "")
        if key in lookup:
            return lookup[key]
    return None

# Prefer departure-delay labels. If unavailable, use arrival-delay labels.
target_col = find_col(["DEP_DEL15", "DepDel15", "DepartureDelay15", "DepDelay15",
                       "ArrDel15", "ARR_DEL15"])
target_type = "existing_binary"
if target_col is None:
    dep_minutes = find_col(["DepDelayMinutes", "DepartureDelayMinutes", "DepDelay"])
    arr_minutes = find_col(["ArrDelayMinutes", "ArrivalDelayMinutes", "ArrDelay"])
    if dep_minutes:
        target_col = dep_minutes
        target_type = "minutes"
    elif arr_minutes:
        target_col = arr_minutes
        target_type = "minutes"
    else:
        raise SystemExit(
            "\nI could not automatically identify a delay target column.\n"
            "Run: python inspect_data.py\n"
            "Then send me the printed column names so I can adapt the code to your dataset."
        )

print(f"Target column detected: {target_col}")
y_raw = df[target_col]
if target_type == "minutes":
    y = (pd.to_numeric(y_raw, errors="coerce") > 15).astype("float")
    y[y_raw.isna()] = np.nan
else:
    y = pd.to_numeric(y_raw, errors="coerce")

valid = y.notna() & y.isin([0, 1])
df = df.loc[valid].copy()
y = y.loc[valid].astype(int)
if y.nunique() < 2:
    raise SystemExit("Target has only one class in the sample. Try a different sample or dataset.")

# Remove target and columns that reveal the outcome after the flight has operated.
leak_names = {
    "arrdelay", "arrdelayminutes", "arrdel15", "arrdelaygroup",
    "depdelay", "depdelayminutes", "depdel15", "depdelaygroup",
    "cancelled", "cancellationcode", "diverted", "wheelsontime",
    "wheelsont", "taxiout", "taxiin", "actualelapsedtime",
    "airtime", "arrivaldelay", "departuredelay"
}
features = df.drop(columns=[target_col], errors="ignore")
drop_cols = []
for c in features.columns:
    norm = c.lower().replace("_", "").replace(" ", "")
    if norm in {x.replace("_", "").replace(" ", "") for x in leak_names}:
        drop_cols.append(c)
features = features.drop(columns=drop_cols, errors="ignore")

# Drop columns that are almost entirely empty, unique identifiers, and text fields that are too high-cardinality.
for c in list(features.columns):
    if features[c].isna().mean() > 0.95:
        features = features.drop(columns=[c])
        continue
    if features[c].nunique(dropna=True) > min(5000, max(1000, int(len(features) * 0.5))):
        if features[c].dtype == "object":
            features = features.drop(columns=[c])

# Convert date-like fields into simple numeric date/time features when possible.
for c in list(features.columns):
    if "date" in c.lower():
        parsed = pd.to_datetime(features[c], errors="coerce")
        if parsed.notna().mean() > 0.5:
            features[c + "_month"] = parsed.dt.month
            features[c + "_dayofweek"] = parsed.dt.dayofweek
            features = features.drop(columns=[c])

# Avoid excessively large one-hot matrices by retaining only manageable columns.
if features.shape[1] == 0:
    raise SystemExit("No usable feature columns remained after cleaning.")

# Keep the feature set modest for a beginner laptop.
if features.shape[1] > 35:
    features = features.iloc[:, :35]

# Convert numeric-looking columns to numeric when most values parse successfully.
for c in features.columns:
    if features[c].dtype == "object":
        converted = pd.to_numeric(features[c], errors="coerce")
        if converted.notna().mean() > 0.8:
            features[c] = converted

numeric_cols = features.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_cols = [c for c in features.columns if c not in numeric_cols]

# Limit category levels to avoid giant encoded matrices.
for c in categorical_cols:
    top = features[c].astype(str).value_counts().head(30).index
    features[c] = features[c].astype(str).where(features[c].astype(str).isin(top), "Other")

# Split before fitting preprocessing, avoiding leakage.
X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, random_state=42, stratify=y
)

try:
    onehot = OneHotEncoder(handle_unknown="ignore", min_frequency=2)
except TypeError:
    onehot = OneHotEncoder(handle_unknown="ignore")

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", onehot)
])
transformers = []
if numeric_cols:
    transformers.append(("num", numeric_pipe, numeric_cols))
if categorical_cols:
    transformers.append(("cat", categorical_pipe, categorical_cols))
preprocessor = ColumnTransformer(transformers=transformers)

models = {
    "Logistic Regression": Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=120, random_state=42, class_weight="balanced_subsample",
            n_jobs=-1, min_samples_leaf=2
        ))
    ])
}

# Small grid keeps the first run reasonable on a typical student laptop.
grids = {
    "Logistic Regression": {"model__C": [0.1, 1.0]},
    "Random Forest": {"model__max_depth": [10, None], "model__min_samples_leaf": [2]}
}

results = {}
best_name, best_search, best_auc = None, None, -1
class_counts = y_train.value_counts()
cv_splits = min(3, int(class_counts.min()))
if cv_splits < 2:
    raise SystemExit("Not enough examples of both classes for cross-validation.")

for name, pipeline in models.items():
    print(f"\nTraining {name} with {cv_splits}-fold cross-validation and Grid Search...")
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    search = GridSearchCV(
        pipeline, grids[name], scoring="roc_auc", cv=cv,
        n_jobs=1, refit=True, verbose=1, error_score="raise"
    )
    search.fit(X_train, y_train)
    pred = search.predict(X_test)
    try:
        proba = search.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
    except Exception:
        auc = float("nan")
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": auc,
        "best_params": search.best_params_,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, zero_division=0, output_dict=True)
    }
    results[name] = metrics
    print(f"{name}: Accuracy={metrics['accuracy']:.3f}, Precision={metrics['precision']:.3f}, "
          f"Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}, ROC-AUC={auc:.3f}")
    if np.isfinite(auc) and auc > best_auc:
        best_name, best_search, best_auc = name, search, auc

if best_search is None:
    best_name = max(results, key=lambda n: results[n]["f1"])
    best_search = models[best_name].fit(X_train, y_train)

bundle = {
    "pipeline": best_search.best_estimator_ if hasattr(best_search, "best_estimator_") else best_search,
    "feature_columns": list(features.columns),
    "numeric_columns": numeric_cols,
    "categorical_columns": categorical_cols,
    "feature_defaults": {
        c: (float(pd.to_numeric(features[c], errors="coerce").median())
            if c in numeric_cols and pd.to_numeric(features[c], errors="coerce").notna().any()
            else (0.0 if c in numeric_cols else str(features[c].mode().iloc[0] if not features[c].mode().empty else "Unknown")))
        for c in features.columns
    },
    "target_column": target_col,
    "target_description": "Delay risk: 1 means delayed more than 15 minutes; 0 means not delayed more than 15 minutes.",
    "best_model_name": best_name
}
joblib.dump(bundle, MODEL_DIR / "flight_delay_model.joblib")

with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump({"best_model": best_name, "results": results, "rows_used": int(len(features)),
               "target_column": target_col}, f, indent=2)

# Save Random Forest feature importance if available.
best_pipe = bundle["pipeline"]
try:
    prep = best_pipe.named_steps["prep"]
    model = best_pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        names = prep.get_feature_names_out()
        imp = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
        imp.sort_values("importance", ascending=False).head(25).to_csv(MODEL_DIR / "feature_importance.csv", index=False)
except Exception as e:
    print("Could not save feature importance:", e)

print("\nTRAINING COMPLETE!")
print("Best model:", best_name)
print("Saved model: models/flight_delay_model.joblib")
print("Saved metrics: models/metrics.json")
print("Rows used:", len(features))
print("\nNote: this first version trains on up to 150,000 rows for manageable memory use.")
