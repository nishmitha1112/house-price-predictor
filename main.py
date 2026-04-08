import pandas as pd

# STEP 1: load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# STEP 2: check missing (optional)
print(train.isnull().sum().sort_values(ascending=False).head(10))

# STEP 3: drop columns
cols_to_drop = ["PoolQC", "MiscFeature", "Alley", "Fence"]

train = train.drop(cols_to_drop, axis=1, errors='ignore')
test = test.drop(cols_to_drop, axis=1, errors='ignore')

# STEP 4: fill missing values

# train
num_cols = train.select_dtypes(include=['int64', 'float64']).columns
train[num_cols] = train[num_cols].fillna(train[num_cols].median())

cat_cols = train.select_dtypes(include=['object']).columns
train[cat_cols] = train[cat_cols].fillna("None")

# test
num_cols_test = test.select_dtypes(include=['int64', 'float64']).columns
test[num_cols_test] = test[num_cols_test].fillna(test[num_cols_test].median())

cat_cols_test = test.select_dtypes(include=['object']).columns
test[cat_cols_test] = test[cat_cols_test].fillna("None")

# STEP 5: convert categorical → numbers
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# STEP 6: USE ONLY 3 FEATURES (IMPORTANT FIX)
X = train[["OverallQual", "GrLivArea", "GarageCars"]]
y = train["SalePrice"]

# STEP 7: match test columns
test = test.reindex(columns=X.columns, fill_value=0)

# ==============================
# 🔥 MODEL PART
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# STEP 8: define model
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    random_state=42
)

# STEP 9: validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

preds = model.predict(X_val)

mae = mean_absolute_error(y_val, preds)
print("MAE:", mae)

# STEP 10: train on full data
model.fit(X, y)

# STEP 11: feature importance
importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop Features:\n")
print(importance.sort_values(ascending=False))

# STEP 12: predict
predictions = model.predict(test)

# STEP 13: save output
test_ids = pd.read_csv("test.csv")["Id"]

output = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": predictions
})

output.to_csv("submission.csv", index=False)

# STEP 14: save model
import joblib
joblib.dump(model, "model.pkl")

print("\n✅ DONE: submission.csv + model.pkl created")