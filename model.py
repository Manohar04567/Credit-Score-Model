# 📌 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from google.colab import files

# 📌 2. Upload CSV file
uploaded = files.upload()  # Upload 'Project_Filtered_test.csv'

# 📌 3. Load dataset
file_name = next(iter(uploaded))
df = pd.read_csv(file_name)

# 📌 4. Create binary target from 'Credit_Mix'
df = df[df['Credit_Mix'].notnull()].copy()
df['Target'] = df['Credit_Mix'].apply(lambda x: 1 if x == 'Good' else 0)

# 📌 5. Drop irrelevant columns
drop_cols = ['ID', 'Customer_ID', 'Name', 'SSN', 'Type_of_Loan', 'Credit_Mix', 'Payment_Behaviour']
df.drop(columns=drop_cols, inplace=True)

# 📌 6. Separate numeric and categorical
categorical_cols = ['Occupation', 'Payment_of_Min_Amount']
numeric_cols = [col for col in df.columns if col not in categorical_cols + ['Target']]

# 📌 7. Impute missing values in numeric columns
imputer = SimpleImputer(strategy="mean")
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# 📌 8. Encode categorical columns
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# 📌 9. Split features and target
X = df.drop(columns=["Target"])
y = df["Target"]

# 📌 10. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 11. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 📌 12. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 📌 13. Evaluate
y_pred = model.predict(X_test_scaled)
print("🔍 Classification Report:\n", classification_report(y_test, y_pred))
print(f"🎯 ROC-AUC Score: {roc_auc_score(y_test, y_pred):.4f}")

# 📊 14. Plot: Target Distribution
plt.figure(figsize=(6,4))
sns.countplot(x=y, palette="Set2")
plt.title("Credit Mix Distribution (Target Variable)")
plt.xlabel("Target (1 = Good, 0 = Others)")
plt.ylabel("Count")
plt.grid(True)
plt.show()

# 📊 15. Plot: Feature Importances
importances = model.feature_importances_
feature_names = X.columns

feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_df = feat_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_df, palette="viridis")
plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.grid(True)
plt.tight_layout()
plt.show()