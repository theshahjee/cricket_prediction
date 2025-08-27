#main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from model_utils import engineer_features, get_features

# Create directories if not exist
os.makedirs('eda_plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Step 1: Load Data
train_path = 'data/cricket_dataset.csv'
test_path = 'data/cricket_dataset_test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Data Loaded:")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Step 2: EDA (Exploratory Data Analysis)
print("\nEDA Summary:")
train_df.info()
print("\nNull Values:\n", train_df.isnull().sum())
print("\nTrain Describe:\n", train_df.describe())
if 'won' in train_df.columns:
    print("\nClass Balance:\n", train_df['won'].value_counts(normalize=True))
print("\nCorrelations:\n", train_df.corr())

# Visualizations
plt.figure(figsize=(10, 6))
train_df['won'].value_counts().plot(kind='bar')
plt.title('Win/Loss Distribution')
plt.savefig('eda_plots/win_loss_dist.png')
plt.close()

plt.figure(figsize=(10, 6))
train_df.plot.scatter(x='total_runs', y='balls_left', c='won', cmap='viridis')
plt.title('Runs vs Balls Left Colored by Win')
plt.savefig('eda_plots/runs_vs_balls.png')
plt.close()

train_df.hist(figsize=(12, 10))
plt.suptitle('Feature Histograms')
plt.savefig('eda_plots/histograms.png')
plt.close()

print("EDA plots saved to 'eda_plots/' directory.")

# Step 3: Data Cleaning
# Drop rows with null values (2 rows)
train_df = train_df.dropna()
test_df = test_df.dropna()
print(f"After dropping nulls, train shape: {train_df.shape}, test shape: {test_df.shape}")

# Clip negative balls_left (already done in engineer_features, but ensure here too)
train_df['balls_left'] = train_df['balls_left'].clip(lower=0)
test_df['balls_left'] = test_df['balls_left'].clip(lower=0)

# Remove duplicates
train_df = train_df.drop_duplicates()
test_df = test_df.drop_duplicates()
print(f"After dropping duplicates, train shape: {train_df.shape}, test shape: {test_df.shape}")

# Step 4: Feature Engineering
train_df = engineer_features(train_df)
test_df = engineer_features(test_df, is_test=True)
print("\nEngineered Features (Train Head):\n", train_df.head())

# Step 5: Prepare Data for Modeling
features = get_features(train_df)
X = train_df[features]
y = train_df['won']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Model Training and Comparison
models = {
    'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
}

results = {}
best_model = None
best_auc = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    logloss = log_loss(y_val, y_prob)
    auc = roc_auc_score(y_val, y_prob)
    
    results[name] = {'Accuracy': acc, 'Log Loss': logloss, 'ROC AUC': auc}
    
    print(f"\n{name} Metrics:\nAccuracy: {acc:.4f}\nLog Loss: {logloss:.4f}\nROC AUC: {auc:.4f}")
    
    if auc > best_auc:
        best_auc = auc
        best_model = model

print("\nModel Comparison:\n", pd.DataFrame(results).T)

# Save best model and scaler
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nBest model saved as 'best_model.pkl'.")

# Step 7: Predictions on Test Data
X_test = test_df[features]
X_test_scaled = scaler.transform(X_test)
test_df['predicted_won'] = best_model.predict(X_test_scaled)
test_df['win_probability'] = best_model.predict_proba(X_test_scaled)[:, 1]

test_df.to_csv('data/predictions.csv', index=False)
print("\nPredictions saved to 'data/predictions.csv'.")