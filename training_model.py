import pandas as pd
import numpy as np
from urllib.parse import urlparse
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Load dataset ---
# NOTE: Replace with the actual path to your dataset if needed.
# file_path = '/mnt/data/Balanced Phishing Dataset.csv'
file_path = 'Balanced Phishing Dataset.csv' 
df = pd.read_csv(file_path)

# --- Feature engineering ---
def shannon_entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * np.log2(count/lns) for count in p.values())

def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    
    return {
        "num_dots": url.count('.'),
        "num_hyphens": url.count('-'),
        "num_at": url.count('@'),
        "subdomain_depth": domain.count('.') - 1 if domain else 0,
        "url_entropy": shannon_entropy(url)
    }

extra_features = df['URL'].apply(extract_features).apply(pd.Series)
df = pd.concat([df, extra_features], axis=1)

# --- Select features ---
X = df[['domain_age_days', 'https_present', 'url_length', 'has_suspicious_keyword',
        'num_dots', 'num_hyphens', 'num_at', 'subdomain_depth', 'url_entropy']]
y = df['Label']

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- XGBoost ---
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_clf.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_clf.predict(X_test_scaled)

xgb_acc = accuracy_score(y_test, y_pred_xgb)
xgb_report = classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_)

print("XGBoost Accuracy:", xgb_acc)
print("--------------------------------------------------")
print(xgb_report)
