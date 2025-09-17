import pandas as pd
import numpy as np
import os
import json
from urllib.parse import urlparse
from collections import Counter
from datetime import datetime
import whois
import tldextract
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import uniform, randint
from tqdm import tqdm

# Caching WHOIS data for efficiency 
WHOIS_CACHE_FILE = 'whois_cache.json'

def load_whois_cache():
    if os.path.exists(WHOIS_CACHE_FILE):
        with open(WHOIS_CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_whois_cache(cache):
    with open(WHOIS_CACHE_FILE, 'w') as f:
        json.dump(cache, f)

# Load dataset 
file_path = 'Balanced Phishing Dataset.csv'
df = pd.read_csv(file_path)

# Feature Engineering Functions 

def shannon_entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count / lns * np.log2(count / lns) for count in p.values())

def extract_advanced_url_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path

    tld_data = tldextract.extract(url)

    features = {
        "url_length": len(url),
        "domain_length": len(domain),
        "path_length": len(path),
        "num_dots": url.count('.'),
        "num_hyphens": url.count('-'),
        "num_at": url.count('@'),
        "subdomain_depth": len(tld_data.subdomain.split('.')) if tld_data.subdomain else 0,
        "url_entropy": shannon_entropy(url),
        "num_digits": sum(c.isdigit() for c in url),
        "is_ip_address": int(domain.replace('.', '').isdigit())
    }
    return features

def extract_whois_features(domain, cache):
    if domain in cache:
        return cache[domain]

    if not domain:
        domain_age_days = -1
    else:
        try:
            whois_info = whois.whois(domain)
            if whois_info.creation_date:
                if isinstance(whois_info.creation_date, list):
                    creation_date = whois_info.creation_date[0]
                else:
                    creation_date = whois_info.creation_date
                
                domain_age_days = (datetime.now() - creation_date).days
            else:
                domain_age_days = -1
        except Exception:
            domain_age_days = -1
    
    cache[domain] = {"domain_age_days": domain_age_days}
    return cache[domain]

# Apply Feature Engineering 
print("Applying URL feature extraction...")
url_features = df['URL'].apply(extract_advanced_url_features).apply(pd.Series)
df = pd.concat([df, url_features], axis=1)

# Apply WHOIS features with caching
print("Extracting WHOIS features with caching (this may take some time on the first run)...")
whois_cache = load_whois_cache()
df['simplified_domain'] = df['URL'].apply(lambda x: urlparse(x).netloc)

# Create a dictionary for quick lookups of whois features
whois_features_dict = {
    domain: extract_whois_features(domain, whois_cache)['domain_age_days']
    for domain in tqdm(df['simplified_domain'].unique())
}

# Map the features back to the main DataFrame
df['domain_age_days'] = df['simplified_domain'].map(whois_features_dict)

save_whois_cache(whois_cache)

# Drop the temporary simplified_domain column
df = df.drop(columns=['simplified_domain'])


#  Select and prepare features

X = df[['domain_age_days', 'https_present', 'url_length', 'domain_length', 
        'path_length', 'num_dots', 'num_hyphens', 'num_at', 
        'subdomain_depth', 'url_entropy', 'num_digits', 'is_ip_address',
        'has_suspicious_keyword']]
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

# LightGBM with Optimised Hyperparameter Tuning 
print("\nStarting optimized hyperparameter tuning...")
param_dist = {
    'n_estimators': randint(100, 300),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 8),
    'num_leaves': randint(20, 50),
    'subsample': uniform(0.6, 0.4)
}

lgbm_clf = LGBMClassifier(random_state=42, verbose=-1, force_row_wise=True)

random_search = RandomizedSearchCV(
    estimator=lgbm_clf,
    param_distributions=param_dist,
    n_iter=15,
    scoring='accuracy',
    cv=2,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_scaled, y_train)

best_lgbm_clf = random_search.best_estimator_
y_pred_lgbm = best_lgbm_clf.predict(X_test_scaled)

lgbm_acc = accuracy_score(y_test, y_pred_lgbm)
lgbm_report = classification_report(y_test, y_pred_lgbm, target_names=label_encoder.classes_)

print("Best Parameters:", random_search.best_params_)
print("Best Cross-validated Accuracy:", random_search.best_score_)
print("\nFinal LightGBM Accuracy (with full optimization):", lgbm_acc)
print("--------------------------------------------------")
print(lgbm_report)
