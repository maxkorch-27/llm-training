import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================
# 1. LOAD DATA
# ==============================================================

train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'class', 'level'
]

print("Loading data...")

df_train = pd.read_csv(train_url, names=columns)
df_test = pd.read_csv(test_url, names=columns)

df_train.drop(columns=['level'], inplace=True)
df_test.drop(columns=['level'], inplace=True)

print(f"Training set: {df_train.shape[0]} records")
print(f"Test set:     {df_test.shape[0]} records")

# ==============================================================
# 2. ENCODE CATEGORICAL FEATURES
# ==============================================================

df_full = pd.concat([df_train, df_test], ignore_index=True)

cat_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df_full[col] = le.fit_transform(df_full[col])
    label_encoders[col] = le

# ==============================================================
# 3. MAP ATTACKS TO 5 CATEGORIES
# ==============================================================

category_map = {
    'normal': 'Normal',

    # DoS
    'neptune': 'DoS',
    'back': 'DoS',
    'land': 'DoS',
    'pod': 'DoS',
    'smurf': 'DoS',
    'teardrop': 'DoS',
    'mailbomb': 'DoS',
    'apache2': 'DoS',
    'processtable': 'DoS',
    'udpstorm': 'DoS',
    'worm': 'DoS',

    # Probe
    'satan': 'Probe',
    'ipsweep': 'Probe',
    'nmap': 'Probe',
    'portsweep': 'Probe',
    'mscan': 'Probe',
    'saint': 'Probe',

    # R2L
    'warezclient': 'R2L',
    'guess_passwd': 'R2L',
    'ftp_write': 'R2L',
    'imap': 'R2L',
    'phf': 'R2L',
    'multihop': 'R2L',
    'warezmaster': 'R2L',
    'spy': 'R2L',
    'xlock': 'R2L',
    'xsnoop': 'R2L',
    'snmpguess': 'R2L',
    'snmpgetattack': 'R2L',
    'httptunnel': 'R2L',
    'sendmail': 'R2L',
    'named': 'R2L',

    # U2R
    'buffer_overflow': 'U2R',
    'loadmodule': 'U2R',
    'rootkit': 'U2R',
    'perl': 'U2R',
    'sqlattack': 'U2R',
    'xterm': 'U2R',
    'ps': 'U2R'
}

df_full['category'] = df_full['class'].map(category_map)

other_count = df_full['category'].isna().sum()
if other_count > 0:
    print(f"Warning: {other_count} rows were not mapped. Dropping them.")
    df_full = df_full.dropna(subset=['category']).copy()

# ==============================================================
# 4. PREPARE FEATURES AND LABELS
# ==============================================================

df_full.drop(columns=['num_outbound_cmds', 'class'], inplace=True)

train_len = len(df_train)
df_train_processed = df_full.iloc[:train_len].copy()
df_test_processed = df_full.iloc[train_len:].copy()

X_train = df_train_processed.drop(columns=['category'])
y_train = df_train_processed['category']

X_test = df_test_processed.drop(columns=['category'])
y_test = df_test_processed['category']

print(f"\nFeatures: {X_train.shape[1]}")
print("\nTraining set class distribution:")
print(y_train.value_counts())
print("\nTest set class distribution:")
print(y_test.value_counts())

# ==============================================================
# 4.1 FEATURE ENGINEERING
# ==============================================================
X_train = X_train.copy()
X_test = X_test.copy()

# Ratio feature
X_train["bytes_ratio"] = (X_train["src_bytes"] + 1) / (X_train["dst_bytes"] + 1)
X_test["bytes_ratio"] = (X_test["src_bytes"] + 1) / (X_test["dst_bytes"] + 1)

# Total traffic
X_train["total_bytes"] = X_train["src_bytes"] + X_train["dst_bytes"]
X_test["total_bytes"] = X_test["src_bytes"] + X_test["dst_bytes"]

# Error intensity
X_train["error_rate"] = X_train["serror_rate"] + X_train["rerror_rate"]
X_test["error_rate"] = X_test["serror_rate"] + X_test["rerror_rate"]

# Clean any inf / nan values
X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)

print(f"\nFeatures: {X_train.shape[1]}")
print("\nTraining set class distribution:")
print(y_train.value_counts())
print("\nTest set class distribution:")
print(y_test.value_counts())


# ==============================================================
# 5. ENCODE TARGET LABELS FOR XGBOOST AND CV
# ==============================================================

label_encoder_y = LabelEncoder()
y_train_encoded = label_encoder_y.fit_transform(y_train)
y_test_encoded = label_encoder_y.transform(y_test)

# ==============================================================
# 6. CROSS-VALIDATION
# ==============================================================

cv_model = Pipeline(steps=[
    ('smote', SMOTE(random_state=42)),
    ('xgb', XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=5,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    cv_model,
    X_train,
    y_train_encoded,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1
)

print(f"\nCross-validation macro F1: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

# ==============================================================
# 7. SMOTE BALANCING FOR FINAL TRAINING
# ==============================================================

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train_encoded)

print("\nAfter SMOTE balancing:")
print(pd.Series(y_resampled).value_counts())

# ==============================================================
# 8. MODEL
# ==============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=5,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

model.fit(X_resampled, y_resampled)

# ==============================================================
# 9. PREDICTION
# ==============================================================

y_pred_encoded = model.predict(X_test)
y_pred = label_encoder_y.inverse_transform(y_pred_encoded.astype(int))

# ==============================================================
# 10. EVALUATION
# ==============================================================

print("\nClassification report:")
print(classification_report(y_test, y_pred))

macro_f1 = f1_score(y_test, y_pred, average='macro')
print("Macro F1:", macro_f1)

# ==============================================================
# 11. CONFUSION MATRIX
# ==============================================================

labels = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(9, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - SMOTE + XGBoost + Engineering +CV\nMacro F1: {macro_f1:.4f}\nCV macro F1: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")
plt.tight_layout()
plt.savefig("confusion_matrix_smote_xgb_engineering_cv.png", dpi=200)

print("\nConfusion matrix saved as confusion_matrix_smote_xgb_engineering_cv.png")

