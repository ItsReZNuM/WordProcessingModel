import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ===== تنظیمات =====
FILE_PATH = "dataset.xlsx"
TEXT_COL = "data"
LABEL_COL = "label"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ===== داده =====
df = pd.read_excel(FILE_PATH)
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
df = df[df[TEXT_COL].ne("") & df[TEXT_COL].ne("nan")].reset_index(drop=True)

X = df[TEXT_COL]
le = LabelEncoder()
y = le.fit_transform(df[LABEL_COL])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ===== TF-IDF =====
# برای RandomForest بهتره تعداد ویژگی‌ها کمتر باشه تا سریع‌تر/کم‌حافظه‌تر اجرا شه
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=20000,
    min_df=2
)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# ===== مدل: Random Forest =====
model = RandomForestClassifier(
    n_estimators=400,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced_subsample",
    max_depth=None,          # می‌تونی 40/60 هم تست کنی
    min_samples_split=5,
    min_samples_leaf=2
)

model.fit(X_train_vec, y_train)

# ===== ارزیابی =====
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print("\n=== Random Forest ===")
print("Accuracy:", round(acc, 4))

for avg in ["macro", "weighted", "micro"]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=avg, zero_division=0
    )
    print(f"{avg.capitalize():9s} -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
