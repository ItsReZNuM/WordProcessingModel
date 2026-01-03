import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
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
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# ===== TF-IDF =====
# برای MLP بهتر است تعداد feature خیلی بزرگ نباشد
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=20000,
    min_df=2,
    sublinear_tf=True
)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# ===== مدل: MLP =====
model = MLPClassifier(
    hidden_layer_sizes=(256, 128),  #  لایه مخفی
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=128,
    learning_rate_init=1e-3,
    max_iter=30,
    early_stopping=True,
    verbose=True,
    random_state=RANDOM_STATE
)

model.fit(X_train_vec, y_train)

# ===== ارزیابی =====
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print("\n=== MLPClassifier ===")
print("Accuracy:", round(acc, 4))

for avg in ["macro", "weighted", "micro"]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=avg, zero_division=0
    )
    print(f"{avg.capitalize():9s} -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
