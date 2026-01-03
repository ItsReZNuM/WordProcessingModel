import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -------- تنظیمات --------
FILE_PATH = "dataset.xlsx"
TEXT_COL = "data"
LABEL_COL = "label"
TEST_SIZE = 0.2
RANDOM_STATE = 42

TRAIN_TXT = "fasttext_train.txt"
TEST_TXT = "fasttext_test.txt"
MODEL_OUT = "fasttext_model.bin"

# -------- پاکسازی سبک (اختیاری ولی مفید) --------
# برای FastText زیاد پیچیده‌اش نکن؛ فقط یک‌دست سازی فاصله‌ها کافیست
_space = re.compile(r"\s+")
def clean_text(s: str) -> str:
    s = str(s).strip()
    s = _space.sub(" ", s)
    return s

def write_fasttext_file(texts, labels, path):
    with open(path, "w", encoding="utf-8") as f:
        for t, lab in zip(texts, labels):
            t = clean_text(t)
            # fastText: هر خط = "__label__X متن ..."
            f.write(f"__label__{lab} {t}\n")

# -------- داده --------
df = pd.read_excel(FILE_PATH)
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
df = df[df[TEXT_COL].ne("") & df[TEXT_COL].ne("nan")].reset_index(drop=True)

X = df[TEXT_COL].tolist()
y_str = df[LABEL_COL].tolist()

# label -> id (برای اینکه برچسب‌ها مشکل کاراکتری نداشته باشن)
le = LabelEncoder()
y = le.fit_transform(y_str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# فایل‌های fastText
write_fasttext_file(X_train, y_train, TRAIN_TXT)
write_fasttext_file(X_test, y_test, TEST_TXT)

# -------- آموزش fastText --------
import fasttext  # بعد از ساخت فایل‌ها import هم اوکیه

# نکته: برای جمله‌های کوتاه wordNgrams خیلی مهمه
model = fasttext.train_supervised(
    input=TRAIN_TXT,
    epoch=40,           # مثل epochs
    lr=0.5,             # learning rate
    dim=100,            # embedding dim
    wordNgrams=3,       # bigram برای جمله کوتاه خیلی کمک می‌کنه
    minn=2, maxn=5,     # character n-gram (کمک به فارسی/کلمات جدید)
    loss="softmax"      # برای multi-class
)

model.save_model(MODEL_OUT)

# -------- پیش‌بینی و ارزیابی --------
def predict_labels(ft_model, texts):
    preds = []
    for t in texts:
        t = clean_text(t)
        labels, probs = ft_model.predict(t, k=1)
        # labels مثل "__label__3"
        pred_id = int(labels[0].replace("__label__", ""))
        preds.append(pred_id)
    return np.array(preds)

y_pred = predict_labels(model, X_test)

acc = accuracy_score(y_test, y_pred)
print("\n=== FastText (supervised) ===")
print("Accuracy:", round(acc, 4))

for avg in ["macro", "weighted", "micro"]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=avg, zero_division=0
    )
    print(f"{avg.capitalize():9s} -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")

# اگر خواستی نگاشت id -> نام کلاس‌ها رو ببینی
print("\nClass mapping (id -> label):")
for i, name in enumerate(le.classes_):
    print(i, "->", name)
