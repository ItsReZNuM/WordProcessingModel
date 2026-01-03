import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import tensorflow as tf
from tensorflow.keras import layers, models

# =========================
# تنظیمات کلی (قابل تغییر)
# =========================
FILE_PATH = "dataset.xlsx"
TEXT_COL = "data"
LABEL_COL = "label"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =========================
# تنظیمات Char-CNN (قابل تغییر)
# =========================
MAX_CHARS = 300          # طول ثابت رشته (اگر جمله‌ها کوتاه‌اند 200-400 خوبه)
EMB_DIM = 64             # embedding کاراکتر
FILTER_SIZES = [3, 5, 7] # کرنل‌ها (n-gram کاراکتری)
NUM_FILTERS = 128        # تعداد فیلتر برای هر کرنل
DROPOUT = 0.3

EPOCHS = 40
BATCH_SIZE = 128
LR = 2e-3

USE_EARLY_STOPPING = False  # False کن تا خاموش شود

# =========================
# پاکسازی سبک (اختیاری)
# =========================
_space = re.compile(r"\s+")
def clean_text(s: str) -> str:
    s = str(s).strip()
    s = _space.sub(" ", s)
    return s

# =========================
# 1) خواندن داده
# =========================
df = pd.read_excel(FILE_PATH)
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
df = df[df[TEXT_COL].ne("") & df[TEXT_COL].ne("nan")].reset_index(drop=True)

texts = [clean_text(t) for t in df[TEXT_COL].tolist()]

le = LabelEncoder()
y = le.fit_transform(df[LABEL_COL])
num_classes = len(le.classes_)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# =========================
# 2) ساخت واژگان کاراکتری از Train
# =========================
# فقط از کاراکترهای train vocab می‌سازیم تا leakage نشه.
all_train = "".join(X_train_text)
chars = sorted(list(set(all_train)))

# توکن‌های خاص
PAD = "<PAD>"
UNK = "<UNK>"

# ایندکس‌دهی
# 0 = PAD
# 1 = UNK
char2idx = {PAD: 0, UNK: 1}
for i, ch in enumerate(chars, start=2):
    char2idx[ch] = i

vocab_size = len(char2idx)
print("Char vocab size:", vocab_size)

def text_to_char_ids(text: str, max_len: int) -> np.ndarray:
    ids = np.zeros((max_len,), dtype=np.int32)  # PAD=0
    for i, ch in enumerate(text[:max_len]):
        ids[i] = char2idx.get(ch, 1)           # UNK=1
    return ids

X_train = np.stack([text_to_char_ids(t, MAX_CHARS) for t in X_train_text], axis=0)
X_test  = np.stack([text_to_char_ids(t, MAX_CHARS) for t in X_test_text], axis=0)

# =========================
# 3) ساخت مدل Char-CNN
# =========================
def build_char_cnn():
    inp = layers.Input(shape=(MAX_CHARS,), dtype="int32")

    x = layers.Embedding(input_dim=vocab_size, output_dim=EMB_DIM, mask_zero=False)(inp)
    x = layers.SpatialDropout1D(0.2)(x)

    conv_blocks = []
    for k in FILTER_SIZES:
        c = layers.Conv1D(filters=NUM_FILTERS, kernel_size=k, padding="valid", activation="relu")(x)
        c = layers.GlobalMaxPooling1D()(c)
        conv_blocks.append(c)

    x = layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    x = layers.Dropout(DROPOUT)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out, name="CharCNN")
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

model = build_char_cnn()
model.summary()

# =========================
# 4) آموزش
# =========================
callbacks = []
if USE_EARLY_STOPPING:
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True
        )
    )

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    callbacks=callbacks
)

# =========================
# 5) ارزیابی
# =========================
y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

acc = accuracy_score(y_test, y_pred)
print("\n=== Char-CNN ===")
print(f"MAX_CHARS={MAX_CHARS} | FILTER_SIZES={FILTER_SIZES} | NUM_FILTERS={NUM_FILTERS} | EPOCHS={EPOCHS}")
print("Accuracy:", round(acc, 4))

for avg in ["macro", "weighted", "micro"]:
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=avg, zero_division=0)
    print(f"{avg.capitalize():9s} -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")

print("\nClass mapping (id -> label):")
for i, name in enumerate(le.classes_):
    print(i, "->", name)
