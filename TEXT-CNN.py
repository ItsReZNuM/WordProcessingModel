import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
# تنظیمات مدل (قابل تغییر)
# =========================
VOCAB_SIZE = 30000
MAX_LEN = 40
EMB_DIM = 200

FILTER_SIZES = [3, 4, 5]    # اندازه پنجره کانولوشن (n-gram)
NUM_FILTERS = 128           # تعداد فیلتر برای هر اندازه
DROPOUT = 0.5

EPOCHS = 8
BATCH_SIZE = 128
LR = 2e-3

USE_EARLY_STOPPING = False   # False کن تا غیرفعال شه

# =========================
# 1) خواندن داده
# =========================
df = pd.read_excel(FILE_PATH)
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
df = df[df[TEXT_COL].ne("") & df[TEXT_COL].ne("nan")].reset_index(drop=True)

X_text = df[TEXT_COL].tolist()
le = LabelEncoder()
y = le.fit_transform(df[LABEL_COL])
num_classes = len(le.classes_)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# =========================
# 2) Tokenize + Padding
# =========================
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq  = tokenizer.texts_to_sequences(X_test_text)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_test_pad  = pad_sequences(X_test_seq,  maxlen=MAX_LEN, padding="post", truncating="post")

# =========================
# 3) ساخت مدل Text-CNN
# =========================
def build_textcnn():
    inp = layers.Input(shape=(MAX_LEN,), dtype="int32")
    x = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMB_DIM, mask_zero=False)(inp)
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

    model = models.Model(inp, out)
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

model = build_textcnn()
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
    X_train_pad, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    callbacks=callbacks
)

# =========================
# 5) ارزیابی (Precision/Recall/F1 یک عددی)
# =========================
y_prob = model.predict(X_test_pad, batch_size=BATCH_SIZE, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

acc = accuracy_score(y_test, y_pred)
print("\n=== Text-CNN ===")
print(f"FILTER_SIZES={FILTER_SIZES} | NUM_FILTERS={NUM_FILTERS} | EPOCHS={EPOCHS} | MAX_LEN={MAX_LEN}")
print("Accuracy:", round(acc, 4))

for avg in ["macro", "weighted", "micro"]:
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=avg, zero_division=0)
    print(f"{avg.capitalize():9s} -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
