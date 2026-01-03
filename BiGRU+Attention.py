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

BIGRU_UNITS = [128]        # مثال: [128] یا [128, 64]
DROPOUT = 0.3
REC_DROPOUT = 0.2

EPOCHS = 8
BATCH_SIZE = 128
LR = 2e-3

USE_EARLY_STOPPING = False  # False کن تا خاموش شود

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
# 3) Attention Layer (ساده)
# =========================
class Attention(layers.Layer):
    """
    Attention ساده برای توالی: x (B, T, D) -> context (B, D)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = None
        self.b = None
        self.u = None

    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(shape=(d, d), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(shape=(d,), initializer="zeros", trainable=True)
        self.u = self.add_weight(shape=(d, 1), initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, x, mask=None):
        # x: (B, T, D)
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)   # (B, T, D)
        ait = tf.tensordot(uit, self.u, axes=1)                  # (B, T, 1)
        ait = tf.squeeze(ait, axis=-1)                           # (B, T)

        if mask is not None:
            mask = tf.cast(mask, tf.float32)                     # (B, T)
            ait = ait + (1.0 - mask) * (-1e9)                    # mask padding

        a = tf.nn.softmax(ait, axis=1)                           # (B, T)
        a = tf.expand_dims(a, axis=-1)                           # (B, T, 1)
        context = tf.reduce_sum(x * a, axis=1)                   # (B, D)
        return context

# =========================
# 4) ساخت مدل BiGRU + Attention
# =========================
def build_bigru_attention_model():
    inp = layers.Input(shape=(MAX_LEN,), dtype="int32")
    x = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMB_DIM, mask_zero=True)(inp)

    # چند لایه BiGRU
    for i, units in enumerate(BIGRU_UNITS):
        return_seq = True  # برای attention باید خروجی توالی داشته باشیم
        x = layers.Bidirectional(
            layers.GRU(
                units,
                return_sequences=return_seq,
                dropout=DROPOUT,
                recurrent_dropout=REC_DROPOUT
            )
        )(x)

    # Attention روی خروجی توالی
    x = Attention()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

model = build_bigru_attention_model()
model.summary()

# =========================
# 5) آموزش
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
# 6) ارزیابی
# =========================
y_prob = model.predict(X_test_pad, batch_size=BATCH_SIZE, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

acc = accuracy_score(y_test, y_pred)
print("\n=== BiGRU + Attention ===")
print("BIGRU_UNITS:", BIGRU_UNITS, "| EPOCHS:", EPOCHS, "| MAX_LEN:", MAX_LEN)
print("Accuracy:", round(acc, 4))

for avg in ["macro", "weighted", "micro"]:
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=avg, zero_division=0)
    print(f"{avg.capitalize():9s} -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
