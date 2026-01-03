import re
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# =========================
# تنظیمات کلی (قابل تغییر)
# =========================
FILE_PATH = "dataset.xlsx"
TEXT_COL = "data"
LABEL_COL = "label"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =========================
# تنظیمات HAN (قابل تغییر)
# =========================
VOCAB_SIZE = 30000
EMB_DIM = 200

MAX_SENTENCES = 3   # حداکثر تعداد جمله در هر متن
MAX_WORDS = 30      # حداکثر تعداد کلمه در هر جمله

WORD_RNN_UNITS = 64
SENT_RNN_UNITS = 64

DROPOUT = 0.3
EPOCHS = 8
BATCH_SIZE = 128
LR = 2e-3

USE_EARLY_STOPPING = True  # False کن تا ارلی‌استاپینگ خاموش شود

# =========================
# 1) توابع کمکی: جمله‌بندی و توکن‌سازی سلسله مراتبی
# =========================
_SENT_SPLIT_RE = re.compile(r"[.!?\n\r؟]+")
_MULTI_SPACE_RE = re.compile(r"\s+")

def split_to_sentences(text: str):
    text = (text or "").strip()
    text = _MULTI_SPACE_RE.sub(" ", text)
    # جدا کردن به جمله‌ها (ساده ولی کارراه‌انداز)
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if not sents:
        sents = [""]  # برای جلوگیری از خالی شدن کامل
    return sents

def build_hier_tensor(texts, tokenizer, max_sents, max_words):
    """
    خروجی: X با شکل (N, max_sents, max_words)
    """
    X = np.zeros((len(texts), max_sents, max_words), dtype=np.int32)

    for i, t in enumerate(texts):
        sents = split_to_sentences(t)[:max_sents]
        # هر جمله → توکن‌های کلمه
        sent_seqs = tokenizer.texts_to_sequences(sents)

        for j, seq in enumerate(sent_seqs):
            seq = seq[:max_words]
            # pad در انتها
            X[i, j, :len(seq)] = seq

    return X

# =========================
# 2) Attention Layer (عمومی)
# =========================
class Attention(layers.Layer):
    """
    Attention ساده: برای یک توالی (T, D) وزن می‌سازد و بردار زمینه می‌دهد.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = None
        self.b = None
        self.u = None

    def build(self, input_shape):
        # input_shape: (batch, time, features)
        feat_dim = input_shape[-1]
        self.W = self.add_weight(shape=(feat_dim, feat_dim), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(shape=(feat_dim,), initializer="zeros", trainable=True)
        self.u = self.add_weight(shape=(feat_dim, 1), initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, x, mask=None):
        # x: (B, T, D)
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)      # (B, T, D)
        ait = tf.tensordot(uit, self.u, axes=1)                     # (B, T, 1)
        ait = tf.squeeze(ait, axis=-1)                              # (B, T)

        if mask is not None:
            # mask: (B, T) -> مقدارهای پد را خیلی منفی کن
            mask = tf.cast(mask, tf.float32)
            ait = ait + (1.0 - mask) * (-1e9)

        a = tf.nn.softmax(ait, axis=1)                              # (B, T)
        a = tf.expand_dims(a, axis=-1)                              # (B, T, 1)
        weighted = x * a                                            # (B, T, D)
        return tf.reduce_sum(weighted, axis=1)                      # (B, D)

# =========================
# 3) خواندن داده
# =========================
df = pd.read_excel(FILE_PATH)
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
df = df[df[TEXT_COL].ne("") & df[TEXT_COL].ne("nan")].reset_index(drop=True)

texts = df[TEXT_COL].tolist()

le = LabelEncoder()
y = le.fit_transform(df[LABEL_COL])
num_classes = len(le.classes_)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# =========================
# 4) Word Tokenizer
# =========================
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

X_train = build_hier_tensor(X_train_text, tokenizer, MAX_SENTENCES, MAX_WORDS)
X_test  = build_hier_tensor(X_test_text,  tokenizer, MAX_SENTENCES, MAX_WORDS)

print("X_train shape:", X_train.shape)  # (N, max_sents, max_words)

# =========================
# 5) ساخت مدل HAN
# =========================
def build_han():
    # ورودی: (max_sents, max_words)
    inp = layers.Input(shape=(MAX_SENTENCES, MAX_WORDS), dtype="int32")

    # ---- Word Encoder ----
    word_inp = layers.Input(shape=(MAX_WORDS,), dtype="int32")
    w = layers.Embedding(VOCAB_SIZE, EMB_DIM, mask_zero=True)(word_inp)
    w = layers.Bidirectional(layers.GRU(WORD_RNN_UNITS, return_sequences=True, dropout=DROPOUT))(w)
    w = Attention()(w)  # (B, 2*units) خلاصه جمله
    w = layers.Dropout(DROPOUT)(w)

    word_encoder = models.Model(word_inp, w, name="word_encoder")

    # روی هر جمله (TimeDistributed)
    sent_vecs = layers.TimeDistributed(word_encoder)(inp)  # (B, max_sents, sent_dim)

    # ---- Sentence Encoder ----
    s = layers.Bidirectional(layers.GRU(SENT_RNN_UNITS, return_sequences=True, dropout=DROPOUT))(sent_vecs)
    doc_vec = Attention()(s)  # (B, doc_dim)
    doc_vec = layers.Dropout(DROPOUT)(doc_vec)

    x = layers.Dense(256, activation="relu")(doc_vec)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out, name="HAN")
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

model = build_han()
model.summary()

# =========================
# 6) آموزش
# =========================
callbacks = []
if USE_EARLY_STOPPING:
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=2, restore_best_weights=True
    ))

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    callbacks=callbacks
)

# =========================
# 7) ارزیابی
# =========================
y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

acc = accuracy_score(y_test, y_pred)
print("\n=== HAN ===")
print(f"MAX_SENTENCES={MAX_SENTENCES} | MAX_WORDS={MAX_WORDS} | EPOCHS={EPOCHS}")
print("Accuracy:", round(acc, 4))

for avg in ["macro", "weighted", "micro"]:
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=avg, zero_division=0)
    print(f"{avg.capitalize():9s} -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
