import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -----------------------------
# تنظیمات محیط (برای ویندوز)
# -----------------------------
# خاموش کردن هشدار symlink در huggingface hub (اختیاری)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# اگر oneDNN پیام‌های TF اذیتت می‌کنه (اختیاری):
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# -----------------------------
# تنظیمات کلی (قابل تغییر)
# -----------------------------
FILE_PATH = "dataset.xlsx"
TEXT_COL = "data"
LABEL_COL = "label"
TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_NAME = "HooshvareLab/bert-base-parsbert-uncased"
MAX_LEN = 128

EPOCHS = 3
TRAIN_BATCH = 8          # برای CPU بهتره 8 یا 4
EVAL_BATCH = 16
LR = 2e-5
WEIGHT_DECAY = 0.01

OUTPUT_DIR = "./parsbert_cls"

# -----------------------------
# کتابخانه‌های HF
# -----------------------------
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
import transformers

print("Transformers version:", transformers.__version__)

# -----------------------------
# 1) خواندن داده
# -----------------------------
df = pd.read_excel(FILE_PATH)
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
df = df[df[TEXT_COL].ne("") & df[TEXT_COL].ne("nan")].reset_index(drop=True)

le = LabelEncoder()
df["label_id"] = le.fit_transform(df[LABEL_COL])
num_labels = len(le.classes_)

train_df, test_df = train_test_split(
    df[[TEXT_COL, "label_id"]],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["label_id"],
)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds  = Dataset.from_pandas(test_df.reset_index(drop=True))

# -----------------------------
# 2) توکنایزر
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(
        batch[TEXT_COL],
        truncation=True,
        max_length=MAX_LEN,
    )

train_ds = train_ds.map(tokenize_fn, batched=True)
test_ds  = test_ds.map(tokenize_fn, batched=True)

train_ds = train_ds.rename_column("label_id", "labels")
test_ds  = test_ds.rename_column("label_id", "labels")

cols = ["input_ids", "attention_mask", "labels"]
train_ds.set_format(type="torch", columns=cols)
test_ds.set_format(type="torch", columns=cols)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -----------------------------
# 3) مدل
# -----------------------------
# پیام "Some weights ... newly initialized" طبیعی است چون head طبقه‌بندی جدید ساخته می‌شود.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

# -----------------------------
# 4) متریک‌ها
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    metrics = {"accuracy": acc}

    for avg in ["macro", "weighted", "micro"]:
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, average=avg, zero_division=0
        )
        metrics[f"precision_{avg}"] = p
        metrics[f"recall_{avg}"] = r
        metrics[f"f1_{avg}"] = f1

    return metrics

# -----------------------------
# 5) TrainingArguments (سازگار با نسخه‌های مختلف)
# -----------------------------
set_seed(RANDOM_STATE)

# تنظیمات پایه که معمولاً در همه نسخه‌ها کار می‌کند
base_kwargs = dict(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    per_device_train_batch_size=TRAIN_BATCH,
    per_device_eval_batch_size=EVAL_BATCH,
    logging_steps=25,
    report_to="none",
    fp16=False,
)

# بعضی نسخه‌های قدیمی پارامترهای evaluation_strategy/save_strategy را ندارند.
# پس با try/except می‌سازیم.
try:
    training_args = TrainingArguments(
        **base_kwargs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
    )
except TypeError:
    # نسخه قدیمی: بدون evaluation_strategy و save_strategy
    # در این حالت eval را بعداً دستی انجام می‌دهیم.
    training_args = TrainingArguments(
        **base_kwargs
    )

# -----------------------------
# 6) Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -----------------------------
# 7) Train + Evaluate
# -----------------------------
trainer.train()

# اگر evaluation_strategy پشتیبانی نشد، اینجا ارزیابی نهایی را می‌گیریم
metrics = trainer.evaluate()
print("\n=== ParsBERT Results (Test) ===")
for k in sorted(metrics.keys()):
    if k.startswith("eval_"):
        print(f"{k}: {metrics[k]:.4f}")

print("\nClass mapping (id -> label):")
for i, name in enumerate(le.classes_):
    print(i, "->", name)
