import os
import re
import numpy as np
import torch
import torch.nn.functional as F
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# 1) تنظیمات
# =========================
MODEL_DIR = Path(r"C:\Users\amir5\PycharmProjects\REZA_mohamadNia\DEEP\parsbert_cls\checkpoint-2814")
MODEL_DIR = MODEL_DIR.resolve()

MAX_LEN = 128
TOP_K = 12
OUT_DIR = "shap_figs"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLES = [
    "پایتون یک زبان برنامه‌نویسی مفسری است",
    "تیم ملی والیبال در بازی دیشب پیروز شد",

]

# اگر اسم کلاس‌ها را دقیق داری اینجا بگذار (اختیاری)
CLASS_NAMES = None  # یا لیست 17تایی

print("MODEL_DIR:", MODEL_DIR)
print("Exists:", MODEL_DIR.exists())
print("Has config.json:", (MODEL_DIR / "config.json").exists())

if not (MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists()):
    raise FileNotFoundError("مسیر مدل اشتباه است یا config.json وجود ندارد. MODEL_DIR را درست کن.")

# =========================
# 2) نمایش درست فارسی در Matplotlib (اختیاری)
# =========================
def fa_text(s: str) -> str:
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        s = arabic_reshaper.reshape(s)
        return get_display(s)
    except Exception:
        return s

# =========================
# 3) Load tokenizer + model (حل مشکل مسیر ویندوزی)
# =========================
model_path = MODEL_DIR.as_posix()  # مهم: تبدیل \ به /

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("Device:", device)
print("Num labels:", model.config.num_labels)

# =========================
# 4) تابع predict_proba برای SHAP
# =========================
def predict_proba(texts):
    # shap ممکن است انواع ورودی بدهد
    if texts is None:
        texts = [""]
    elif isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()
    elif not isinstance(texts, (list, tuple)):
        texts = [str(texts)]

    # تبدیل همه به str
    texts = ["" if t is None else (t if isinstance(t, str) else str(t)) for t in texts]

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    return probs

# =========================
# 5) SHAP Explainer
# =========================
explainer = shap.Explainer(
    predict_proba,
    shap.maskers.Text(tokenizer)
)

# =========================
# 6) ادغام زیرواژه‌ها برای خوانایی (WordPiece)
# =========================
def merge_wordpieces(tokens, values):
    """
    تبدیل توکن‌های WordPiece مثل ## به کلمات قابل خواندن و جمع کردن SHAP.
    """
    merged_tokens = []
    merged_vals = []

    cur = ""
    cur_val = 0.0

    def flush():
        nonlocal cur, cur_val
        if cur != "":
            merged_tokens.append(cur)
            merged_vals.append(cur_val)
            cur = ""
            cur_val = 0.0

    for t, v in zip(tokens, values):
        t = str(t)

        if t in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        # برخی tokenizerها ▁ دارند
        if t.startswith("▁"):
            flush()
            cur = t[1:]
            cur_val = float(v)
            continue

        # wordpiece continuation
        if t.startswith("##"):
            cur += t[2:]
            cur_val += float(v)
            continue

        flush()
        cur = t
        cur_val = float(v)

    flush()

    merged_tokens = [re.sub(r"\s+", " ", x).strip() for x in merged_tokens]
    return merged_tokens, np.array(merged_vals, dtype=float)

# =========================
# 7) رسم نمودار Matplotlib
# =========================
def plot_shap_bar(tokens, vals, title, out_path, top_k=12):
    vals = np.array(vals, dtype=float)

    # top مثبت و top منفی
    pos_idx = np.argsort(vals)[-top_k:][::-1]
    neg_idx = np.argsort(vals)[:top_k]

    sel_idx = np.unique(np.concatenate([neg_idx, pos_idx]))
    sel_tokens = [tokens[i] for i in sel_idx]
    sel_vals = vals[sel_idx]

    # مرتب برای نمایش
    order = np.argsort(sel_vals)
    sel_tokens = [sel_tokens[i] for i in order]
    sel_vals = sel_vals[order]

    # آماده‌سازی RTL
    sel_tokens_disp = [fa_text(t) for t in sel_tokens]
    title_disp = fa_text(title)

    plt.figure(figsize=(10, 6))
    y = np.arange(len(sel_tokens_disp))
    plt.barh(y, sel_vals)
    plt.yticks(y, sel_tokens_disp, fontsize=11)
    plt.axvline(0, linewidth=1)
    plt.title(title_disp, fontsize=12)
    plt.xlabel("SHAP value (token impact)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()

# =========================
# 8) محاسبه SHAP و ساخت شکل‌ها
# =========================
shap_values = explainer(SAMPLES)

vals_all = shap_values.values      # (n_samples, n_tokens, n_classes) یا مشابه
tokens_all = shap_values.data      # توکن‌ها

probs = predict_proba(SAMPLES)
preds = probs.argmax(axis=1)

for i, text in enumerate(SAMPLES):
    pred_c = int(preds[i])

    tokens = tokens_all[i]
    if isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()

    # SHAP values برای کلاس pred_c
    # اگر شکل آرایه فرق داشت، این روش معمولاً جواب می‌دهد:
    vals = vals_all[i][:, pred_c]

    mtoks, mvals = merge_wordpieces(tokens, vals)

    cname = f"class_{pred_c}"
    if CLASS_NAMES and pred_c < len(CLASS_NAMES):
        cname = CLASS_NAMES[pred_c]

    title = f"ParsBERT+SHAP | Pred: {cname} | {text}"
    out_path = os.path.join(OUT_DIR, f"shap_bar_{i}_pred{pred_c}.png")

    plot_shap_bar(mtoks, mvals, title, out_path, top_k=TOP_K)
    print("✅ Saved:", os.path.abspath(out_path))

print("\nDone. Figures saved in:", os.path.abspath(OUT_DIR))
