import re
import math
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# =========================
# تنظیمات (قابل تغییر)
# =========================
FILE_PATH = "dataset.xlsx"
TEXT_COL = "data"
LABEL_COL = "label"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# گراف
MAX_VOCAB = 30000          # سقف واژگان
MIN_DF = 2                 # حذف کلمات خیلی کم‌تکرار
WINDOW_SIZE = 10           # پنجره هم‌وقوعی برای Word-Word
MAX_PPMI_EDGES = 200000    # سقف تعداد یال‌های Word-Word (برای کنترل RAM/زمان)

# مدل
HIDDEN_DIM = 256
DROPOUT = 0.5
LR = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# =========================
# پاکسازی و توکن‌سازی ساده
# =========================
_space = re.compile(r"\s+")
def clean_text(s: str) -> str:
    s = str(s).strip()
    s = _space.sub(" ", s)
    return s

def simple_tokenize(s: str):
    return clean_text(s).split()


# =========================
# 1) خواندن داده
# =========================
df = pd.read_excel(FILE_PATH)
df[TEXT_COL] = df[TEXT_COL].astype(str).map(clean_text)
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
df = df[df[TEXT_COL].ne("") & df[TEXT_COL].ne("nan")].reset_index(drop=True)

texts = df[TEXT_COL].tolist()

le = LabelEncoder()
y_all = le.fit_transform(df[LABEL_COL])
num_classes = len(le.classes_)
num_docs = len(texts)

doc_indices = np.arange(num_docs)
train_docs, test_docs, _, _ = train_test_split(
    doc_indices,
    y_all,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_all
)

train_mask_docs = np.zeros(num_docs, dtype=bool)
test_mask_docs = np.zeros(num_docs, dtype=bool)
train_mask_docs[train_docs] = True
test_mask_docs[test_docs] = True


# =========================
# 2) TF-IDF (Doc-Word edges) + واژگان
# =========================
tfidf = TfidfVectorizer(
    tokenizer=simple_tokenize,
    preprocessor=None,
    token_pattern=None,
    max_features=MAX_VOCAB,
    min_df=MIN_DF
)

X_tfidf = tfidf.fit_transform(texts)      # (num_docs, vocab_size)
vocab = tfidf.vocabulary_                 # word -> id
vocab_size = len(vocab)

print("Docs:", num_docs)
print("Vocab size:", vocab_size)

# mapping node ids:
# doc nodes: 0..num_docs-1
# word nodes: num_docs..num_docs+vocab_size-1
num_nodes = num_docs + vocab_size


# =========================
# 3) Word-Word edges با PPMI (هم‌وقوعی)
# =========================
word_counts = Counter()
pair_counts = Counter()
total_windows = 0

for text in texts:
    tokens = [t for t in simple_tokenize(text) if t in vocab]
    if not tokens:
        continue

    for t in tokens:
        word_counts[t] += 1

    # ساخت پنجره‌ها
    if len(tokens) <= WINDOW_SIZE:
        windows = [tokens]
    else:
        windows = [tokens[i:i + WINDOW_SIZE] for i in range(len(tokens) - WINDOW_SIZE + 1)]

    for w in windows:
        total_windows += 1
        uniq = list(dict.fromkeys(w))  # unique داخل پنجره
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = uniq[i], uniq[j]
                if a == b:
                    continue
                pair_counts[(a, b)] += 1
                pair_counts[(b, a)] += 1

def compute_ppmi(a, b, cab):
    ca = word_counts[a]
    cb = word_counts[b]
    pmi = math.log((cab * total_windows) / (ca * cb) + 1e-12)
    return max(pmi, 0.0)

ppmi_triplets = []
for (a, b), cab in pair_counts.items():
    val = compute_ppmi(a, b, cab)
    if val > 0:
        ppmi_triplets.append((a, b, val))

# مرتب‌سازی و محدودسازی تعداد یال‌ها
ppmi_triplets.sort(key=lambda x: x[2], reverse=True)
ppmi_triplets = ppmi_triplets[:MAX_PPMI_EDGES]

word_word_u, word_word_v, word_word_w = [], [], []
for a, b, val in ppmi_triplets:
    u = num_docs + vocab[a]
    v = num_docs + vocab[b]
    word_word_u.append(u)
    word_word_v.append(v)
    word_word_w.append(float(val))

print("PPMI edges kept:", len(word_word_u))


# =========================
# 4) Doc-Word edges از TF-IDF
# =========================
X_coo = X_tfidf.tocoo()
doc_word_u = X_coo.row
doc_word_v = X_coo.col + num_docs
doc_word_w = X_coo.data.astype(np.float32)

# =========================
# 5) ساخت edge_index و edge_weight (Undirected)
# =========================
# doc-word (u->v و v->u)
u_dw = np.concatenate([doc_word_u, doc_word_v])
v_dw = np.concatenate([doc_word_v, doc_word_u])
w_dw = np.concatenate([doc_word_w, doc_word_w])

# word-word (u->v و v->u)
u_ww = np.concatenate([np.array(word_word_u), np.array(word_word_v)])
v_ww = np.concatenate([np.array(word_word_v), np.array(word_word_u)])
w_ww = np.concatenate([np.array(word_word_w, dtype=np.float32),
                       np.array(word_word_w, dtype=np.float32)])

u_all = np.concatenate([u_dw, u_ww])
v_all = np.concatenate([v_dw, v_ww])
w_all = np.concatenate([w_dw, w_ww])

edge_index = torch.tensor([u_all, v_all], dtype=torch.long)
edge_weight = torch.tensor(w_all, dtype=torch.float32)

# =========================
# 6) آماده‌سازی Data (labels فقط برای doc nodes)
# =========================
y_nodes = torch.full((num_nodes,), -1, dtype=torch.long)
y_nodes[:num_docs] = torch.tensor(y_all, dtype=torch.long)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[:num_docs] = torch.tensor(train_mask_docs)
test_mask[:num_docs] = torch.tensor(test_mask_docs)

data = Data(
    edge_index=edge_index,
    edge_weight=edge_weight,
    y=y_nodes,
    train_mask=train_mask,
    test_mask=test_mask
).to(DEVICE)

print("Total nodes:", num_nodes)
print("Total edges (directed):", data.edge_index.size(1))


# =========================
# 7) مدل TextGCN (Embedding برای node features + 2-layer GCN)
# =========================
class TextGCN(nn.Module):
    def __init__(self, num_nodes, hidden_dim, num_classes, dropout):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, edge_index, edge_weight):
        x = self.node_emb.weight  # (num_nodes, hidden_dim)
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x  # logits (num_nodes, num_classes)

model = TextGCN(num_nodes, HIDDEN_DIM, num_classes, DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


# =========================
# 8) Train / Evaluate
# =========================
def compute_all_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    out = {"accuracy": acc}
    for avg in ["macro", "weighted", "micro"]:
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        out[f"precision_{avg}"] = p
        out[f"recall_{avg}"] = r
        out[f"f1_{avg}"] = f1
    return out

def train_one_epoch():
    model.train()
    optimizer.zero_grad()
    logits = model(data.edge_index, data.edge_weight)
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(mask):
    model.eval()
    logits = model(data.edge_index, data.edge_weight)
    preds = logits.argmax(dim=1)

    y_true = data.y[mask].cpu().numpy()
    y_pred = preds[mask].cpu().numpy()
    return compute_all_metrics(y_true, y_pred)

best_f1_macro = -1.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    loss = train_one_epoch()
    test_metrics = evaluate(data.test_mask)

    if test_metrics["f1_macro"] > best_f1_macro:
        best_f1_macro = test_metrics["f1_macro"]
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if epoch == 1 or epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | loss={loss:.4f} | test_acc={test_metrics['accuracy']:.4f} | test_f1_macro={test_metrics['f1_macro']:.4f}")

if best_state is not None:
    model.load_state_dict(best_state)

final_metrics = evaluate(data.test_mask)
print("\n=== TextGCN (final on test) ===")
for k, v in final_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nClass mapping (id -> label):")
for i, name in enumerate(le.classes_):
    print(i, "->", name)
