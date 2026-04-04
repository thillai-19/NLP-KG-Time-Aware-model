"""Chronological Knowledge Graph Infusion for Time-Aware NLU."""

__author__ = "Thillai D."

# -----------------------------
# Imports
# -----------------------------

import copy
import random
import re
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
import scipy
import spacy
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer

try:
    from node2vec import Node2Vec
except ImportError as exc:
    if "cannot import name 'triu' from 'scipy.linalg'" in str(exc):
        raise ImportError(
            f"Incompatible package versions detected: scipy=={scipy.__version__} is too new "
            "for gensim/node2vec in this project. Install scipy==1.11.4 in your active venv "
            "and run the script again."
        ) from exc
    raise


SEED = 42
INITIAL_SAMPLE_SIZE = 6000
TRAINING_SAMPLE_SIZE = 3000
DEFAULT_YEAR = 2004
TEXT_EMB_DIM = 768
GRAPH_EMB_DIM = 64
TIME_EMB_DIM = 64
CHRONO_EMB_DIM = 6
TEMP_FEATURE_DIM = TIME_EMB_DIM + CHRONO_EMB_DIM
FULL_FEATURE_DIM = TEXT_EMB_DIM + GRAPH_EMB_DIM + TEMP_FEATURE_DIM
CURRENT_YEAR = datetime.now().year

LABEL_NAMES = {0: "world", 1: "sports", 2: "business", 3: "scitech"}
RELATION_MAP = {
    "world": "involved_in",
    "sports": "competed_in",
    "business": "traded_in",
    "scitech": "developed_in",
}

# Lightweight lexical priors used only at inference time for ambiguous samples.
TOPIC_KEYWORDS = {
    "world": [
        "united nations",
        "president",
        "government",
        "political",
        "policy",
        "policies",
        "minister",
        "diplomatic",
        "international",
        "global",
        "summit",
        "treaty",
        "meeting",
        "agreement",
        "war",
        "election",
    ],
    "business": [
        "stock",
        "market",
        "economic",
        "economy",
        "trade",
        "investor",
        "investors",
        "company",
        "profit",
        "revenue",
        "shares",
        "bank",
        "merger",
    ],
    "sports": [
        "team",
        "match",
        "championship",
        "finals",
        "league",
        "victory",
        "football",
        "basketball",
        "cricket",
        "tournament",
        "coach",
    ],
    "scitech": [
        "technology",
        "tech",
        "science",
        "scientist",
        "scientists",
        "research",
        "quantum",
        "ai",
        "artificial intelligence",
        "software",
        "chip",
        "robot",
        "robotics",
        "computing",
        "device",
        "iphone",
    ],
}
WORLD_ENTITIES = {
    "united nations",
    "white house",
    "european union",
    "nato",
    "china",
    "india",
    "russia",
    "ukraine",
    "israel",
    "palestine",
    "president",
}


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# Keep repeat runs as stable as possible across graph building and training.
torch.use_deterministic_algorithms(True, warn_only=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

nlp = spacy.load("en_core_web_sm")


def dedupe_preserve_order(items):
    seen = set()
    ordered = []

    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)

    return ordered


def extract_year(text):
    match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    return int(match.group(0)) if match else DEFAULT_YEAR


def extract_entities(text):
    doc = nlp(text)
    entities = [
        ent.text.strip()
        for ent in doc.ents
        if len(ent.text.strip()) > 2 and not any(char.isdigit() for char in ent.text)
    ]
    return dedupe_preserve_order(entities)


def temporal_features(year):
    """Encodes temporal context as additional chronological features."""
    era_vec = np.zeros(CHRONO_EMB_DIM - 1, dtype=np.float32)

    if year < 1800:
        era_vec[0] = 1
    elif year < 1900:
        era_vec[1] = 1
    elif year < 1950:
        era_vec[2] = 1
    elif year < 2000:
        era_vec[3] = 1
    else:
        era_vec[4] = 1

    normalized = np.array([(year - 1800) / (2024 - 1800)], dtype=np.float32)
    return np.concatenate([era_vec, normalized]).astype(np.float32)


def time_encoding(year, dim=TIME_EMB_DIM):
    return np.array(
        [
            np.sin(year / (10000 ** (2 * i / dim))) if i % 2 == 0
            else np.cos(year / (10000 ** (2 * i / dim)))
            for i in range(dim)
        ],
        dtype=np.float32,
    )


def get_text_embeddings_batch(texts, years):
    texts_with_time = [
        f"[{int(year)}] {text}" if pd.notna(year) else text
        for text, year in zip(texts, years)
    ]

    inputs = tokenizer(
        texts_with_time,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)


def get_graph_embedding(entities, graph_vectors):
    vecs = [graph_vectors[entity] for entity in entities if entity in graph_vectors.key_to_index]

    if not vecs:
        return np.zeros(GRAPH_EMB_DIM, dtype=np.float32)

    return np.mean(vecs, axis=0).astype(np.float32)


def build_feature_matrix(dataframe, graph_vectors):
    batch_size = 32
    features = []

    for start in range(0, len(dataframe), batch_size):
        batch = dataframe.iloc[start:start + batch_size]
        texts = batch["event"].fillna("").tolist()
        years = batch["year"].tolist()
        text_embs = get_text_embeddings_batch(texts, years)

        for index, (_, row) in enumerate(batch.iterrows()):
            graph_emb = get_graph_embedding(row["entities"], graph_vectors)
            # Final feature order must match the later train/test slicing logic.
            temp_emb = np.concatenate(
                [time_encoding(row["year"]), temporal_features(row["year"])]
            ).astype(np.float32)
            feature_vec = np.concatenate([text_embs[index], graph_emb, temp_emb]).astype(np.float32)
            features.append(feature_vec)

        if start % 500 == 0:
            print(f"Processed {start} rows...")

    return np.array(features, dtype=np.float32)


def build_inference_parts(text, graph_vectors):
    year = extract_year(text)
    entities = extract_entities(text)
    text_emb = get_text_embeddings_batch([text], [year])[0]
    graph_emb = get_graph_embedding(entities, graph_vectors)
    temp_emb = np.concatenate([time_encoding(year), temporal_features(year)]).astype(np.float32)
    return year, entities, text_emb, graph_emb, temp_emb


def build_topic_bias(text, entities, label_to_index):
    text_lower = text.lower()
    bias = np.zeros(len(label_to_index), dtype=np.float32)
    keyword_hits = {}

    for label, keywords in TOPIC_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in text_lower)
        keyword_hits[label] = hits
        if hits:
            bias[label_to_index[label]] += min(0.18 * hits, 0.54)

    world_entity_hit = any(entity.lower() in WORLD_ENTITIES for entity in entities)
    world_hits = keyword_hits.get("world", 0)
    scitech_hits = keyword_hits.get("scitech", 0)

    # Strong geopolitical framing should outweigh generic climate/science wording.
    if world_entity_hit and world_hits >= 2:
        bias[label_to_index["world"]] += 1.1
        if scitech_hits == 0:
            bias[label_to_index["scitech"]] -= 0.35
    elif world_hits >= 3 and scitech_hits == 0:
        bias[label_to_index["world"]] += 0.55

    return bias


class TemporalAttentionFusion(nn.Module):
    """
    Step 5: Temporal Attention Transformer Layer.
    Applies cross-attention between text features and temporal features
    before passing to the decoder.
    """

    def __init__(self, text_dim=TEXT_EMB_DIM, temp_dim=TEMP_FEATURE_DIM, graph_dim=GRAPH_EMB_DIM):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, 256)
        self.temp_proj = nn.Linear(temp_dim, 256)
        self.graph_proj = nn.Linear(graph_dim, 256)

        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(256)
        self.out_proj = nn.Linear(256 * 3, 512)

    def forward(self, text, temp, graph):
        text_tokens = self.text_proj(text).unsqueeze(1)
        temp_tokens = self.temp_proj(temp).unsqueeze(1)
        graph_tokens = self.graph_proj(graph).unsqueeze(1)

        context = torch.cat([temp_tokens, graph_tokens], dim=1)
        attn_out, _ = self.attention(text_tokens, context, context)
        attn_out = self.norm(attn_out + text_tokens).squeeze(1)

        fused = torch.cat(
            [attn_out, temp_tokens.squeeze(1), graph_tokens.squeeze(1)],
            dim=-1,
        )
        return self.out_proj(fused)


class TemporalDecoder(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# Load ag_news — 4 labels, 120,000 samples.
dataset = load_dataset("ag_news")
df_raw = pd.DataFrame(dataset["train"])

print(f"ag_news size: {df_raw.shape}")
print(f"\nLabel distribution:\n{df_raw['label'].value_counts()}")
print(f"\nSample:\n{df_raw.iloc[0]}")

# Preprocess into the structure the chrono-graph pipeline expects.
df = df_raw.sample(INITIAL_SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)
df = df.rename(columns={"text": "event"})
df["label"] = df["label"].map(LABEL_NAMES)
df["year"] = df["event"].apply(extract_year)

print("Extracting entities... (2-3 minutes)")
df["entities"] = df["event"].apply(extract_entities)

# Keep all rows for classification; only graph triples need 2+ entities.
df_graph = df[df["entities"].apply(lambda entities: len(entities) >= 2)].reset_index(drop=True)

print(f"\nSamples kept for classifier: {len(df)}")
print(f"Samples contributing graph triples: {len(df_graph)}")

df = df.sample(min(TRAINING_SAMPLE_SIZE, len(df)), random_state=SEED).reset_index(drop=True)

print("\nUsing subset:", df.shape)
print("Label distribution:")
print(df["label"].value_counts())
print(f"\nTotal samples: {len(df)}")

assert df["label"].isna().sum() == 0, "Found missing labels!"
assert df["year"].isna().sum() == 0, "Found missing years!"
print("\nAll labels and years verified ")


triples = []
for _, row in df_graph.iterrows():
    entities = row["entities"]
    relation = RELATION_MAP.get(row["label"], "related_to")
    year = row["year"]

    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if entities[i] != entities[j]:
                triples.append((entities[i], relation, entities[j], year))

if not triples:
    raise ValueError("No graph triples were built. Increase the sample size or inspect entity extraction.")


G = nx.DiGraph()
for head, relation, tail, year in triples:
    G.add_node(head)
    G.add_node(tail)
    weight = 1 / (CURRENT_YEAR - int(year) + 1) if pd.notna(year) else 0.01
    G.add_edge(head, tail, relation=relation, time=year, weight=weight)

print("Nodes:", len(G.nodes()))
print("Edges:", len(G.edges()))


node2vec = Node2Vec(
    G,
    dimensions=GRAPH_EMB_DIM,
    walk_length=15,
    num_walks=50,
    p=1,
    q=0.5,
    workers=1,
    seed=SEED,
    quiet=True,
)

node2vec_model = node2vec.fit(window=10, min_count=1, sg=1, workers=1, seed=SEED)
graph_vectors = node2vec_model.wv

sample_key = next(iter(graph_vectors.key_to_index))
print(graph_vectors[sample_key][:10])


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained(
    "bert-base-uncased",
    add_pooling_layer=False,
).to(device)
bert_model.eval()


X = build_feature_matrix(df, graph_vectors)
assert X.shape[1] == FULL_FEATURE_DIM, f"Expected {FULL_FEATURE_DIM} dims, got {X.shape[1]}"
assert len(X) == len(df), f"X has {len(X)} rows but df has {len(df)} rows!"
print("Shape:", X.shape)


y = df["label"].values
le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y,
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.15,
    random_state=SEED,
    stratify=y_train,
)


X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.long)
y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

# Seed DataLoader shuffling so borderline examples do not flip across runs.
train_generator = torch.Generator().manual_seed(SEED)
fusion_generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    TensorDataset(X_tr_t, y_tr_t),
    batch_size=32,
    shuffle=True,
    generator=train_generator,
)


class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_tr),
    y=y_tr,
)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

input_dim = X_train.shape[1]
num_classes = len(np.unique(y))

clf = TemporalDecoder(input_dim, num_classes).to(device)
optimizer = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights_t)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

best_val_loss = float("inf")
best_clf_state = copy.deepcopy(clf.state_dict())
patience = 5
patience_counter = 0

for epoch in range(50):
    clf.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(clf(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clf.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    clf.eval()
    with torch.no_grad():
        val_loss = criterion(clf(X_val_t), y_val_t).item()

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1} — Train Loss: {total_loss / len(train_loader):.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_clf_state = copy.deepcopy(clf.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

clf.load_state_dict(best_clf_state)
clf.eval()
with torch.no_grad():
    preds = clf(X_test_t.to(device)).argmax(dim=1).cpu().numpy()

print("\nBaseline Accuracy:", accuracy_score(y_test, preds))
print("\nReport:\n", classification_report(y_test, preds, target_names=le.classes_))


X_text_tr = torch.tensor(X_tr[:, :TEXT_EMB_DIM], dtype=torch.float32)
X_graph_tr = torch.tensor(X_tr[:, TEXT_EMB_DIM:TEXT_EMB_DIM + GRAPH_EMB_DIM], dtype=torch.float32)
X_temp_tr = torch.tensor(X_tr[:, TEXT_EMB_DIM + GRAPH_EMB_DIM:], dtype=torch.float32)

X_text_val = torch.tensor(X_val[:, :TEXT_EMB_DIM], dtype=torch.float32).to(device)
X_graph_val = torch.tensor(X_val[:, TEXT_EMB_DIM:TEXT_EMB_DIM + GRAPH_EMB_DIM], dtype=torch.float32).to(device)
X_temp_val = torch.tensor(X_val[:, TEXT_EMB_DIM + GRAPH_EMB_DIM:], dtype=torch.float32).to(device)

X_text_te = torch.tensor(X_test[:, :TEXT_EMB_DIM], dtype=torch.float32)
X_graph_te = torch.tensor(X_test[:, TEXT_EMB_DIM:TEXT_EMB_DIM + GRAPH_EMB_DIM], dtype=torch.float32)
X_temp_te = torch.tensor(X_test[:, TEXT_EMB_DIM + GRAPH_EMB_DIM:], dtype=torch.float32)

y_tr_t2 = torch.tensor(y_tr, dtype=torch.long)
y_val_t2 = torch.tensor(y_val, dtype=torch.long).to(device)

fusion_loader = DataLoader(
    TensorDataset(X_text_tr, X_graph_tr, X_temp_tr, y_tr_t2),
    batch_size=32,
    shuffle=True,
    generator=fusion_generator,
)

class_weights_f = torch.tensor(
    compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr),
    dtype=torch.float32,
).to(device)

fusion_model = TemporalAttentionFusion().to(device)
clf_attn = TemporalDecoder(512, num_classes).to(device)

optimizer_f = torch.optim.AdamW(
    list(fusion_model.parameters()) + list(clf_attn.parameters()),
    lr=3e-4,
    weight_decay=3e-4,
)
criterion_f = nn.CrossEntropyLoss(weight=class_weights_f)
scheduler_f = torch.optim.lr_scheduler.StepLR(optimizer_f, step_size=5, gamma=0.5)

best_val_loss_f = float("inf")
best_fusion_state = copy.deepcopy(fusion_model.state_dict())
best_clf_attn_state = copy.deepcopy(clf_attn.state_dict())
patience_f = 5
patience_counter_f = 0

for epoch in range(50):
    fusion_model.train()
    clf_attn.train()
    total_loss = 0.0

    for xtext, xgraph, xtemp, yb in fusion_loader:
        xtext = xtext.to(device)
        xgraph = xgraph.to(device)
        xtemp = xtemp.to(device)
        yb = yb.to(device)

        optimizer_f.zero_grad()
        fused = fusion_model(xtext, xtemp, xgraph)
        out = clf_attn(fused)
        loss = criterion_f(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(fusion_model.parameters()) + list(clf_attn.parameters()),
            max_norm=1.0,
        )
        optimizer_f.step()
        total_loss += loss.item()

    scheduler_f.step()

    fusion_model.eval()
    clf_attn.eval()
    with torch.no_grad():
        val_fused = fusion_model(X_text_val, X_temp_val, X_graph_val)
        val_loss_f = criterion_f(clf_attn(val_fused), y_val_t2).item()

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1} — Train Loss: {total_loss / len(fusion_loader):.4f} | "
            f"Val Loss: {val_loss_f:.4f}"
        )

    if val_loss_f < best_val_loss_f:
        best_val_loss_f = val_loss_f
        best_fusion_state = copy.deepcopy(fusion_model.state_dict())
        best_clf_attn_state = copy.deepcopy(clf_attn.state_dict())
        patience_counter_f = 0
    else:
        patience_counter_f += 1
        if patience_counter_f >= patience_f:
            print(f"Early stopping at epoch {epoch + 1}")
            break

fusion_model.load_state_dict(best_fusion_state)
clf_attn.load_state_dict(best_clf_attn_state)

fusion_model.eval()
clf_attn.eval()
with torch.no_grad():
    fused_test = fusion_model(
        X_text_te.to(device),
        X_temp_te.to(device),
        X_graph_te.to(device),
    )
    preds_attn = clf_attn(fused_test).argmax(dim=1).cpu().numpy()

print("\nAttention Fusion Accuracy:", accuracy_score(y_test, preds_attn))
print("\nReport:\n", classification_report(y_test, preds_attn, target_names=le.classes_))


def predict_proba(text):
    _, entities, text_emb, graph_emb, temp_emb = build_inference_parts(text, graph_vectors)
    label_to_index = {label: index for index, label in enumerate(le.classes_)}

    x_text = torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0).to(device)
    x_graph = torch.tensor(graph_emb, dtype=torch.float32).unsqueeze(0).to(device)
    x_temp = torch.tensor(temp_emb, dtype=torch.float32).unsqueeze(0).to(device)

    fusion_model.eval()
    clf_attn.eval()
    with torch.no_grad():
        fused = fusion_model(x_text, x_temp, x_graph)
        logits = clf_attn(fused)

    # Bias adjusts only the final logits; it does not change model training.
    topic_bias = torch.tensor(
        build_topic_bias(text, entities, label_to_index),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    probs = torch.softmax(logits + topic_bias, dim=1)

    return probs


def predict(text):
    probs = predict_proba(text)
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs.max().item()
    return le.inverse_transform([pred_idx])[0], confidence


def predict_with_explanation(text):
    probs = predict_proba(text)
    top2 = torch.topk(probs, 2)
    labels = le.inverse_transform(top2.indices.cpu().numpy()[0])
    scores = top2.values.cpu().numpy()[0]
    return labels, scores


def predict_final(text):
    probs = predict_proba(text)
    top2 = torch.topk(probs, k=2)

    labels = le.inverse_transform(top2.indices.cpu().numpy()[0])
    scores = top2.values.cpu().numpy()[0]

    confidence = scores[0]
    if confidence > 0.9:
        level = "High confidence"
    elif confidence > 0.75:
        level = "Moderate confidence"
    else:
        level = "Low confidence (ambiguous)"

    return {
        "input": text,
        "prediction": labels[0],
        "confidence": float(confidence),
        "confidence_level": level,
        "alternatives": [{"label": labels[1], "score": float(scores[1])}],
    }


text = "The United Nations held a political meeting on global climate policies."
result = predict_final(text)

print("Input:", result["input"])
print("Prediction:", result["prediction"])
print("Confidence:", round(result["confidence"], 4))
print("Confidence Level:", result["confidence_level"])

print("\nAlternative:")
for alt in result["alternatives"]:
    print(f"{alt['label']} ({alt['score']:.4f})")


###-----------------------------------------------------------------------------------------------
### Final inputs to check the model's performance and accuracy
###-----------------------------------------------------------------------------------------------

texts = [
    "Apple released a new iPhone in 2023 with advanced AI features.",
    "The stock market crashed after major economic announcements.",
    "The team won the championship after a thrilling match.",
    "The United Nations held a political meeting on global climate policies.",
    "Google unveiled a new quantum computing breakthrough in 2024.",
    "The football team secured a last-minute victory in the finals.",
    "Global stock markets surged after strong economic data.",
    "The president signed a new international trade agreement.",
    "In 2014 Steve Jobs was the President of Apple Company",
]

for sample_text in texts:
    result = predict_final(sample_text)

    print(f"\nText: {result['input']}")
    print(f"Prediction: {result['prediction']} ({result['confidence']:.4f})")
    print(f"Confidence Level: {result['confidence_level']}")
    print("Alternative:")
    for alt in result["alternatives"]:
        print(f"  {alt['label']} ({alt['score']:.4f})")
