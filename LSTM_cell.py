import math, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ╔══════════════════════════════════════════════════════════════════╗
# ║              ★  CHOOSE YOUR INPUTS HERE  ★                      ║
# ║                                                                  ║
# ║  FEATURE_SET — which features to train on:                       ║
# ║    "full"    → PCA components + market state features (X_final)  ║
# ║    "pca"     → PCA components only (no market state features)    ║
# ║    "raw"     → original 60 LOB features (no PCA, no market)      ║
# ╚══════════════════════════════════════════════════════════════════╝
FEATURE_SET = "raw"    # ← change me: "full" | "pca" | "raw"

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 42
SEQ_LEN     = 30
HIDDEN_SIZE = 64       # LSTM hidden state dimension
N_LAYERS    = 2        # stacked LSTM layers
DROPOUT     = 0.1
BATCH_SIZE  = 256
EPOCHS      = 30
LR          = 3e-4
PATIENCE    = 6

torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Feature Selection  (identical to Transformer cell)
# ══════════════════════════════════════════════════════════════════════════════
def select_features(feature_set: str):
    """
    Returns numpy arrays (X, y) based on the chosen feature set.
    Reads X_final, Y_final, df from the notebook scope (Cell 1 output).
    """
    if feature_set == "full":
        print("Feature set : FULL  (PCA + market state)")
        X = X_final.values.astype(np.float32)

    elif feature_set == "pca":
        pca_cols = [c for c in X_final.columns if c.startswith("pca_")]
        print(f"Feature set : PCA only  ({len(pca_cols)} components)")
        X = X_final[pca_cols].values.astype(np.float32)

    elif feature_set == "raw":
        lob_cols = []
        for i in range(15):
            lob_cols += [f"bids_distance_{i}", f"asks_distance_{i}",
                         f"bids_notional_{i}", f"asks_notional_{i}"]
        print(f"Feature set : RAW LOB  ({len(lob_cols)} features, no PCA)")
        X = df[lob_cols].values.astype(np.float32)

    else:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Choose 'full', 'pca', or 'raw'.")

    y = (Y_final.values + 1).astype(np.int64)   # {-1, 0, 1} → {0, 1, 2}
    print(f"X shape: {X.shape}  |  y shape: {y.shape}")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Dataset & DataLoaders  (identical to Transformer cell)
# ══════════════════════════════════════════════════════════════════════════════
class LOBDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y       = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, i):
        return self.X[i : i + self.seq_len], self.y[i + self.seq_len - 1]


def build_loaders(X, y):
    """Chronological 80/20 split → normalise → sliding-window DataLoaders."""
    split  = int(len(X) * 0.8)
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X[:split])
    X_te   = scaler.transform(X[split:])
    y_tr, y_te = y[:split], y[split:]

    counts  = np.bincount(y_tr, minlength=3).astype(float)
    class_w = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(LOBDataset(X_tr, y_tr, SEQ_LEN),
                              batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(LOBDataset(X_te, y_te, SEQ_LEN),
                              batch_size=BATCH_SIZE * 2, shuffle=False)

    print(f"Train samples: {len(train_loader.dataset):,}  |  "
          f"Test samples: {len(test_loader.dataset):,}")
    print(f"Class weights (down/flat/up): {class_w.cpu().numpy().round(3)}")
    return train_loader, test_loader, class_w


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — LSTM Model
# ══════════════════════════════════════════════════════════════════════════════
class LOBLSTM(nn.Module):
    """
    Stacked Bidirectional LSTM for LOB price-movement classification.

    Architecture
    ────────────
    Input : (B, T, n_features)
    ↓
    Projection layer : Linear(n_features → HIDDEN_SIZE)   # align input dim
    ↓
    Bidirectional LSTM × N_LAYERS                          # capture temporal order-flow dynamics
      - hidden_size = HIDDEN_SIZE
      - dropout between layers
    ↓
    Take the last time-step hidden state (both directions concat)
    ↓
    Classification head:
      Linear(2 * HIDDEN_SIZE → HIDDEN_SIZE) → GELU → Dropout
      Linear(HIDDEN_SIZE → 3)               # three classes: Down / Flat / Up
    """

    def __init__(self, n_features: int):
        super().__init__()

        # Project raw features into the LSTM's expected input size
        self.input_proj = nn.Linear(n_features, HIDDEN_SIZE)

        # Stacked bidirectional LSTM
        # dropout only applies between layers (not after the last layer)
        self.lstm = nn.LSTM(
            input_size   = HIDDEN_SIZE,
            hidden_size  = HIDDEN_SIZE,
            num_layers   = N_LAYERS,
            batch_first  = True,
            bidirectional= True,
            dropout      = DROPOUT if N_LAYERS > 1 else 0.0,
        )

        # Layer norm on the concatenated final hidden state
        self.norm = nn.LayerNorm(2 * HIDDEN_SIZE)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(2 * HIDDEN_SIZE, HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, 3),
        )

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)   # orthogonal init for recurrent weights
            elif "bias" in name:
                param.data.fill_(0)
                # Set forget-gate bias to 1 to help with gradient flow
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x : (B, T, n_features)
        x = self.input_proj(x)              # (B, T, HIDDEN_SIZE)
        out, (h_n, _) = self.lstm(x)       # out: (B, T, 2*HIDDEN_SIZE)

        # Take the last-step output for both directions
        # h_n shape: (N_LAYERS * 2, B, HIDDEN_SIZE)
        # Grab the top-layer forward & backward hidden states
        h_fwd = h_n[-2]                     # (B, HIDDEN_SIZE)  forward, last layer
        h_bwd = h_n[-1]                     # (B, HIDDEN_SIZE)  backward, last layer
        h     = torch.cat([h_fwd, h_bwd], dim=-1)   # (B, 2 * HIDDEN_SIZE)

        h = self.norm(h)
        return self.head(h)                 # (B, 3)


def build_model(n_features: int) -> LOBLSTM:
    model    = LOBLSTM(n_features).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}  |  Input features: {n_features}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Training
# ══════════════════════════════════════════════════════════════════════════════
def train(model, train_loader, test_loader, class_w):
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc, best_state, patience_cnt = 0.0, None, 0

    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train(); running = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # gradient clipping
            optimizer.step()
            running += loss.item() * len(yb)
        scheduler.step()
        train_loss = running / len(train_loader.dataset)

        # ── validate ──
        model.eval(); vloss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits  = model(Xb)
                vloss  += criterion(logits, yb).item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                total   += len(yb)
        val_loss = vloss / total
        val_acc  = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Ep {epoch:3d}/{EPOCHS} | "
              f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, patience_cnt = val_acc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early stop at epoch {epoch}. Best val acc: {best_acc:.4f}")
                break

    model.load_state_dict(best_state)
    return history, best_acc


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Evaluation
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            all_preds.append(model(Xb.to(DEVICE)).argmax(1).cpu().numpy())
            all_labels.append(yb.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    print(classification_report(y_true, y_pred, target_names=["Down", "Flat", "Up"]))
    return y_pred, y_true


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Plots
# ══════════════════════════════════════════════════════════════════════════════
def plot_results(history, y_pred, y_true, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"LSTM LOB — {title}", fontsize=14)

    ep = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(ep, history["train_loss"], label="Train", color="steelblue")
    axes[0].plot(ep, history["val_loss"],   label="Val",   color="tomato")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    best = max(history["val_acc"])
    axes[1].plot(ep, history["val_acc"], color="seagreen")
    axes[1].axhline(best, color="tomato", linestyle="--", label=f"Best={best:.4f}")
    axes[1].set_title("Val Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
                xticklabels=["Down", "Flat", "Up"],
                yticklabels=["Down", "Flat", "Up"])
    axes[2].set_title("Confusion Matrix")
    axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True")

    plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Run pipeline
# ══════════════════════════════════════════════════════════════════════════════
def run_lstm(feature_set: str):
    """
    Full LSTM pipeline in one call.

    Parameters
    ----------
    feature_set : "full" | "pca" | "raw"
        Controls which features are fed to the LSTM.
        Reads X_final, Y_final (and df for "raw") from the notebook scope.
    """
    print("=" * 60)
    print(f"  LSTM LOB — feature_set = '{feature_set}'")
    print("=" * 60)

    X, y                       = select_features(feature_set)
    train_ld, test_ld, class_w = build_loaders(X, y)
    model                      = build_model(n_features=X.shape[1])

    print("\n--- Training ---")
    history, best_acc = train(model, train_ld, test_ld, class_w)

    print("\n--- Evaluation ---")
    y_pred, y_true = evaluate(model, test_ld)

    plot_results(history, y_pred, y_true, title=f"feature_set={feature_set}")

    return model, history


# ══════════════════════════════════════════════════════════════════════════════
#  ★  ENTRY POINT — change FEATURE_SET at the top, then run this cell  ★
# ══════════════════════════════════════════════════════════════════════════════
model, history = run_lstm(FEATURE_SET)
