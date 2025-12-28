import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base = Path("checkpoints")

logs = {
    "Base":  base / "tf_abs_layer_256_100k_heavy" / "log_transformer.csv",
    "max_len256": base / "tf_abs_layer_256_100k_maxlen256" / "log_transformer.csv",
}

# 颜色和线型可按需调整
colors = {
    "Base": "#1f77b4",
    "max_len256": "#2ca02c",
}

# 1) 绘制 loss 曲线
plt.figure(figsize=(6, 4))
for name, path in logs.items():
    df = pd.read_csv(path)
    plt.plot(df["epoch"], df["loss"], label=name, color=colors[name])
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("Transformer (larger max_len) training loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/tf_maxlen256_loss.png", dpi=300)

# 2) 绘制验证 BLEU 曲线
plt.figure(figsize=(6, 4))
for name, path in logs.items():
    df = pd.read_csv(path)
    plt.plot(df["epoch"], df["bleu"], label=name, color=colors[name])
plt.xlabel("Epoch")
plt.ylabel("Validation BLEU")
plt.title("Larger Transformer (Parameter Sensitivity) validation BLEU")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/tf_maxlen256_bleu.png", dpi=300)
print("Saved figures to figures/tf_maxlen256_loss.png and figures/tf_maxlen256_bleu.png")