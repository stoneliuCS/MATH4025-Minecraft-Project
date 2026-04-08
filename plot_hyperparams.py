"""
Generate a matplotlib hyperparameter comparison table for all runs.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LABELS = [
    "SAC Main v2",
    "PPO Main v1",
    "SAC Fine-tuned v6",
    "PPO Taiga v1",
    "SAC Taiga v1",
]

# Rows: (display name, values per run in LABELS order)
# Order: SAC Main v2, PPO Main v1, SAC Fine-tuned v6, PPO Taiga v1, SAC Taiga v1
ROWS = [
    ("Algorithm",            ["SAC",       "PPO",    "SAC",       "PPO",    "SAC"]),
    ("Learning Rate",        ["3e-4",      "1e-4",   "3e-4",      "1e-4",   "3e-4"]),
    ("Batch Size",           ["512",       "256",    "256",       "256",    "512"]),
    ("Total Timesteps",      ["1,266,367", "1,324,302", "568,513", "1,104,536", "646,896"]),
    ("Avg Wood/Episode",     ["3.27", "62.62", "3.90", "78.88", "2.29"]),
    ("Gamma",                ["0.99",      "0.995",  "0.99",      "0.995",  "0.99"]),
    # SAC-specific
    ("Buffer Size",          ["500,000",   "—",      "1,000",     "—",      "500,000"]),
    ("Learning Starts",      ["500",       "—",      "0",         "—",      "500"]),
    ("Tau",                  ["0.005",     "—",      "0.005",     "—",      "0.005"]),
    ("Gradient Steps",       ["8",         "—",      "1",         "—",      "8"]),
    ("Ent Coef",             ["auto",      "0.01",   "auto",      "0.01",   "auto"]),
    ("Target Entropy",       ["-8.0",      "—",      "-8.0",      "—",      "-8.0"]),
    ("Target Update Int",    ["1",         "—",      "1",         "—",      "1"]),
    ("Train Freq (steps)",   ["4",         "—",      "1",         "—",      "4"]),
    ("Replay Buf n_steps",   ["10",        "—",      "1 (def)",   "—",      "10"]),
    ("Optimize Memory",      ["False",     "—",      "False",     "—",      "False"]),
    ("Use SDE",              ["False",     "False",  "False",     "False",  "False"]),
    # PPO-specific
    ("N Steps",              ["—",         "2048",   "—",         "2048",   "—"]),
    ("N Epochs",             ["—",         "10",     "—",         "10",     "—"]),
    ("GAE Lambda",           ["—",         "0.95",   "—",         "0.95",   "—"]),
    ("Clip Range",           ["—",         "0.2",    "—",         "0.2",    "—"]),
    ("VF Coef",              ["—",         "0.5",    "—",         "0.5",    "—"]),
    ("Max Grad Norm",        ["—",         "0.5",    "—",         "0.5",    "—"]),
    ("Normalize Advantage",  ["—",         "True",   "—",         "True",   "—"]),
    ("Clip Range VF",        ["—",         "None",   "—",         "None",   "—"]),
    ("Target KL",            ["—",         "None",   "—",         "None",   "—"]),
]

N_COLS = len(LABELS)
N_ROWS = len(ROWS)

fig, ax = plt.subplots(figsize=(14, 11))
ax.axis("off")
fig.patch.set_facecolor("#1e1e2e")

HEADER_BG   = "#313244"
HEADER_FG   = "#cdd6f4"
ODD_BG      = "#181825"
EVEN_BG     = "#1e1e2e"
CELL_FG     = "#cdd6f4"
NA_FG       = "#585b70"
DIFF_BG     = "#2a2a3e"   # highlight cells that differ from the column default
SAC_ACCENT  = "#89b4fa"
PPO_ACCENT  = "#a6e3a1"

col_labels = ["Parameter"] + LABELS
col_widths = [0.18] + [0.155] * N_COLS
x_positions = []
x = 0.01
for w in col_widths:
    x_positions.append(x + w / 2)
    x += w

row_height = 0.054
header_y = 1.0 - row_height / 2

def cell_bg(row_idx):
    return ODD_BG if row_idx % 2 == 0 else EVEN_BG

def draw_cell(ax, x, y, w, h, text, bg, fg, fontsize=9, bold=False):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2 + 0.003, y - h / 2 + 0.004),
        w - 0.006, h - 0.008,
        boxstyle="round,pad=0.01",
        facecolor=bg, edgecolor="none",
        transform=ax.transAxes, clip_on=False
    )
    ax.add_patch(rect)
    ax.text(x, y, text,
            ha="center", va="center",
            fontsize=fontsize,
            color=fg,
            fontweight="bold" if bold else "normal",
            transform=ax.transAxes)

# Header
for j, label in enumerate(col_labels):
    accent = SAC_ACCENT if "SAC" in label else (PPO_ACCENT if "PPO" in label else HEADER_FG)
    draw_cell(ax, x_positions[j], header_y, col_widths[j], row_height,
              label, HEADER_BG, accent, fontsize=9, bold=True)

# Rows
for i, (param, values) in enumerate(ROWS):
    y = header_y - (i + 1) * row_height
    bg = cell_bg(i)

    # Param name
    draw_cell(ax, x_positions[0], y, col_widths[0], row_height,
              param, HEADER_BG, HEADER_FG, fontsize=8.5, bold=True)

    # Detect differing values (excluding "—")
    real_vals = [v for v in values if v != "—"]
    all_same = len(set(real_vals)) <= 1

    for j, val in enumerate(values):
        col_label = LABELS[j]
        is_sac = "SAC" in col_label
        is_ppo = "PPO" in col_label

        if val == "—":
            fg = NA_FG
            cell_bg_col = bg
        else:
            fg = SAC_ACCENT if is_sac else PPO_ACCENT
            cell_bg_col = DIFF_BG if (not all_same and val != real_vals[0]) else bg

        draw_cell(ax, x_positions[j + 1], y, col_widths[j + 1], row_height,
                  val, cell_bg_col, fg, fontsize=8.5)

fig.suptitle("Hyperparameter Comparison — All Runs",
             fontsize=13, fontweight="bold", color=HEADER_FG, y=0.97)

plt.tight_layout(rect=[0, 0, 1, 0.95])

import os
out_path = os.path.join(os.path.dirname(__file__), "plots", "hyperparameters.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out_path}")
