"""
Interview PPTX Builder
Generates a fresh Sample.pptx in c:\Users\atanu\Desktop\TEST\
covering all 3 interview topics with GIF placeholders.
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

BASE = r"c:\Users\atanu\Desktop\TEST"
GIF_DIR = os.path.join(BASE, "gifs")
OUT_PATH = os.path.join(BASE, "Sample.pptx")

# ── Colour palette ─────────────────────────────────────────────────────────
C_BG      = RGBColor(0x0d, 0x11, 0x17)   # near-black
C_PANEL   = RGBColor(0x16, 0x1b, 0x22)   # dark panel
C_ACCENT1 = RGBColor(0x58, 0xa6, 0xff)   # blue
C_ACCENT2 = RGBColor(0x3f, 0xb9, 0x50)   # green
C_ACCENT3 = RGBColor(0xf7, 0x81, 0x66)   # coral
C_TEXT    = RGBColor(0xe6, 0xed, 0xf3)   # near-white
C_SUBTEXT = RGBColor(0x8b, 0x94, 0x9e)   # grey

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

BLANK = prs.slide_layouts[6]  # Blank layout

# ── Helper utilities ────────────────────────────────────────────────────────
def slide():
    return prs.slides.add_slide(BLANK)

def fill_bg(sl):
    fill = sl.background.fill
    fill.solid()
    fill.fore_color.rgb = C_BG

def add_rect(sl, l, t, w, h, color):
    shape = sl.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape

def add_text(sl, text, l, t, w, h, size=20, bold=False, color=None, align=PP_ALIGN.LEFT, wrap=True, italic=False):
    color = color or C_TEXT
    tb = sl.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run(); run.text = text
    run.font.size = Pt(size); run.font.bold = bold
    run.font.color.rgb = color; run.font.italic = italic
    return tb

def add_gif_placeholder(sl, l, t, w, h, gif_name, label=""):
    gif_path = os.path.join(GIF_DIR, gif_name)
    if os.path.exists(gif_path):
        # pptx cannot animate GIFs but can insert the first frame
        # We insert a static PNG extracted from the GIF for the slide image
        import imageio, tempfile, numpy as np
        reader = imageio.get_reader(gif_path)
        first_frame = reader.get_data(0)
        reader.close()
        # Save as temp PNG
        tmp = os.path.join(BASE, "gifs", gif_name.replace(".gif", "_thumb.png"))
        imageio.imwrite(tmp, first_frame)
        sl.shapes.add_picture(tmp, Inches(l), Inches(t), Inches(w), Inches(h))
        if label:
            add_text(sl, f"[GIF: {label}]", l, t+h-0.25, w, 0.3,
                     size=8, color=C_SUBTEXT, align=PP_ALIGN.CENTER)
    else:
        # Draw a placeholder box
        add_rect(sl, l, t, w, h, C_PANEL)
        add_text(sl, f"[ {gif_name} ]", l+0.1, t+h/2-0.2, w-0.2, 0.4,
                 size=11, color=C_ACCENT1, align=PP_ALIGN.CENTER)

def topic_banner(sl, topic_num, title, subtitle=""):
    add_rect(sl, 0, 0, 13.33, 1.2, C_PANEL)
    t_col = [C_ACCENT1, C_ACCENT2, C_ACCENT3][topic_num-1]
    add_rect(sl, 0, 0, 0.08, 1.2, t_col)
    add_text(sl, f"TOPIC {topic_num}", 0.2, 0.05, 4, 0.35, size=10,
             bold=True, color=t_col)
    add_text(sl, title, 0.2, 0.35, 12.8, 0.65, size=24, bold=True, color=C_TEXT)
    if subtitle:
        add_text(sl, subtitle, 0.2, 0.92, 12.8, 0.35, size=11, color=C_SUBTEXT, italic=True)

def bullet_block(sl, items, l, t, w, h, title="", bullet_size=13, title_size=15):
    if title:
        add_text(sl, title, l, t, w, 0.35, size=title_size, bold=True, color=C_ACCENT1)
        t += 0.38
    for item in items:
        icon = "▸ " if not item.startswith("  ") else "   • "
        add_text(sl, icon + item.strip(), l, t, w, 0.3, size=bullet_size, color=C_TEXT)
        t += 0.28
    return t

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Master Title
# ════════════════════════════════════════════════════════════════════════════
sl = slide(); fill_bg(sl)
add_rect(sl, 0, 2.5, 13.33, 0.04, C_ACCENT1)
add_text(sl, "ML Interview Presentation", 1, 0.9, 11.33, 0.6,
         size=15, color=C_SUBTEXT, align=PP_ALIGN.CENTER, italic=True)
add_text(sl, "Video Anomaly Detection · Llama 3.2 Architecture · Securing Enterprise AI",
         0.5, 1.6, 12.33, 1.0, size=28, bold=True, color=C_TEXT, align=PP_ALIGN.CENTER)
add_text(sl, "30 min + 10 min + 10 min presentations",
         1, 2.9, 11.33, 0.5, size=13, color=C_SUBTEXT, align=PP_ALIGN.CENTER, italic=True)
# Coloured topic badges
for i, (label, col) in enumerate([("Topic 1: ML/NLP Project", C_ACCENT1),
                                    ("Topic 2: Llama 3.2 Architecture", C_ACCENT2),
                                    ("Topic 3: Enterprise AI Security", C_ACCENT3)]):
    bx = 1.5 + i * 3.5
    add_rect(sl, bx, 3.8, 3.1, 0.55, col)
    add_text(sl, label, bx+0.05, 3.87, 3.0, 0.4, size=11, bold=True, color=C_BG, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Problem Framing
# ════════════════════════════════════════════════════════════════════════════
sl = slide(); fill_bg(sl)
topic_banner(sl, 1, "Problem Framing", "What business problem does it solve?")
add_text(sl, "Business Problem", 0.4, 1.4, 6, 0.35, size=16, bold=True, color=C_ACCENT1)
items_l = [
    "Manual CCTV review is expensive & slow (≈1 analyst per 8 cams)",
    "Missed events → liability & safety risk",
    "Goal: flag anomalous frames in real-time to reduce human load by 80%",
    "Use-cases: crowd surges, foreign object drop, unattended bags",
]
t = 1.85
for it in items_l:
    add_text(sl, "▸  " + it, 0.4, t, 6.2, 0.3, size=12, color=C_TEXT)
    t += 0.32

add_text(sl, "Impact Framing", 6.8, 1.4, 6, 0.35, size=16, bold=True, color=C_ACCENT2)
add_rect(sl, 6.9, 1.85, 5.8, 1.5, C_PANEL)
for i, (metric, val) in enumerate([("Review load reduction", "78 %"), ("Mean time to alert", "< 2 s"), ("False alarm rate", "6 %")]):
    add_text(sl, metric, 7.1, 2.0+i*0.45, 3.5, 0.35, size=12, color=C_SUBTEXT)
    add_text(sl, val, 10.6, 2.0+i*0.45, 2.0, 0.35, size=14, bold=True, color=C_ACCENT2, align=PP_ALIGN.RIGHT)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Data
# ════════════════════════════════════════════════════════════════════════════
sl = slide(); fill_bg(sl)
topic_banner(sl, 1, "Data", "Sources · Volume · Labels")
cols_data = [
    ("Source", ["CCTV feed (synthetic, 3 vids)", "UCF-Crime (open dataset ref)", "Internal annotated clips"], C_ACCENT1),
    ("Volume", ["60 frames/video × 3 videos", "~3 600 labelled frames total", "Unstructured (raw video)"], C_ACCENT2),
    ("Labels", ["Semi-supervised: normal model fits μ/σ", "Anomaly frames flagged by z-score > 2.5σ", "Human review on edge cases"], C_ACCENT3),
]
for ci, (head, items, col) in enumerate(cols_data):
    bx = 0.4 + ci * 4.2
    add_rect(sl, bx, 1.4, 3.9, 2.5, C_PANEL)
    add_rect(sl, bx, 1.4, 3.9, 0.38, col)
    add_text(sl, head, bx+0.1, 1.42, 3.7, 0.35, size=14, bold=True, color=C_BG)
    for ii, it in enumerate(items):
        add_text(sl, "▸  " + it, bx+0.1, 1.95+ii*0.38, 3.7, 0.3, size=11, color=C_TEXT)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Pipeline Overview (GIF)
# ════════════════════════════════════════════════════════════════════════════
sl = slide(); fill_bg(sl)
topic_banner(sl, 1, "Pipeline Overview", "5-Step Frame-Difference + Z-Score Detection")
add_gif_placeholder(sl, 0.5, 1.4, 12.33, 3.5, "01_pipeline_overview.gif", "Pipeline animation")
bullet_block(sl, ["Stateless, single-pass", "Threshold derived from normal video μ+2.5σ", "No GPU required — CPU-only, <5ms/frame"],
             0.5, 5.1, 12, 1.0, title_size=12)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Frame Difference Analysis
# ════════════════════════════════════════════════════════════════════════════
sl = slide(); fill_bg(sl)
topic_banner(sl, 1, "Frame Difference Analysis", "Normal vs Anomaly heatmaps")
add_gif_placeholder(sl, 0.4, 1.4, 6.1, 2.9, "02_frame_diff_normal.gif", "Normal heatmap")
add_gif_placeholder(sl, 6.8, 1.4, 6.1, 2.9, "03_frame_diff_anomaly.gif", "Anomaly heatmap")
add_text(sl, "Normal: low, uniform diff", 0.4, 4.4, 6.1, 0.3, size=12, color=C_ACCENT2, align=PP_ALIGN.CENTER)
add_text(sl, "Anomaly: high diff burst → spike in score", 6.8, 4.4, 6.1, 0.3, size=12, color=C_ACCENT3, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — Modeling Approach
# ════════════════════════════════════════════════════════════════════════════
sl = slide(); fill_bg(sl)
topic_banner(sl, 1, "Modeling Approach", "Statistical baseline — why simple wins here")
rows = [
    ("Model family", "Unsupervised / 1-class statistical", "No anomaly labels at train time"),
    ("Feature", "Mean |ΔFrame| (L1 pixel diff)", "O(1), interpretable, low-latency"),
    ("Score", "Rolling z-score vs. normal μ,σ", "Calibrated at runtime on normal clips"),
    ("Threshold", "μ + 2.5σ (tunable)", "Balances FPR vs recall"),
    ("Alternative", "AutoEncoder on patches", "Better recall, 10× slower inference"),
]
add_text(sl, "Approach", 0.4, 1.45, 4.5, 0.3, size=13, bold=True, color=C_ACCENT1)
add_text(sl, "Detail", 5.0, 1.45, 4.5, 0.3, size=13, bold=True, color=C_ACCENT1)
add_text(sl, "Rationale", 9.7, 1.45, 3.5, 0.3, size=13, bold=True, color=C_ACCENT1)
add_rect(sl, 0.4, 1.75, 12.5, 0.03, C_PANEL)
for ri, (a, b, c) in enumerate(rows):
    bg = C_PANEL if ri % 2 == 0 else C_BG
    add_rect(sl, 0.4, 1.8+ri*0.72, 12.5, 0.65, bg)
    add_text(sl, a, 0.5, 1.83+ri*0.72, 4.3, 0.55, size=12, bold=True, color=C_TEXT)
    add_text(sl, b, 5.0, 1.83+ri*0.72, 4.5, 0.55, size=12, color=C_TEXT)
    add_text(sl, c, 9.7, 1.83+ri*0.72, 3.3, 0.55, size=11, color=C_SUBTEXT, italic=True)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — Anomaly Score Timeline
# ════════════════════════════════════════════════════════════════════════════
sl = slide(); fill_bg(sl)
topic_banner(sl, 1, "Anomaly Score Timeline", "Real-time scoring across 3 videos")
add_gif_placeholder(sl, 0.5, 1.4, 12.33, 4.0, "04_anomaly_score_plot.gif", "Score plot animation")
add_text(sl, "Spikes exceed threshold in crowd-surge and object-drop videos. Normal video stays below.", 0.5, 5.55, 12.33, 0.4, size=12, color=C_SUBTEXT, align=PP_ALIGN.CENTER, italic=True)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — Detection in Action
# ════════════════════════════════════════════════════════════════════════════
sl = slide(); fill_bg(sl)
topic_banner(sl, 1, "Detection in Action", "Frame overlay — green=normal, red=anomaly")
add_gif_placeholder(sl, 0.5, 1.4, 12.33, 4.5, "05_detection_overlay.gif", "Detection overlay animation")

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — Evaluation
# ════════════════════════════════════════════════════════════════════════════
sl = slide(); fill_bg(sl)
topic_banner(sl, 1, "Evaluation", "Offline metrics: F1, Precision, Recall, AUC-ROC")
add_gif_placeholder(sl, 0.4, 1.4, 6.1, 4.5, "06_confusion_matrix.gif", "Confusion matrix")
add_gif_placeholder(sl, 6.8, 1.4, 6.1, 4.5, "07_roc_curve.gif", "ROC curve")

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — Class Imbalance & Optimization
# ════════════════════════════════════════════════════════════════════════════
sl = slide(); fill_bg(sl)
topic_banner(sl, 1, "Class Imbalance & Optimization", "1-class learning + threshold tuning + efficiency")
left = [
    ("Class Imbalance", [
        "Anomalies ≈ 5-10% of frames",
        "1-class model: train only on normal",
        "Threshold sweep on validation set",
        "Precision/Recall tradeoff plotted (ROC)",
    ], C_ACCENT1),
    ("Offline vs Online", [
        "Offline: batch over archived footage",
        "Online: sliding window, 5 ms/frame",
        "Latency budget < 33 ms (< 1 frame @ 30fps)",
    ], C_ACCENT2),
]
right = [
    ("Efficiency", [
        "No quantization needed (CPU only)",
        "Batching: process 8 frames in parallel",
        "Caching μ/σ stats avoids re-fitting",
        "Future: TFLite AutoEncoder for edge",
    ], C_ACCENT3),
]
t = 1.5
for head, items, col in left:
    add_text(sl, head, 0.4, t, 6.0, 0.35, size=14, bold=True, color=col)
    for it in items:
        t += 0.3
        add_text(sl, "▸  " + it, 0.55, t, 5.8, 0.28, size=11, color=C_TEXT)
    t += 0.45

t = 1.5
for head, items, col in right:
    add_text(sl, head, 7.1, t, 5.8, 0.35, size=14, bold=True, color=col)
    for it in items:
        t += 0.3
        add_text(sl, "▸  " + it, 7.25, t, 5.6, 0.28, size=11, color=C_TEXT)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — Productization
# ════════════════════════════════════════════════════════════════════════════
sl = slide(); fill_bg(sl)
topic_banner(sl, 1, "Productization", "How it was served and consumed")
steps = [
    ("Edge Device", "OpenCV pipeline runs on Jetson Nano / Pi 5", C_ACCENT1),
    ("REST API", "FastAPI endpoint: POST /analyze-frame → {score, flag}", C_ACCENT2),
    ("Alert Bus", "Flagged frames published to Kafka topic", C_ACCENT3),
    ("Dashboard", "Grafana live view: score time-series + overlay", C_ACCENT1),
    ("Feedback", "Analysts confirm/reject → label pool for fine-tuning", C_ACCENT2),
]
for si, (head, detail, col) in enumerate(steps):
    bx = 0.4 + (si % 3) * 4.2
    ty = 1.55 if si < 3 else 3.8
    add_rect(sl, bx, ty, 3.9, 1.85, C_PANEL)
    add_rect(sl, bx, ty, 0.1, 1.85, col)
    add_text(sl, head, bx+0.2, ty+0.12, 3.5, 0.35, size=13, bold=True, color=col)
    add_text(sl, detail, bx+0.2, ty+0.55, 3.5, 1.1, size=11, color=C_TEXT)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — Impact & Reflection
# ════════════════════════════════════════════════════════════════════════════
sl = slide(); fill_bg(sl)
topic_banner(sl, 1, "Impact & Reflection", "Outcome, problems encountered, what I'd redo")
add_text(sl, "Outcomes", 0.4, 1.4, 5.8, 0.35, size=15, bold=True, color=C_ACCENT2)
for i, it in enumerate(["78% reduction in manual review hours", "Mean alert latency: 1.8 s", "AUC-ROC: 0.94 on held-out clips", "Deployed on 3 simulated camera feeds"]):
    add_text(sl, "✓  " + it, 0.4, 1.85+i*0.38, 5.8, 0.3, size=12, color=C_TEXT)

add_text(sl, "What I'd Redo", 7.0, 1.4, 5.9, 0.35, size=15, bold=True, color=C_ACCENT3)
for i, it in enumerate(["Replace frame-diff with lightweight autoencoder", "Add optical flow (Farneback) for motion direction", "Online learning to adapt to scene changes", "Add explainability: highlight anomalous region"]):
    add_text(sl, "⟳  " + it, 7.0, 1.85+i*0.38, 5.9, 0.3, size=12, color=C_TEXT)

add_rect(sl, 0.4, 4.1, 12.5, 0.04, C_PANEL)
add_text(sl, "Key lesson: 'Good enough' statistical baseline ships faster than a deep model. Start simple, measure, then upgrade.", 0.4, 4.25, 12.5, 0.5, size=12, color=C_SUBTEXT, italic=True, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 13 — Llama 3.2 Architecture (Slide 1/2)
# ════════════════════════════════════════════════════════════════════════════
sl = slide(); fill_bg(sl)
topic_banner(sl, 2, "Llama 3.2 1B — Architecture", "Decoder-only causal transformer, 1B parameters")
# Left column: architecture stack
arch_items = [
    ("Tokenizer (BPE, 128k vocab)", C_ACCENT1),
    ("Token Embeddings  d_model=2048", C_ACCENT2),
    ("× 16 Decoder Layers:", C_TEXT),
    ("  RMSNorm  (pre-norm)", C_SUBTEXT),
    ("  Multi-Head Attention (GQA, 32 heads, 8 KV)", C_ACCENT1),
    ("    — Rotary Positional Encoding (RoPE)", C_SUBTEXT),
    ("    — Q/K/V linear projections", C_SUBTEXT),
    ("    — Causal mask (autoregressive)", C_SUBTEXT),
    ("  RMSNorm", C_SUBTEXT),
    ("  FFN: SwiGLU (gate × up) → down", C_ACCENT2),
    ("Output Head: Linear → Softmax → token", C_ACCENT3),
]
t = 1.45
for txt, col in arch_items:
    add_text(sl, txt, 0.4, t, 6.5, 0.28, size=11, color=col)
    t += 0.3

# Right column: param breakdown
add_text(sl, "Parameter Budget (≈1B)", 7.2, 1.45, 5.7, 0.35, size=14, bold=True, color=C_ACCENT1)
params = [("Embedding table", "128k × 2048", "262M"), ("Attn projections × 16", "4 × 2048²/16", "~134M"), ("FFN × 16 (SwiGLU)", "3 × 2048 × 5504", "~542M"), ("RMSNorm, heads", "—", "~12M"), ("Total", "", "~950M")]
for pi, (name, formula, val) in enumerate(params):
    bg = C_PANEL if pi % 2 == 0 else C_BG
    add_rect(sl, 7.2, 1.95+pi*0.68, 5.7, 0.62, bg)
    add_text(sl, name, 7.3, 2.0+pi*0.68, 2.8, 0.28, size=11, bold=pi==4, color=C_TEXT)
    add_text(sl, formula, 10.2, 2.0+pi*0.68, 1.5, 0.28, size=9, color=C_SUBTEXT, italic=True)
    add_text(sl, val, 11.8, 2.0+pi*0.68, 1.0, 0.28, size=12, bold=pi==4, color=C_ACCENT2, align=PP_ALIGN.RIGHT)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 14 — Llama 3.2 The Math (Slide 2/2)
# ════════════════════════════════════════════════════════════════════════════
sl = slide(); fill_bg(sl)
topic_banner(sl, 2, "Llama 3.2 — The Underlying Math", "Formulas for every key component")
math_blocks = [
    ("RMSNorm", "RMSNorm(x) = x / RMS(x) · γ,   RMS(x) = √(1/d Σxᵢ²)\nNo mean subtraction → faster than LayerNorm", C_ACCENT1),
    ("RoPE (Rotary Position)", "q̃ₘ = Rotate(qₘ, mθ),  kₙ = Rotate(kₙ, nθ)\nθᵢ = 10000^(−2i/d)  →  relative position via dot product", C_ACCENT2),
    ("Causal Self-Attention (GQA)", "Attn(Q,K,V) = softmax(QKᵀ / √dₖ + mask) · V\nGrouped-Query: 32 Q-heads share 8 KV-heads → 4× KV cache saving", C_ACCENT1),
    ("SwiGLU FFN", "FFN(x) = (W_up · x ⊙ SiLU(W_gate · x)) · W_down\nSiLU(x) = x · σ(x) — gated, smoother than ReLU", C_ACCENT2),
    ("Output & Generation", "logits = x_final · Wᵀ_embed  (weight-tied)\nnext token = argmax(softmax(logits / T))  [greedy / top-p]", C_ACCENT3),
]
t = 1.42
for head, body, col in math_blocks:
    add_rect(sl, 0.4, t, 12.5, 0.04, col)
    add_text(sl, head, 0.4, t+0.07, 12.5, 0.3, size=13, bold=True, color=col)
    add_text(sl, body, 0.4, t+0.38, 12.5, 0.55, size=11, color=C_TEXT)
    t += 1.08

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 15 — Enterprise AI Threat Surface
# ════════════════════════════════════════════════════════════════════════════
sl = slide(); fill_bg(sl)
topic_banner(sl, 3, "Enterprise AI Threat Surface", "Where an LLM deployment is exposed")
quadrants = [
    ("🔧 Build / Supply Chain", ["Poisoned base weights (Hugging Face)", "Malicious LoRA adapters / plugins", "Dependency confusion (pip, npm)", "Insecure model serialisation (pickle)"], C_ACCENT1, 0.4, 1.4),
    ("💾 Data Layer", ["Prompt/response leakage → PII exfiltration", "RAG source exposure (internal docs)", "Training data poisoning (backdoor)", "Embedding inversion attacks"], C_ACCENT2, 6.9, 1.4),
    ("⚡ Runtime", ["Prompt injection (direct & indirect)", "Jailbreaks / goal hijacking", "Model denial-of-service (long prompts)", "Insecure output (code exec, XSS, SQLi)"], C_ACCENT3, 0.4, 4.0),
    ("🤖 Agentic / Tool Use", ["Tool misuse (rm -rf via code interpreter)", "Memory poisoning across turns", "Excessive privilege (no least-privilege)", "MCP server compromise → lateral movement"], C_ACCENT1, 6.9, 4.0),
]
for head, items, col, bx, ty in quadrants:
    add_rect(sl, bx, ty, 6.1, 2.8, C_PANEL)
    add_rect(sl, bx, ty, 6.1, 0.38, col)
    add_text(sl, head, bx+0.1, ty+0.03, 5.9, 0.32, size=13, bold=True, color=C_BG)
    for ii, it in enumerate(items):
        add_text(sl, "• " + it, bx+0.15, ty+0.5+ii*0.48, 5.8, 0.4, size=10.5, color=C_TEXT)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 16 — My Project's Exposure
# ════════════════════════════════════════════════════════════════════════════
sl = slide(); fill_bg(sl)
topic_banner(sl, 3, "How My Project Was Exposed", "Applying the threat surface to the anomaly detector")
rows_exp = [
    ("Supply Chain", "OpenCV + NumPy from PyPI", "Medium", "Pinned versions + hash verification", C_ACCENT2),
    ("Data", "Raw CCTV frames — no PII labelled", "Low", "Frames processed in-memory, not stored", C_ACCENT2),
    ("Runtime", "REST API accepts raw video bytes", "High", "Input validation, size cap; no eval() of model output", C_ACCENT3),
    ("Agentic", "Not applicable (no tool/agent layer)", "None", "N/A — simple inference endpoint", C_ACCENT2),
    ("Model", "Static model (μ,σ) — no user influence", "Low", "Normal profile refreshed on trusted feeds only", C_ACCENT2),
]
hdrs = ["Threat Area", "Exposure", "Risk", "Mitigation"]
for hi, h in enumerate(hdrs):
    add_text(sl, h, [0.4,3.2,6.5,8.0][hi], 1.4, [2.7,3.2,1.4,4.3][hi], 0.3, size=13, bold=True, color=C_ACCENT1)
add_rect(sl, 0.4, 1.75, 12.5, 0.03, C_PANEL)
for ri, (area, exp, risk, mit, col) in enumerate(rows_exp):
    bg = C_PANEL if ri%2==0 else C_BG
    add_rect(sl, 0.4, 1.82+ri*0.78, 12.5, 0.72, bg)
    add_text(sl, area, 0.5,  1.86+ri*0.78, 2.6, 0.55, size=11, bold=True, color=col)
    add_text(sl, exp,  3.2,  1.86+ri*0.78, 3.2, 0.55, size=11, color=C_TEXT)
    add_text(sl, risk, 6.5,  1.86+ri*0.78, 1.4, 0.55, size=11, bold=True, color=col)
    add_text(sl, mit,  8.0,  1.86+ri*0.78, 4.7, 0.55, size=10, color=C_SUBTEXT)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 17 — Cisco AI Defense vs Palo Alto AIRS
# ════════════════════════════════════════════════════════════════════════════
sl = slide(); fill_bg(sl)
topic_banner(sl, 3, "AI Security Products — Reasoning", "Cisco AI Defense vs Palo Alto Prisma AIRS")
add_text(sl, "My reasoning approach (not product memorization)", 0.4, 1.4, 12.5, 0.35, size=14, bold=True, color=C_ACCENT1)
add_text(sl, "Both products sit between the AI deployment and the threat surface — they instrument, inspect and enforce. Key question: which layer?", 0.4, 1.8, 12.5, 0.4, size=11, color=C_SUBTEXT, italic=True)

compare = [
    ("Dimension", "Cisco AI Defense", "Palo Alto Prisma AIRS", True),
    ("Primary layer", "Network/API gateway — inspects LLM I/O at the transport layer", "Platform-wide: covers model registry, runtime & agentic chain", False),
    ("Prompt injection", "Detects injected instructions in HTTP payloads; inline blocking", "Contextual analysis of multi-turn sessions; intent classification", False),
    ("Data leakage", "DLP on model responses (regex + ML classifier)", "RAG-aware: traces retrieval source to flag sensitive doc exposure", False),
    ("Agentic / MCP", "Limited — focused on API calls", "MCP server policy enforcement; tool-call allow-listing", False),
    ("Deployment", "On-prem appliance or SASE integration", "Cloud-native SaaS; SDK for LangChain/custom agents", False),
    ("My take", "Strong for org already on Cisco stack; fastest time-to-value for API layer", "Better for orgs building agentic systems; deeper model-level visibility", False),
]
for ri, row in enumerate(compare):
    bg = C_PANEL if ri%2==0 else C_BG
    add_rect(sl, 0.4, 2.4+ri*0.63, 12.5, 0.58, bg)
    cols_w = [2.5, 4.8, 4.8]
    cols_x = [0.5, 3.1, 8.0]
    for ci, val in enumerate(row[:3]):
        add_text(sl, val, cols_x[ci], 2.45+ri*0.63, cols_w[ci], 0.5, size=10.5,
                 bold=row[3] or ci==0, color=C_ACCENT1 if row[3] else (C_TEXT if ci>0 else C_ACCENT2))

prs.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
print(f"Slides: {len(prs.slides)}")
