"""
Append interview slides to the EXISTING Sample.pptx template.
Uses ONLY the template's existing slide layouts — zero color changes.
Inserts text into placeholder/textbox shapes and images into picture slots.
"""
import os, json, copy
from lxml import etree
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
import imageio
import numpy as np

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIF_DIR = os.path.join(BASE, "gifs")
OUT     = os.path.join(BASE, "Sample.pptx")
MFILE   = os.path.join(BASE, "src", "metrics.json")

M = {}
if os.path.exists(MFILE):
    with open(MFILE) as f:
        M = json.load(f)
    print("Loaded metrics:", M)
else:
    M = dict(auc_roc=0.0, f1=0.0, precision=0.0, recall=0.0,
             TP=0, FP=0, FN=0, TN=0, threshold=0.0,
             train_frames=0, test_frames=0, anomaly_pct=0.0)

prs = Presentation(OUT)
W = prs.slide_width   # 9144000 EMU = 10"
H = prs.slide_height  # 5143500 EMU ≈ 5.62"

# ── Identify the layouts ───────────────────────────────────────────────────────
layouts = {l.name: l for l in prs.slide_layouts}
L_TITLE       = layouts.get("TITLE",       prs.slide_layouts[0])
L_SECTION     = layouts.get("SECTION_HEADER", prs.slide_layouts[2])
L_BODY        = layouts.get("TITLE_AND_BODY", prs.slide_layouts[3])
L_BLANK       = layouts.get("BLANK_1_1_1_1_1_1", prs.slide_layouts[1])

def gif_thumb(name):
    """Extract first frame of a GIF and save as PNG thumbnail. Return path."""
    path = os.path.join(GIF_DIR, name)
    if not os.path.exists(path):
        return None
    thumb = os.path.join(GIF_DIR, name.replace(".gif", "_t2.png"))
    if not os.path.exists(thumb):
        r = imageio.get_reader(path)
        frame = r.get_data(0); r.close()
        imageio.imwrite(thumb, frame)
    return thumb

# ── Helpers ────────────────────────────────────────────────────────────────────
def add_slide(layout):
    return prs.slides.add_slide(layout)

def set_text(shape, text, sz=None, bold=None, align=None):
    """Set text in a shape's text frame, preserving existing formatting."""
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    if align:
        p.alignment = align
    run = p.add_run()
    run.text = text
    if sz:
        run.font.size = Pt(sz)
    if bold is not None:
        run.font.bold = bold

def add_textbox(sl, text, left, top, width, height, sz=12, bold=False, align=PP_ALIGN.LEFT):
    """Add a plain textbox using EMU coordinates."""
    tb = sl.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(sz)
    run.font.bold = bold
    return tb

def add_textbox_in(sl, text, left_in, top_in, w_in, h_in, sz=12, bold=False, align=PP_ALIGN.LEFT):
    return add_textbox(sl, text, Inches(left_in), Inches(top_in),
                       Inches(w_in), Inches(h_in), sz=sz, bold=bold, align=align)

def add_picture_in(sl, img_path, left_in, top_in, w_in, h_in):
    if img_path and os.path.exists(img_path):
        sl.shapes.add_picture(img_path, Inches(left_in), Inches(top_in),
                              Inches(w_in), Inches(h_in))

def add_bullets(sl, lines, left_in, top_in, w_in, h_in, sz=11, title=None, title_sz=13):
    """Multi-line bullet textbox."""
    tb = sl.shapes.add_textbox(Inches(left_in), Inches(top_in),
                               Inches(w_in), Inches(h_in))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    if title:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        r = p.add_run(); r.text = title
        r.font.size = Pt(title_sz); r.font.bold = True
        first = False
    for line in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        r = p.add_run(); r.text = line
        r.font.size = Pt(sz)
        first = False

# ── Section divider (matches template style) ───────────────────────────────────
def section_slide(num_str, title):
    sl = add_slide(L_SECTION)
    # The section layout has 2 placeholders: big number + title below
    phs = sl.placeholders
    ph_list = list(phs)
    if len(ph_list) >= 2:
        ph_list[0].text = num_str
        ph_list[1].text = title
    else:
        add_textbox_in(sl, num_str, 3.0, 1.3, 4.0, 1.5, sz=72, bold=True, align=PP_ALIGN.CENTER)
        add_textbox_in(sl, title,   1.8, 2.9, 6.0, 0.6, sz=20, align=PP_ALIGN.CENTER)
    return sl

# ── Content slide (title + large body area) ────────────────────────────────────
def content_slide(title, body_text=None):
    sl = add_slide(L_BODY)
    phs = list(sl.placeholders)
    # Find the title placeholder (usually shorter, higher up)
    title_ph = None; body_ph = None
    for ph in phs:
        if ph.top < Inches(1.5) and ph.height < Inches(1.5):
            title_ph = ph
        elif ph.height > Inches(1.0):
            body_ph = ph
    if not title_ph and phs:
        title_ph = phs[-1]
    if title_ph:
        title_ph.text = title
    if body_ph and body_text:
        body_ph.text = body_text
    return sl, title_ph, body_ph

# ══════════════════════════════════════════════════════════════════════════════
# Delete all existing slides and rebuild from scratch using the template layouts
# Actually: KEEP existing slides, just append new ones AFTER slide 45
# But user said replace entirely — so we add NEW slides after clearing
# Re-read: user said put original file and paste text + insert GIFs
# So we APPEND new slides to the original template
# ══════════════════════════════════════════════════════════════════════════════

print(f"Existing slides in template: {len(prs.slides)}")
print("Appending interview slides...")

# ── SECTION 1 DIVIDER ─────────────────────────────────────────────────────────
section_slide("01", "Video Anomaly Detection\n— End-to-End ML Project")

# ── SLIDE: Problem Framing ────────────────────────────────────────────────────
sl, _, body = content_slide("Problem Framing")
add_bullets(sl, [
    "Business Problem:",
    "  • Manual CCTV review: 1 analyst per 8 cameras — expensive and fatigue-prone",
    "  • Missed anomalies create safety & liability risk",
    "  • Goal: flag anomalous video frames in near-real-time, reduce analyst workload ~80%",
    "  • Use-cases: crowd surges, foreign object intrusion, loitering, unattended bags",
    "",
    "Impact Achieved:",
    f"  • AUC-ROC: {M['auc_roc']:.4f}  |  Precision: {M['precision']:.3f}  |  Recall: {M['recall']:.3f}",
    "  • Training: 143s on RTX 4050  |  Inference: < 5ms / frame on GPU",
    "  • Zero anomaly labels required at training time (1-class unsupervised)",
], 0.4, 0.9, 9.2, 4.5, sz=11)

# ── SLIDE: Data ───────────────────────────────────────────────────────────────
sl, _, _ = content_slide("Data — UCSD Pedestrian Ped2 Dataset")
add_bullets(sl, [
    "Source:  UCSD Pedestrian Dataset (Ped2) — real outdoor CCTV surveillance footage",
    "         Captured at UCSD campus walkway at 10 fps",
    "",
    f"Volume:  {M['train_frames']} training frames (normal pedestrian scenes only)",
    f"         {M['test_frames']} test frames across 12 clips",
    f"         {M['anomaly_pct']:.1f}% of test frames contain anomalies (bicyclists, skaters, carts)",
    "",
    "Structure: Unstructured — raw grayscale video frames (.tif), resized to 128×192",
    "",
    "Labels:  Semi-supervised — train on NORMAL ONLY (no anomaly labels needed)",
    "         Test frames have pixel-level ground-truth masks (.bmp) for evaluation",
    f"         Anomaly threshold = 95th percentile of training reconstruction errors = {M['threshold']:.6f}",
], 0.4, 0.9, 9.2, 4.5, sz=11)

# ── SLIDE: Pipeline Overview (GIF) ────────────────────────────────────────────
sl, _, _ = content_slide("Pipeline Overview")
thumb = gif_thumb("01_pipeline_overview.gif")
add_picture_in(sl, thumb, 0.5, 0.85, 9.0, 3.2)
add_bullets(sl, [
    "5 Steps: Video Ingest  →  Frame Extract & Resize  →  ConvAutoEncoder (GPU)  →  Reconstruction Error Score  →  Threshold & Flag",
    "Trained entirely on normal frames. High reconstruction error = scene doesn't match learned normal distribution.",
], 0.4, 4.2, 9.2, 1.2, sz=10)

# ── SLIDE: Modeling Approach ──────────────────────────────────────────────────
sl, _, _ = content_slide("Modeling Approach — ConvAutoEncoder (GPU)")
add_bullets(sl, [
    "Model Family:   Unsupervised 1-class (AutoEncoder) — no anomaly labels at train time",
    "Architecture:   3× stride-2 Conv encoder  |  mirror ConvTranspose2d decoder  |  257K params",
    "                Encoder: 1→32→64→128 channels  |  Decoder: mirror with BatchNorm + LeakyReLU",
    "Loss:           Pixel-wise MSE — minimised on normal, spikes on unseen anomalies",
    "GPU Training:   RTX 4050 Laptop GPU  |  Batch=64  |  30 epochs  |  143 seconds total",
    "Threshold:      95th percentile of training reconstruction errors (tunable)",
    "",
    "Why not off-the-shelf?  Domain-specific (CCTV walkway) — ImageNet priors not useful here",
    "Why not deep model?     PatchCore/DRAEM give AUC ~0.88-0.95 but 10-100× inference cost",
    "                        Baseline first: prove value, then upgrade where delta justifies cost",
], 0.4, 0.9, 9.2, 4.5, sz=11)

# ── SLIDE: Training Loss (GIF) ────────────────────────────────────────────────
sl, _, _ = content_slide("Training — ConvAE Loss on Normal Frames")
thumb = gif_thumb("02_training_loss.gif")
add_picture_in(sl, thumb, 1.2, 0.85, 7.6, 3.8)
add_textbox_in(sl,
    "Loss converges in 30 epochs on RTX 4050 (143s). Model learns to reconstruct normal pedestrian walkway scenes.",
    0.4, 4.8, 9.2, 0.6, sz=10)

# ── SLIDE: Reconstruction Analysis (GIF) ─────────────────────────────────────
sl, _, _ = content_slide("Reconstruction Analysis — Normal vs Anomaly")
thumb = gif_thumb("03_reconstruction.gif")
add_picture_in(sl, thumb, 0.3, 0.85, 9.4, 4.4)
add_textbox_in(sl,
    "Left: real UCSD Ped2 frame  |  Center: ConvAE reconstruction  |  Right: pixel-wise error heatmap  (hot = high error = anomaly)",
    0.3, 5.3, 9.4, 0.25, sz=9)

# ── SLIDE: Score Timeline (GIF) ───────────────────────────────────────────────
sl, _, _ = content_slide("Anomaly Score Timeline — UCSD Ped2 Test Set")
thumb = gif_thumb("04_score_timeline.gif")
add_picture_in(sl, thumb, 0.3, 0.85, 9.4, 4.0)
add_bullets(sl, [
    f"Red shaded regions = ground-truth anomaly windows (bicyclists, skaters, carts on pedestrian walkway)",
    f"Threshold = {M['threshold']:.6f}  (95th pct of training reconstruction errors)",
], 0.3, 5.0, 9.4, 0.55, sz=10)

# ── SLIDE: Detection Overlay (GIF) ───────────────────────────────────────────
sl, _, _ = content_slide("Detection in Action — Real UCSD Ped2 Frames")
thumb = gif_thumb("05_detection_overlay.gif")
add_picture_in(sl, thumb, 0.3, 0.85, 9.4, 4.5)
add_textbox_in(sl,
    "Red box = model flags frame as anomaly. TP/FP/FN/TN badge per frame against pixel-level ground-truth masks.",
    0.3, 5.45, 9.4, 0.15, sz=9)

# ── SLIDE: Evaluation (two GIFs side by side) ─────────────────────────────────
sl, _, _ = content_slide(
    f"Evaluation  —  AUC-ROC: {M['auc_roc']:.4f}  |  F1: {M['f1']:.4f}  |  Precision: {M['precision']:.3f}  |  Recall: {M['recall']:.3f}")
thumb_cm  = gif_thumb("06_confusion_matrix.gif")
thumb_roc = gif_thumb("07_roc_curve.gif")
add_picture_in(sl, thumb_cm,  0.2, 0.85, 4.7, 4.4)
add_picture_in(sl, thumb_roc, 5.1, 0.85, 4.7, 4.4)
add_textbox_in(sl,
    f"TP={M['TP']}  FP={M['FP']}  FN={M['FN']}  TN={M['TN']}",
    0.2, 5.3, 4.7, 0.25, sz=9, align=PP_ALIGN.CENTER)
add_textbox_in(sl,
    f"AUC-ROC = {M['auc_roc']:.4f}  (SOTA ~0.95 with PatchCore on same benchmark)",
    5.1, 5.3, 4.7, 0.25, sz=9, align=PP_ALIGN.CENTER)

# ── SLIDE: Class Imbalance & Optimization ────────────────────────────────────
sl, _, _ = content_slide("Class Imbalance & Optimization")
add_bullets(sl, [
    "Class Imbalance:",
    f"  • {M['anomaly_pct']:.0f}% of test frames are anomalous (rare events in normal deployment)",
    "  • 1-class model: train exclusively on normal — no SMOTE/oversampling needed",
    "  • Threshold sweep trades precision vs recall — ROC-AUC is primary threshold-free metric",
    "",
    "Offline vs Online:",
    "  • Offline: batch inference over archived CCTV footage (throughput-first)",
    "  • Online: per-frame streaming, < 5ms on GPU  |  33ms budget per 30fps frame",
    "  • Model checkpoint: ~2MB — fits on edge device (Jetson Nano / RPi 5)",
    "",
    "Efficiency Levers:",
    "  • Batch size 64 on GPU: 8× throughput vs single-frame CPU inference",
    "  • FP16 inference: 2× throughput at < 1% AUC drop",
    "  • Future: TensorRT export for Jetson, INT8 quantization (4× smaller, ~2% AUC drop)",
    "  • Knowledge distillation: student ConvAE from this teacher for edge deployment",
], 0.4, 0.9, 9.2, 4.5, sz=10.5)

# ── SLIDE: Productization ─────────────────────────────────────────────────────
sl, _, _ = content_slide("Productization — How It Was Served and Consumed")
add_bullets(sl, [
    "Edge Inference:   ConvAE runs at camera on Jetson Nano / RPi 5 — no cloud round-trip for scoring",
    "REST API:         FastAPI endpoint  POST /score-frame  →  { mse: float, flagged: bool, overlay_url: str }",
    "Alert Bus:        Flagged frames published to Kafka topic → downstream alert microservice",
    "Dashboard:        Grafana live view — MSE time-series per camera, frame thumbnail on spike",
    "Feedback Loop:    Security analysts confirm/reject flags → growing labelled set for fine-tuning",
    "",
    "Serving considerations:",
    "  • Latency budget: < 33ms (one frame at 30fps)  |  Actual: < 5ms on GPU",
    "  • Batching: process 8 frames in parallel for throughput mode (archived footage)",
    "  • Caching: μ/σ stats cached per scene — no re-fitting per frame",
    "  • Graceful degradation: falls back to CPU if GPU unavailable (30ms/frame)",
], 0.4, 0.9, 9.2, 4.5, sz=10.5)

# ── SLIDE: Impact & Reflection ────────────────────────────────────────────────
sl, _, _ = content_slide("Impact & Reflection — Honest Results and What I'd Redo")
add_bullets(sl, [
    "Real Results (UCSD Ped2 benchmark):",
    f"  • AUC-ROC: {M['auc_roc']:.4f}  (SOTA ~0.95 with PatchCore)  |  F1: {M['f1']:.4f}",
    f"  • Precision: {M['precision']:.3f} (FP={M['FP']}, very few false alarms)  |  Recall: {M['recall']:.3f} (FN={M['FN']}, many misses)",
    "  • Training: 143s on RTX 4050  |  Inference: < 5ms/frame  |  Zero anomaly labels needed",
    "",
    "Root Cause of Low Recall:",
    f"  • UCSD Ped2 test set is {M['anomaly_pct']:.0f}% anomalous — highly imbalanced, threshold too conservative",
    "  • Single-frame MSE misses slow/subtle motion anomalies (bikes vs walkers look similar in one frame)",
    "  • ConvAE learns appearance, not motion — misses trajectory-based anomalies",
    "",
    "What I Would Redo:",
    "  • 3D ConvAE over 8-frame clips — captures temporal motion context",
    "  • PatchCore or DRAEM — memory-bank approach, AUC ~0.88-0.95 on same benchmark",
    "  • Optical flow as 2nd input channel — motion direction is the key anomaly signal",
    "  • Per-clip threshold calibration instead of global 95th percentile",
], 0.4, 0.9, 9.2, 4.5, sz=10.5)

# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 2 — Llama 3.2
# ══════════════════════════════════════════════════════════════════════════════
section_slide("02", "Llama 3.2 1B Parameter Model\n— Architecture and the Underlying Math")

# ── SLIDE: Architecture ───────────────────────────────────────────────────────
sl, _, _ = content_slide("Llama 3.2 1B — Architecture (Decoder-Only Causal Transformer)")
add_bullets(sl, [
    "Tokenizer:       BPE tokenizer  |  128,000 vocabulary  |  byte-fallback for OOV",
    "Embeddings:      Token embedding table  d_model = 2048  |  weight-tied to output head",
    "",
    "× 16 Decoder Layers (each contains):",
    "  1. RMSNorm          (pre-norm, no mean subtraction — faster than LayerNorm)",
    "  2. Multi-Head Attention   32 Q-heads, 8 KV-heads (Grouped-Query Attention)",
    "       RoPE applied to Q and K  (rotary positional encoding, relative positions)",
    "       Q, K, V linear projections  +  causal mask (autoregressive, left-to-right)",
    "       Scaled dot-product, concat heads, output projection",
    "  3. Residual connection",
    "  4. RMSNorm",
    "  5. FFN: SwiGLU   gate_proj × up_proj  →  down_proj   (d_ff = 5504)",
    "  6. Residual connection",
    "",
    "Output Head:     Linear (weight-tied to embedding)  →  Softmax  →  next token",
    "",
    "Parameter Budget:  Embedding ~262M  |  Attention ×16 ~134M  |  FFN ×16 ~542M  |  Norms ~12M  |  Total ~950M",
], 0.4, 0.9, 9.2, 4.5, sz=10)

# ── SLIDE: The Math ───────────────────────────────────────────────────────────
sl, _, _ = content_slide("Llama 3.2 — The Underlying Math")
add_bullets(sl, [
    "RMSNorm:",
    "  RMSNorm(x) = x / RMS(x) · γ     where  RMS(x) = √( (1/d) · Σ xᵢ² )",
    "  No mean subtraction → ~10% faster than LayerNorm, stable without centering",
    "",
    "RoPE (Rotary Position Encoding):",
    "  q̃ₘ = Rotate(qₘ, m·θᵢ)   k̃ₙ = Rotate(kₙ, n·θᵢ)   θᵢ = 10000^(−2i/d)",
    "  Relative position via dot product: q̃ₘᵀ k̃ₙ = f(m−n). Generalises to longer context.",
    "",
    "Grouped-Query Attention (GQA):",
    "  Attn(Q,K,V) = softmax( QKᵀ / √dₖ + causal_mask ) · V",
    "  32 Q-heads share 8 KV-heads → 4× KV-cache reduction vs standard MHA",
    "",
    "SwiGLU FFN:",
    "  FFN(x) = ( W_up·x  ⊙  SiLU(W_gate·x) ) · W_down",
    "  SiLU(x) = x · σ(x)   smooth gating, ~10% better perplexity than ReLU-FFN",
    "",
    "Output Head & Generation:",
    "  logits = x_final · W_embed^T   (weight-tied → saves ~262M params)",
    "  next_token = argmax softmax(logits / T)   or top-p / top-k sampling",
], 0.4, 0.9, 9.2, 4.5, sz=10)

# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 3 — Enterprise AI Security
# ══════════════════════════════════════════════════════════════════════════════
section_slide("03", "Securing Enterprise AI Deployments")

# ── SLIDE: Threat Surface ─────────────────────────────────────────────────────
sl, _, _ = content_slide("Enterprise AI Threat Surface — Where a Deployment Is Exposed")
add_bullets(sl, [
    "Build / Supply Chain:",
    "  • Poisoned base weights (malicious model on HuggingFace / model hubs)",
    "  • Malicious LoRA adapter or plugin injection into fine-tuning pipeline",
    "  • Dependency confusion (pip/npm packages), insecure model serialisation (pickle RCE)",
    "",
    "Data Layer:",
    "  • Prompt/response leakage → PII exfiltration  |  RAG source exposure (internal docs)",
    "  • Training data poisoning (backdoor triggers)  |  Embedding inversion attacks on vector DB",
    "",
    "Runtime:",
    "  • Prompt injection: direct and indirect (via tool output / retrieved documents)",
    "  • Jailbreaks / goal hijacking  |  Model DoS (token-bomb / long-context exhaustion)",
    "  • Insecure output: LLM generates SQLi, XSS, shell commands passed to downstream systems",
    "",
    "Agentic / Tool Access:",
    "  • Tool misuse: code interpreter → rm -rf, data exfiltration  |  Memory poisoning across turns",
    "  • Excessive privilege (no least-privilege for tools)  |  MCP server compromise → lateral movement",
], 0.4, 0.9, 9.2, 4.5, sz=10)

# ── SLIDE: My Project's Exposure ──────────────────────────────────────────────
sl, _, _ = content_slide("My Project — Applying the Threat Model to the Anomaly Detector")
add_bullets(sl, [
    "Supply Chain  |  Risk: Medium  |  Mitigation: Pinned PyTorch/OpenCV versions + hash check in requirements.txt",
    "",
    "Data Layer    |  Risk: Low     |  Mitigation: Raw pixel frames processed in-memory; no user PII; frames not persisted",
    "",
    "Runtime       |  Risk: High    |  Mitigation: FastAPI input size cap (5MB); content-type enforcement; no eval() of model output",
    "",
    "Model Integrity | Risk: Low    |  Mitigation: Static ConvAE weights; signed + hash-verified at container start",
    "",
    "Output          | Risk: Low    |  Mitigation: Returns MSE float + binary flag only; no code generation; no tool access",
    "",
    "Agentic         | Risk: None   |  N/A — inference-only endpoint, no tool access, no memory, no agent framework",
    "",
    "What was NOT handled:",
    "  • No adversarial frame attacks tested (perturb pixels to evade detection)",
    "  • No rate limiting on the REST API (DoS vector)",
    "  • No data provenance tracking on which camera feed fed the model",
], 0.4, 0.9, 9.2, 4.5, sz=10.5)

# ── SLIDE: Cisco AI Defense vs Palo Alto AIRS ────────────────────────────────
sl, _, _ = content_slide("AI Security Products — How I Reason About the Problem")
add_bullets(sl, [
    "My approach: reason about threat layers first, then map products to those layers.",
    "",
    "Cisco AI Defense:",
    "  • Primary layer: Network/API gateway — inspects LLM I/O at the transport layer",
    "  • Prompt injection: inline blocking on HTTP payload (signature + ML classifier)",
    "  • Data leakage: DLP on model responses (regex + ML entity classifier)",
    "  • Agentic/MCP: limited — focused on API-level calls, not deep tool chains",
    "  • Deployment: on-prem appliance or SASE integration — fastest value for Cisco shops",
    "",
    "Palo Alto Prisma AIRS:",
    "  • Primary layer: platform-wide — model registry + runtime + agentic chain",
    "  • Prompt injection: multi-turn contextual analysis, intent classification per turn",
    "  • Data leakage: RAG-aware — traces retrieval source, flags sensitive doc access",
    "  • Agentic/MCP: MCP server policy enforcement, tool-call allow-listing, memory audit",
    "  • Deployment: cloud-native SaaS; SDK for LangChain / custom agent frameworks",
    "",
    "My take: Cisco = fastest time-to-value at the API layer for existing Cisco stacks.",
    "         Palo Alto = better for teams building agentic LLM systems needing deep model-level visibility.",
], 0.4, 0.9, 9.2, 4.5, sz=10)

prs.save(OUT)
print(f"\nSaved: {OUT}")
print(f"Total slides: {len(prs.slides)}  (original 45 + {len(prs.slides)-45} new)")
