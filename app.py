"""Household Object Recognition — A2 demo app (Gradio).

upload / camera  ->  prediction  ->  Grad-CAM + nearest neighbours  ->  latency + environment
Run:  python app.py --bundle app_bundle            (local, http://127.0.0.1:7860)
      python app.py --bundle app_bundle --share    (temporary public link, e.g. from Colab)
"""
import argparse, io, json, os, platform, time
from collections import deque
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from scipy import ndimage
from torchvision import models, transforms

# ----------------------------------------------------------------------------- model bundle
parser = argparse.ArgumentParser()
parser.add_argument("--bundle", default=os.environ.get("APP_BUNDLE", "app_bundle"))
parser.add_argument("--share", action="store_true")
parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
args, _ = parser.parse_known_args()
B = Path(args.bundle)
META = json.load(open(B / "meta.json"))
CLASSES, ARCH = META["classes"], META["arch"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build(arch, n):
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, n))
        return m, m.layer4[-1], nn.Sequential(*list(m.children())[:-1])
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, n))
    return m, m.features[-1], nn.Sequential(m.features, m.avgpool)


MODEL, CAM_LAYER, BODY = build(ARCH, len(CLASSES))
MODEL.load_state_dict(torch.load(B / "model.pt", map_location="cpu"))
MODEL.to(DEVICE).eval()
TF_EVAL = transforms.Compose([transforms.Resize(META["img_size"]), transforms.CenterCrop(META["img_size"]),
                              transforms.ToTensor(), transforms.Normalize(META["mean"], META["std"])])

GAL_E = np.load(B / "embeddings.npy")
GAL = json.load(open(B / "gallery.json"))
SYN = json.load(open(B / "synthetic.json")) if (B / "synthetic.json").exists() else []

# ----------------------------------------------------------------------------- Grad-CAM
_acts, _grads = {}, {}
CAM_LAYER.register_forward_hook(lambda m, i, o: _acts.__setitem__("a", o.detach()))
CAM_LAYER.register_full_backward_hook(lambda m, gi, go: _grads.__setitem__("g", go[0].detach()))


def overlay(img224, cam):
    """Jet-like colour map blended over the image, no matplotlib needed."""
    c = np.clip(cam, 0, 1)
    rgb = np.stack([np.clip(1.5 - abs(4 * c - 3), 0, 1), np.clip(1.5 - abs(4 * c - 2), 0, 1),
                    np.clip(1.5 - abs(4 * c - 1), 0, 1)], -1)
    base = np.asarray(img224, np.float32) / 255
    return Image.fromarray((255 * (0.55 * base + 0.45 * rgb)).astype(np.uint8))


# ----------------------------------------------------------------------------- stress-test corruptions (same as notebook)
def motion_blur(img, length=21, angle=0.0):
    k = np.zeros((length, length), np.float32); k[length // 2, :] = 1
    k = np.asarray(Image.fromarray((k * 255).astype(np.uint8)).rotate(angle, resample=Image.BILINEAR), np.float32)
    k /= k.sum(); a = np.asarray(img, np.float32)
    return Image.fromarray(np.stack([ndimage.convolve(a[..., c], k, mode="reflect") for c in range(3)], -1)
                           .clip(0, 255).astype(np.uint8))


def low_light(img, seed=0):
    a = np.asarray(ImageEnhance.Brightness(img).enhance(0.3), np.float32)
    a += np.random.default_rng(seed).normal(0, 8, a.shape)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def jpeg(img, q=8):
    b = io.BytesIO(); img.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")


def occlude(img, seed=0):
    rng = np.random.default_rng(seed); w, h = img.size; img = img.copy(); d = ImageDraw.Draw(img)
    bw, bh = int(w * 0.45), int(h * 0.45); x, y = rng.integers(0, w - bw), rng.integers(0, h - bh)
    d.rectangle([x, y, x + bw, y + bh], fill=(128, 128, 128)); return img


def perspective(img, s=0.35):
    w, h = img.size; dx, dy = int(w * s / 2), int(h * s / 2)
    return TF.perspective(img, [[0, 0], [w, 0], [w, h], [0, h]], [[dx, dy], [w - dx, 0], [w, h], [0, h - dy]], fill=0)


CORRUPTIONS = {
    "none (clean)": lambda im: im,
    "motion blur": lambda im: motion_blur(im, 21, 30),
    "defocus blur": lambda im: im.filter(ImageFilter.GaussianBlur(4)),
    "low light + noise": low_light,
    "rotation 60°": lambda im: im.rotate(60, expand=True),
    "viewpoint (perspective)": perspective,
    "occlusion 45%": occlude,
    "JPEG q=8": jpeg,
}

# ----------------------------------------------------------------------------- inference
LAT = deque(maxlen=200)


def env_text(request):
    ua = (request.headers.get("user-agent", "") if request is not None else "") or "unknown"
    browser = next((b for b in ("Edg", "OPR", "Chrome", "Firefox", "Safari") if b in ua), "unknown browser")
    browser = {"Edg": "Edge", "OPR": "Opera"}.get(browser, browser)
    mobile = "mobile" if any(k in ua for k in ("Mobile", "Android", "iPhone")) else "desktop"
    dev = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else f"CPU ({platform.machine()}, {os.cpu_count()} threads)"
    return (f"**Server:** {dev} · {platform.system()} · torch {torch.__version__}  \n"
            f"**Your browser:** {browser} ({mobile})")


def run(img):
    img = ImageOps.exif_transpose(img).convert("RGB")
    x = TF_EVAL(img).unsqueeze(0).to(DEVICE)
    t0 = time.perf_counter()
    with torch.no_grad():
        prob = MODEL(x).softmax(1)[0].cpu().numpy()
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    LAT.append(ms)
    # Grad-CAM for the predicted class (separate pass, not counted in latency)
    x.requires_grad_(True); MODEL.zero_grad()
    out = MODEL(x); c = int(out.argmax(1)); out[0, c].backward()
    a, g = _acts["a"], _grads["g"]
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).detach().cpu().numpy()
    shown = transforms.CenterCrop(META["img_size"])(transforms.Resize(META["img_size"])(img))
    # nearest neighbours in the training set (real + synthetic)
    with torch.no_grad():
        e = F.normalize(BODY(x.detach()).flatten(1), dim=1)[0].cpu().numpy()
    sims = GAL_E @ e
    nn_items = [(str(B / GAL[i]["file"]),
                 f"{GAL[i]['label']} · {'SYNTHETIC' if GAL[i]['synthetic'] else 'real'} · sim {sims[i]:.2f}")
                for i in np.argsort(-sims)[:6]]
    return {CLASSES[i]: float(p) for i, p in enumerate(prob)}, overlay(shown, cam), nn_items, ms


def stats(ms, request):
    avg = float(np.mean(LAT))
    return (f"**Latency:** {ms:.1f} ms this image · **average {avg:.1f} ms** over {len(LAT)} requests "
            f"(≈ {1000 / avg:.0f} FPS, batch 1)  \n" + env_text(request))


def predict(img, request: gr.Request):
    if img is None:
        return None, None, None, "Upload a photo or take one with the camera."
    label, cam, nn_items, ms = run(img)
    return label, cam, nn_items, stats(ms, request)


def stress(img, corruption, request: gr.Request):
    if img is None:
        return None, None, None, "Upload a photo first."
    bad = CORRUPTIONS[corruption](ImageOps.exif_transpose(img).convert("RGB"))
    label, cam, _, ms = run(bad)
    return bad, label, cam, stats(ms, request)


# ----------------------------------------------------------------------------- UI
cv = META.get("cv_metrics", {})
cv_line = ""
if cv:
    cv_line = (f"Model: **{ARCH}**, trained on {META['trained_on']['real']} real photos + "
               f"{META['trained_on']['synthetic']} **synthetic** video frames. "
               f"5-fold CV on real photos: clean accuracy {cv['clean']['accuracy']:.1%}, "
               f"stress-test accuracy {cv['stress_all']['accuracy']:.1%}.")

with gr.Blocks(title="Household Object Recognition — A2") as demo:
    gr.Markdown("# Household object recognition: bottle · cup · key · pen\n" + cv_line)
    with gr.Tab("Recognize"):
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(sources=["upload", "webcam", "clipboard"], type="pil", label="Photo or camera")
                btn = gr.Button("Recognize", variant="primary")
                ex = sorted(str(p) for p in (B / "examples").glob("*.jpg")) if (B / "examples").exists() else []
                if ex: gr.Examples(ex, inp)
            with gr.Column(scale=1):
                lab = gr.Label(num_top_classes=4, label="Prediction")
                cam = gr.Image(label="Grad-CAM: where the model looked", type="pil")
        nn_gal = gr.Gallery(label="Nearest neighbours in the training set", columns=6, height=220)
        info = gr.Markdown()
        btn.click(predict, inp, [lab, cam, nn_gal, info])
        inp.upload(predict, inp, [lab, cam, nn_gal, info])
    with gr.Tab("Stress test"):
        gr.Markdown("Apply a phone-style corruption and see whether the prediction survives.")
        with gr.Row():
            s_in = gr.Image(sources=["upload", "webcam", "clipboard"], type="pil", label="Photo")
            s_bad = gr.Image(label="Corrupted input", type="pil")
            s_cam = gr.Image(label="Grad-CAM", type="pil")
        with gr.Row():
            s_kind = gr.Dropdown(list(CORRUPTIONS), value="motion blur", label="Corruption")
            s_btn = gr.Button("Run", variant="primary")
        s_lab = gr.Label(num_top_classes=4, label="Prediction")
        s_info = gr.Markdown()
        s_btn.click(stress, [s_in, s_kind], [s_bad, s_lab, s_cam, s_info])
    with gr.Tab("Synthetic data (AI-generated)"):
        gr.Markdown("### ⚠️ SYNTHETIC MEDIA — generated offline with CogVideoX-2B (text-to-video)\n"
                    "These clips were generated from text prompts; they do not show real objects or places. "
                    "Frames sampled from such clips (after a CLIP quality check) were added to the training set. "
                    "Every clip carries a red **SYNTHETIC** watermark.")
        for k in range(0, len(SYN), 2):
            with gr.Row():
                for s in SYN[k:k + 2]:
                    with gr.Column():
                        gr.Video(str(B / s["file"]), label=f"SYNTHETIC · {s['label']}", autoplay=False)
                        gr.Markdown(f"*Prompt:* {s['prompt']}")
    with gr.Tab("About & ethics"):
        gr.Markdown(
            "**Data & consent.** Real photos were taken by the authors of their own household objects; no people or "
            "personal documents appear. Uploaded photos are processed in memory and not stored.\n\n"
            "**Synthetic media.** Synthetic clips are clearly labelled and watermarked; they are used for training "
            "only and never for testing.\n\n"
            "**Limitations.** Small dataset from one home and one phone; four classes only — any other object is "
            "forced into one of them; confidence is not a guarantee.\n\n"
            "**Model card.**\n" + "\n".join(f"- {k}: `{META[k]}`" for k in ('arch', 'condition', 't2v_model', 'clip_model', 'trained_on') if k in META))

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share, allowed_paths=[str(B.resolve())])
