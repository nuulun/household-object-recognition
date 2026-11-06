# Household Object Recognition: bottle · cup · key · pen

Computer Vision course project, Astana IT University. Authors: **Nurlan Bekov, Raiymbek Mussin**.

* **Assignment 1:** a self-collected iPhone dataset, classical baselines (HOG + LogReg/SVM), a CNN from scratch and transfer learning (ResNet-18, EfficientNet-B0).
* **Assignment 2:** the same task levelled up with a SOTA **text-to-video** generator. **CogVideoX-2B**, an open alternative to Step-Video-T2V, produces clips with camera motion, blur and new backgrounds. A **CLIP** quality gate filters the frames, robustness is measured with a 7-corruption **stress test**, and a **Gradio app** explains its predictions.

```
text prompt → CogVideoX-2B clip → sample frames → CLIP quality gate → + real photos → fine-tune CNN
                                                         evaluation: real photos only (clean + 7 corruptions)
```

## Results (A2, 5-fold cross-validation on 115 real photos)

| Model | Clean accuracy | Stress-test accuracy |
|---|---|---|
| A1 classical: HOG + SVM | 79.1% | 61.7% |
| A1 baseline: ResNet-18, real only | 93.9% | 78.9% |
| ResNet-18 + synthetic frames (CLIP-gated) | 91.3% | 77.0% |
| EfficientNet-B0, real only | 96.5% | 85.2% |
| **EfficientNet-B0 + synthetic frames (CLIP-gated)** | **97.4%** | **86.2%** |
| Zero-shot CLIP ViT-B/32 (no training) | 100% | 95.0% |

Speed at batch 1: ResNet-18 runs at 4.7 ms on a T4 (213 FPS) and 60 ms on CPU. EfficientNet-B0 runs at 8.6 ms on a T4 (117 FPS) and 36 ms on CPU. Generating one 6-second clip with CogVideoX-2B takes about 10 minutes on a T4. Full analysis, ablations and ethics are in [`report/`](report/).

## Repository

| Path | What it is |
|---|---|
| `notebooks/A1_household_object_recognition.ipynb` | A1: dataset cleaning, EDA, baselines, transfer learning |
| `notebooks/A2_household_objects_SOTA.ipynb` | A2: text-to-video generation, CLIP gate, cross-validated experiments, stress test, app bundle |
| `app.py` | Gradio app: upload/camera → prediction → Grad-CAM + nearest neighbours, latency, environment |
| `requirements.txt` | Packages for the app |
| `app_bundle.zip` (in **Releases**) | Trained model + gallery + watermarked synthetic clips, created by the A2 notebook (too large for the repo itself) |
| `report/` | A2 report (PDF); add the A1 report PDF here too |
| `HF_SPACE_README.md` | Header for a Hugging Face Space (rename to `README.md` there) |

**Dataset** (real photos, synthetic clips and frames):
<https://drive.google.com/drive/folders/1N-FRkymzGX3ZxU2KcoHOKBSCnagD7rEJ?usp=sharing>

## Run the app

Locally (CPU is enough, Python ≥ 3.10):

```bash
git clone https://github.com/<user>/<repo>.git && cd <repo>
pip install -r requirements.txt
# download app_bundle.zip from the Releases page of this repo and unzip it here -> ./app_bundle/model.pt, meta.json, ...
python app.py --bundle app_bundle          # then open http://127.0.0.1:7860
```

In Google Colab, with a temporary public link:

```python
!pip -q install gradio
!python app.py --bundle /content/drive/MyDrive/household_objects/a2/app_bundle --share
```

What the app shows:

* **Recognize:** class probabilities, a Grad-CAM heat-map and the 6 nearest training images (tagged *real* or *SYNTHETIC*). It also shows the latency of each request, the session average and FPS, the server device and your browser.
* **Stress test:** apply motion blur, defocus, low light, rotation, perspective, occlusion or JPEG compression, then predict again.
* **Synthetic data:** the generated clips with their prompts, each watermarked **SYNTHETIC**.
* **About & ethics.**

## Reproduce the experiments

1. Put the photos on Google Drive as `MyDrive/household_objects/raw/{bottle,cup,key,pen}/*.jpg`.
2. Open a notebook in Colab, choose *Runtime → Change runtime type → T4 GPU*, then *Run all* and allow Drive access.
3. In A2, the 16 clips take about 2.7 h. Each clip is saved to Drive as soon as it is finished; if the session stops, run all again and finished clips are skipped.
4. Outputs: `household_objects/a2/results/` (tables, figures, `a2_metrics.json`) and `household_objects/a2/app_bundle.zip`.

Seed 42 everywhere. The 5-fold split, every prompt and every per-clip seed are saved.
Weights (downloaded automatically): `THUDM/CogVideoX-2b` (Apache-2.0), `openai/clip-vit-base-patch32`, and torchvision ImageNet weights.

## Ethics

The real photos show only the authors' own objects: no people, faces or documents. All generated media are labelled and watermarked as synthetic, and they are used for training only, never for evaluation. Uploaded images are not stored. Some generated frames reproduce fake stock-footage logos, which is discussed in the report.
