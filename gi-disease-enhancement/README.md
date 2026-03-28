# GI Disease Enhancement: Impact of Image Enhancement on Deep Learning Classification

## Project Overview

This project investigates how image enhancement techniques affect the classification accuracy of gastrointestinal (GI) disease detection using deep learning. Endoscopic images often suffer from noise, blur, uneven illumination, and compression artifacts that degrade diagnostic performance. We systematically evaluate whether preprocessing with adaptive enhancement (CLAHE, denoising, sharpening) improves ResNet-50 classification under realistic degradation conditions.

### Objectives

1. **Baseline performance** -- Train and evaluate ResNet-50 on clean GI endoscopy images.
2. **Degradation impact** -- Quantify accuracy loss under synthetic noise, blur, contrast reduction, and JPEG compression at multiple severity levels.
3. **Enhancement recovery** -- Measure how much accuracy an adaptive enhancement pipeline recovers on degraded images.
4. **Ablation study** -- Isolate the contribution of each enhancement component (CLAHE, denoising, sharpening) and their combinations.

---

## Project Structure

```
gi-disease-enhancement/
├── src/
│   ├── enhancement/
│   │   ├── clahe.py            # CLAHE adaptive histogram equalization
│   │   ├── denoise.py          # Bilateral and Non-Local Means denoising
│   │   ├── sharpen.py          # Unsharp mask sharpening
│   │   └── pipeline.py         # ImageEnhancer: combined enhancement pipeline
│   ├── quality/
│   │   ├── assessment.py       # Blur detection, noise estimation (BRISQUE, NIQE)
│   │   └── degradation.py      # Synthetic noise, blur, contrast, JPEG artifacts
│   ├── classification/
│   │   ├── model.py            # ResNet-50 classifier definition
│   │   ├── train.py            # Training loop with early stopping
│   │   └── evaluate.py         # Evaluation and metrics computation
│   └── utils/
│       ├── data_loader.py      # PyTorch Dataset and DataLoader utilities
│       ├── metrics.py          # Accuracy, precision, recall, F1, confusion matrix
│       └── visualization.py    # Training curves, confusion matrices, sample grids
├── scripts/
│   ├── organize_dataset.py     # Split raw data into train/val/test (70/10/20)
│   └── create_degraded_dataset.py  # Generate degraded test images
├── experiments/                # Experiment scripts (exp1, exp2, exp3)
├── configs/                    # YAML configuration files
├── data/
│   ├── raw/kvasir-dataset-v2/  # Original dataset (8 classes)
│   ├── splits/                 # Train/val/test splits
│   └── degraded/               # Synthetically degraded images
├── results/                    # Figures, tables, model checkpoints
├── notebooks/                  # Exploratory Jupyter notebooks
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Dataset

**Kvasir v2** -- 8,000 labeled endoscopic images across 8 GI classes, curated by experienced gastroenterologists.

| Class | Train | Val | Test | Total |
|-------|------:|----:|-----:|------:|
| dyed-lifted-polyps | 699 | 100 | 201 | 1,000 |
| dyed-resection-margins | 699 | 100 | 201 | 1,000 |
| esophagitis | 699 | 100 | 201 | 1,000 |
| normal-cecum | 699 | 100 | 201 | 1,000 |
| normal-pylorus | 699 | 100 | 201 | 1,000 |
| normal-z-line | 699 | 100 | 201 | 1,000 |
| polyps | 699 | 100 | 201 | 1,000 |
| ulcerative-colitis | 699 | 100 | 201 | 1,000 |
| **Total** | **5,592** | **800** | **1,608** | **8,000** |

Split ratio: 70% train / 10% val / 20% test (seed=42, stratified per class).

### Download

1. Download from https://datasets.simula.no/kvasir/
2. Extract into `data/raw/`:
   ```bash
   unzip kvasir-dataset-v2.zip -d data/raw/
   ```
3. Run the split script:
   ```bash
   python scripts/organize_dataset.py
   ```

---

## Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU + drivers (optional; CPU mode works out of the box)

### Setup and Run

```bash
# Build the Docker image
docker compose build

# Start services (Jupyter on :8888, TensorBoard on :6006)
docker compose up -d

# Open a shell inside the container
docker compose exec dev bash
```

### Running Without GPU

The GPU block in `docker-compose.yml` is commented out by default. To enable GPU support, uncomment the `deploy:` section (lines 14-19) and ensure the NVIDIA Container Toolkit is installed.

---

## Degradation Pipeline

Synthetic degradations applied to test images to simulate real-world quality issues:

| Degradation | Low | Medium | High |
|-------------|-----|--------|------|
| Gaussian noise | sigma=10 | sigma=20 | sigma=30 |
| Gaussian blur | kernel=3 | kernel=5 | kernel=7 |
| Contrast reduction | gamma=0.5 | gamma=0.7 | -- |
| JPEG compression | quality=30 | quality=50 | -- |
| Combined (blur+noise+contrast) | -- | -- | severe |

Generate degraded images:
```bash
python scripts/create_degraded_dataset.py
```

Output: `data/degraded/{type}/test/{class}/`

---

## Enhancement Pipeline

The `ImageEnhancer` class in `src/enhancement/pipeline.py` chains three components:

1. **CLAHE** (`src/enhancement/clahe.py`) -- Contrast Limited Adaptive Histogram Equalization on the L channel (LAB color space). Improves local contrast without amplifying noise.
2. **Denoising** (`src/enhancement/denoise.py`) -- Bilateral filtering or Non-Local Means to reduce noise while preserving edges.
3. **Sharpening** (`src/enhancement/sharpen.py`) -- Unsharp masking to restore edge detail lost during denoising or blur.

---

## Experiments

### Experiment 1: Baseline Classification

Train ResNet-50 on clean images and evaluate on the clean test set.

```bash
python experiments/exp1_baseline.py \
    --data-dir data/splits \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --output-dir results/exp1_baseline
```

### Experiment 2: Enhancement Comparison

Compare classification on clean vs. degraded vs. enhanced test sets.

```bash
python experiments/exp2_enhancement.py \
    --data-dir data/splits \
    --degradation-levels low medium high \
    --output-dir results/exp2_enhancement
```

### Experiment 3: Ablation Study

Test individual enhancement components and their combinations.

```bash
python experiments/exp3_ablation.py \
    --data-dir data/splits \
    --degradation-level medium \
    --output-dir results/exp3_ablation
```

---

## Results

Each experiment produces:

| Output | Location | Description |
|--------|----------|-------------|
| Model checkpoints | `results/*/models/` | `best_model.pth` and `final_model.pth` |
| Training curves | `results/*/figures/training_curves.png` | Loss and accuracy over epochs |
| Confusion matrices | `results/*/figures/confusion_matrix_*.png` | Per-condition matrices |
| Metrics | `results/*/tables/` | Per-class precision, recall, F1 (JSON) |
| TensorBoard logs | `results/*/tensorboard/` | Interactive plots at `http://localhost:6006` |

### Key Metrics

- **Accuracy drop**: clean baseline vs. degraded (quantifies degradation impact)
- **Recovery rate**: `(enhanced_acc - degraded_acc) / (clean_acc - degraded_acc)`
- **Per-class F1**: identifies which disease classes benefit most from enhancement
- **Ablation ranking**: which component contributes most to recovery

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, torchvision, TensorBoard |
| Image Processing | OpenCV, scikit-image, Pillow, albumentations |
| Quality Assessment | pyiqa (BRISQUE, NIQE) |
| ML / Data Science | scikit-learn, NumPy, pandas, SciPy |
| Visualization | matplotlib, seaborn, plotly |
| Infrastructure | Docker, Docker Compose |

---

## Team Roles

| Member | Role | Responsibilities |
|--------|------|-----------------|
| **Member 1** | Project Lead / Enhancement | Enhancement pipeline, adaptive parameter tuning, quality assessment |
| **Member 2** | Classification / Training | ResNet-50 architecture, training loop, hyperparameter tuning |
| **Member 3** | Data / Evaluation | Dataset preparation, degradation pipeline, evaluation metrics |
| **Member 4** | Experiments / Visualization | Experiment scripts, ablation study, result visualization, documentation |

---

## Citation

If you use the Kvasir dataset, please cite:

```bibtex
@inproceedings{pogorelov2017kvasir,
  title     = {KVASIR: A Multi-Class Image Dataset for Computer Aided
               Gastrointestinal Disease Detection},
  author    = {Pogorelov, Konstantin and Randel, Kristin Ranheim and
               Grber, Christine and Eskeland, Sigrun Losada and
               Peez, Thomas de Lange and Dang-Nguyen, Duc-Tien and
               Lux, Mathias and Spampinato, Concetto and
               Lange, Thomas and Halvorsen, Pal and Riegler, Michael},
  booktitle = {Proceedings of the 8th ACM on Multimedia Systems Conference},
  pages     = {164--169},
  year      = {2017},
  doi       = {10.1145/3083187.3083212}
}
```

---

## License

This project is for academic research purposes. The Kvasir dataset is released under a Creative Commons Attribution 4.0 International License. See the [dataset page](https://datasets.simula.no/kvasir/) for details.