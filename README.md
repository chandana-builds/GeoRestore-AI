# 🌍 GeoRestore AI: Satellite Imagery Cloud & Haze Removal

GeoRestore AI is a deep learning and physics-guided computer vision framework designed to detect, penetrate, and reconstruct cloudy or hazy optical satellite imagery. It restores clear surface textures while preserving non-cloud terrain, water bodies, and vegetation.

---

## 🚀 Key Features

- **Residual U-Net Architecture**: Custom deep convolutional network with skip connections tailored for high-resolution satellite imagery restoration.
- **Adaptive Resolution Handling**: Handles arbitrary image dimensions via dynamic padding and unpadding.
- **Physics-Guided Haze Removal**: Leverages Dark Channel Prior (DCP) and atmospheric scattering models for thin cloud and haze penetration.
- **Contextual Terrain Inpainting**: Texture-synthesis and boundary-blending inpainting for opaque, thick cloud occlusions.
- **Hybrid AI Engine**: Combines deep semantic predictions with physics-based and texture reconstruction for artifacts-free outputs.
- **Streamlit Interactive UI**: Real-time web application to upload satellite images, adjust cloud detection thresholds, and preview side-by-side restorations.
- **Comprehensive Evaluation Suite**: Built-in scripts for PSNR, SSIM, loss visualization, and comparative analysis.

---

## 📁 Project Structure

```text
GeoRestore-AI/
├── app/                      # Streamlit interactive web application
│   ├── app.py                # Web dashboard and UI controls
│   └── utils.py              # Image processing and inference helpers
├── data/                     # Dataset storage (raw, processed)
├── notebooks/                # Jupyter research and validation notebooks
│   ├── 01_preprocessing.ipynb
│   ├── 02_dataset.ipynb
│   ├── 03_unet.ipynb
│   ├── ...
├── outputs/                  # Evaluation figures, logs, and predictions
│   ├── comparisons/
│   ├── predictions/
│   └── training_loss.png
├── src/                      # Core framework package
│   ├── dataset/              # PyTorch Dataset and DataLoaders
│   ├── evaluation/           # Quantitative metrics (PSNR, SSIM)
│   ├── inference/            # Prediction and cloud removal pipeline
│   ├── models/               # U-Net blocks and model definitions
│   ├── preprocessing/        # Cloud mask estimation and transforms
│   ├── training/             # Loss functions, optimizers, train/val loops
│   └── utils/                # Checkpoint loading and saving utilities
├── compare.py                # Visual comparison tool
├── evaluate.py               # Dataset-wide evaluation script
├── plot_loss.py              # Loss curve plotter
├── predict.py                # CLI inference tool
├── requirements.txt          # Python dependencies
├── test_setup.py             # Architecture verification script
└── train.py                  # Model training pipeline
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/chandana-builds/GeoRestore-AI.git
   cd GeoRestore-AI
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python test_setup.py
   ```

---

## 💻 Usage

### 1. Launch the Streamlit Web Application
```bash
streamlit run app/app.py
```
Upload any satellite image (PNG/JPG), choose between **Hybrid AI + Inpainting**, **Contextual Inpainting**, or **Deep Residual U-Net**, and download restored results directly from the browser.

### 2. Run CLI Inference
```bash
python predict.py --input path/to/cloudy_image.png --output outputs/prediction.png --mode hybrid
```

### 3. Model Training
```bash
python train.py --epochs 50 --batch-size 8 --lr 1e-4
```

### 4. Quantitative Evaluation
```bash
python evaluate.py --model outputs/checkpoints/best_unet.pth
```

### 5. Plot Loss Curves
```bash
python plot_loss.py
```

---

## 📊 Evaluation & Metrics

The framework evaluates restorations against ground truth using:
- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index Measure)**

---

## 📄 License

This project is licensed under the MIT License.
