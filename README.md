# Satellite image classification: Random Forest vs. a small CNN

Machine Learning course project (Computer Engineering, 3rd year, December 2025) by **Emir Varol** and **Berat Kerem Aydın**. It classifies 64×64 RGB satellite tiles into land-use / land-cover classes and compares a classical baseline (Random Forest on raw pixels) with a small convolutional network written in PyTorch.

## Results

Saved output of the notebook: one run, one stratified 80/20 split (`random_state=42`), accuracy on the 20 % test split.

| Model | Test accuracy |
|---|---|
| Random Forest (100 trees on flattened raw pixels) | **73.22 %** |
| CNN (3 convolution blocks with BatchNorm, PyTorch, 15 epochs) | **89.75 %** |

The notebook also draws both confusion matrices, the CNN training-loss curve, and has a helper (`test_tek_resim`) that classifies a single image with both models.

## Data

EuroSAT-style RGB tiles (Sentinel-2, 64×64 px). The authors used a folder provided for the course with **7 class folders**: Forest, HerbaceousVegetation, Highway, Industrial, Residential, River, SeaLake. The notebook reads every sub-folder of `EuroSAT_RGB/Data` as one class: **17,850 images**, split into **14,280 train / 3,570 test**.

The data is **not included** in this repository. Get EuroSAT from the official repository, <https://github.com/phelber/EuroSAT>. The original dataset has 10 classes, so with the full download the notebook runs but the numbers will differ from the table above (the exact figures need the same 7-class subset).

## Method

- **Preprocessing:** images converted to RGB, resized to 64×64, scaled to [0, 1].
- **Random Forest:** `RandomForestClassifier(n_estimators=100, random_state=42)` on the flattened 64×64×3 vector.
- **CNN (`SatelliteCNN`):** three blocks of `Conv2d(3→32→64→128, 3×3) → BatchNorm → ReLU → MaxPool`, then `Flatten → Linear(8192→512) → ReLU → Dropout(0.5) → Linear(512→classes)`. Adam (lr 0.001), batch size 64, 15 epochs, cross-entropy loss.

## Run it

```bash
pip install -r requirements.txt
# point the notebook at the folder that contains one sub-folder per class
export EUROSAT_DIR=/path/to/EuroSAT_RGB/Data       # Windows PowerShell: $env:EUROSAT_DIR="C:\path\to\EuroSAT_RGB\Data"
jupyter notebook satellite_image_classification.ipynb
```

The saved notebook kernel was Python 3.11.5; package versions were not recorded.

## Limitations

- One random split and one run: no cross-validation, no confidence intervals. The CNN is not seeded, so its accuracy varies a little between runs.
- No separate validation set, no data augmentation, no early stopping, no hyperparameter search.
- Random Forest on raw pixels is a deliberately simple baseline.
- The saved output of the last cell shows one example where the CNN labels a file taken from `Test/Residential/` as *Forest* with 100 % softmax confidence. The confidences are not calibrated probabilities.
- The 7-class dataset folder was provided in the course; its exact preparation is not documented here.

## Citation

Helber, Bischke, Dengel, Borth. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.

## Türkçe özet

Makine Öğrenmesi dersi projesi (Emir Varol ve Berat Kerem Aydın, Aralık 2025). EuroSAT tabanlı 7 sınıflı uydu görüntüsü veri kümesinde (17.850 görüntü, %80 eğitim / %20 test) **Random Forest (%73,22)** ile PyTorch'ta yazılmış küçük bir **CNN'i (%89,75)** karşılaştırır. Veri depoda yoktur; `EUROSAT_DIR` ortam değişkeniyle klasör yolu verilir. Tek bölme ve tek çalıştırma sonucudur; sınırlar yukarıda listelenmiştir.

## Note

No license is specified: this is a shared course project published for reference.
