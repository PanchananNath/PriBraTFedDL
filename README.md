# Federated deep bearning framework for brain tumor detection

The codebase consist of **five custom deep learning models** and trains them in a **federated manner** across multiple clients.

`brain_tumor_federated_learning.py` is a Python implementation developed by **Er. Panchanan Nath** for simulating **Federated Learning (FL)** on a **Dataset consisting of MRI image of human brain tumor Classification **.  



---

## ✨ Features

- Implements **5 custom CNN architectures**:
  - `SimpleCNN`
  - `VGGLike`
  - `ResNetLike`
  - `SimpleDenseNetLike`
  - `MobileNetLite`
- Simulation of **Federated Learning** with:
  - Client data partitioning
  - Local client training
  - **Federated Averaging (FedAvg)** for model aggregation
  - Multiple communication rounds between server and clients
- Supports:
  - **Central validation** at the server
  - **Testing on held-out dataset**
  - Saving training history, confusion matrices, classification reports, and predictions
  - Visualization of results

---

## 📂 Dataset Structure

Ensure your dataset is structured as follows:

```
DATA_DIR/
│
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

- `Training/` → Used for federated client training & server validation split  
- `Testing/` → Used for final model evaluation  

---

## ⚙️ Hyperparameters

| Parameter                | Value (default) |
|---------------------------|-----------------|
| `NUM_CLASSES`             | 4 (glioma, meningioma, pituitary, notumor) |
| `NUM_COMMUNICATION_ROUNDS`| 100 |
| `CLIENT_LOCAL_EPOCHS`     | 10 |
| `NUM_CLIENTS`             | 10 |
| `CLIENT_FRACTION`         | 0.5 |
| `BATCH_SIZE`              | 32 |
| `IMAGE_SIZE`              | 224x224 |
| `LEARNING_RATE`           | 1e-3 |
| `WEIGHT_DECAY`            | 1e-4 |

---

## 🚀 Usage

1. Clone this repository and place your dataset under the required structure.
2. Update the `DATA_DIR` in the script to point to your dataset root.
   ```python
   DATA_DIR = "./BT"   # Replace with your dataset folder containing Training/ and Testing/
   ```
3. Run the script (preferably on GPU/Colab):
   ```bash
   python brain_tumor_federated_learning.py
   ```

---

## 📊 Outputs

All results are stored in `./results_brain_tumor_federated/` including:

- **Best & final global model weights** (`.pt`)
- **Training history** plots & CSV
- **Classification reports** (`.txt`)
- **Confusion matrices** (`.png`)
- **Prediction results** (`.csv`)
- **Sample prediction visualizations**

---

## 📈 Example Results

- Training & validation loss/accuracy curves per communication round
- Confusion matrices (raw & normalized)
- Per-class precision, recall, and F1-score
- Best federated model selected by validation accuracy

---

## 📌 Requirements

- Python 3.8+
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- tqdm

Install dependencies:
```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn tqdm
```


---

## 📜 License

This experiment is released under the **MIT License**. Feel free to use and modify for research purposes with attribution.

---
