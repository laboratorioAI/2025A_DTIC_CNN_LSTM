# 2025A_DTIC_CNN_LSTM
# Multi-Month Evaluation for EMG-Based Hand Gesture Recognition

This repository contains two evaluation scripts designed to assess the performance of deep learning models (CNN and CNN-LSTM) for hand gesture recognition (HGR) using electromyographic (EMG) signals over multiple months of data.

---

## 📁 Folder Structure

Expected dataset organization:

```
EMG-EPN-612 dataset/
├── Mes0/
│   └── trainingJSON/
│       └── userX.json
├── Mes1/
│   └── trainingJSON/
│       └── userY.json
...
```

---

## 🧠 Scripts Overview

### 1. `multiFolderEvaluation.m` – CNN Model Evaluation

This script evaluates a trained CNN model on all `Mes*/trainingJSON` folders. For each month and each user, it computes:

* **Classification accuracy**
* **Recognition accuracy**
* **Overlapping factor**
* **Average processing time per frame**

It generates:

* Per-user performance evolution plots (average of training and validation).
* Global plots across all users.
* Summary tables exported to Excel.

#### ✅ Requirements

* `Shared.m` – constants and preprocessing functions
* `evalRecognition.m` – metric evaluation
* Trained CNN model (`.mat` file), e.g., `Models/model_cnn_final.mat`

---

### 2. `multiFolderEvaluation_LSTM.m` – CNN-LSTM Model Evaluation

This script evaluates a trained CNN-LSTM model following the same logic as the CNN version. It performs:

* Per-month evaluation (train/validation)
* Per-user performance plots
* Global performance plots
* Excel export of all aggregated metrics

#### ✅ Requirements

* `Shared.m`
* `evalRecognition.m`
* Trained CNN-LSTM model (`.mat` file), e.g., `ModelsLSTM/model_cnn-lstm_final.mat`

---

## 📊 Metrics Computed

For each user and each month:

* **Classification Accuracy**: correct gesture type prediction
* **Recognition Accuracy**: correct gesture recognition (including timing)
* **Overlapping Factor**: alignment accuracy between prediction and ground truth
* **Processing Time**: average time per spectrogram prediction

---

## 📤 Outputs

* **Console**: average + standard deviation metrics per month
* **Figures**:

  * Per-user evolution of metrics across months
  * Global evolution of averages across users
* **Excel file**:

  * `Resumen-Metricas-AVG-CNN.xlsx` (for CNN)
  * `Resultados_CNN-LSTM.xlsx` (for CNN-LSTM)

Each file contains four sheets:

* `Classification`
* `Recognition`
* `Overlap`
* `Processing_Time`

---

## ⚙️ How to Run

1. Ensure all required files (`Shared.m`, `evalRecognition.m`, model `.mat`) are on your MATLAB path.
2. Set the `baseDataDir` and `modelFile` variables if needed.
3. Run the corresponding script:

   ```matlab
   multiFolderEvaluation()        % For CNN model
   multiFolderEvaluation_LSTM()   % For CNN-LSTM model
   ```

---

## 👨‍💼 Authors

Developed as part of a research project at the **Alan Turing Artificial Intelligence Laboratory – Escuela Politécnica Nacional**.

---

## 📄 License

This project is for academic and research purposes only. Contact the authors for use in other contexts.
