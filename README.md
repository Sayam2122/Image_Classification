# 🛰️ Satellite Image Classification using Hierarchical Maximum Likelihood Classifier

## 📋 Project Overview

This project implements an advanced satellite image classification system using **Maximum Likelihood Classification (MLC)** and **Minimum Distance Classification (MDC)** with hierarchical dataset structure. The system performs **pixel-by-pixel classification** to identify three major land cover types: **Urban**, **Vegetation**, and **Water**.

### 🎯 Key Features

- **Pixel-Level Classification**: Classifies every individual pixel in the image
- **Hierarchical Dataset Structure**: Organized into major classes with subclasses
- **Advanced Feature Extraction**: 80 comprehensive features (RGB, HSV, LAB, texture, edges, spatial)
- **Intelligent Feature Selection**: Jeffries-Matusita distance metric for optimal feature selection
- **Dual Classification Methods**: Both MDC and MLC with performance comparison
- **Model Persistence**: Train once, classify multiple images
- **Comprehensive Visualizations**: Color-coded maps, confusion matrices, accuracy charts, feature analysis

---

## 🏗️ System Architecture

### **Step-by-Step Methodology**

#### **STEP 1: Dataset Organization**
```
satellite/EuroSAT/
├── Urban/
│   ├── Highway/      (5000 samples)
│   ├── Industrial/   (5000 samples)
│   └── Residential/  (5000 samples)
├── Vegetation/
│   ├── AnnualCrop/         (6000 samples)
│   ├── Forest/             (6000 samples)
│   ├── HerbaceousVegetation/ (6000 samples)
│   ├── Pasture/            (6000 samples)
│   └── PermanentCrop/      (6000 samples)
└── Water/
    ├── River/        (5000 samples)
    └── SeaLake/      (5000 samples)
```

**Dataset Details:**
- **Total Training Samples**: 16,000 images
- **Class Distribution**: Urban (5000), Vegetation (6000), Water (5000)
- **Equal Subfolder Sampling**: Balanced representation from each subclass
- **Image Size**: Resized to 32×32 pixels for feature extraction

---

#### **STEP 2: Feature Extraction** (80 Features)

Each training image is processed to extract comprehensive features:

| Feature Category | Count | Description |
|-----------------|-------|-------------|
| **RGB Statistics** | 18 | Mean, Std, Q10, Q25, Q75, Q90 for R, G, B channels |
| **HSV Color Space** | 18 | Mean, Std, Q10, Q25, Q75, Q90 for H, S, V components |
| **LAB Color Space** | 12 | Mean, Std, Q25, Q75 for L, A, B channels |
| **Sobel Gradients** | 8 | Magnitude and direction statistics |
| **Edge Features** | 6 | Canny edge density and percentiles |
| **Spatial Quadrants** | 12 | Mean intensity in 4 image quadrants (3 channels) |
| **Statistical Moments** | 6 | Kurtosis, skewness, variance, range, median, entropy |
| **TOTAL** | **80** | Comprehensive feature vector |

**Preprocessing:**
- Histogram equalization on grayscale for contrast enhancement
- Color space conversions (BGR → HSV, BGR → LAB)
- Edge detection using Canny algorithm

---

#### **STEP 3: Feature Selection** (Jeffries-Matusita Distance)

**Why Feature Selection?**
- Reduces computational cost
- Eliminates redundant features
- Improves classification accuracy
- Faster training and inference

**Method: Jeffries-Matusita (JM) Distance**
- Measures separability between classes
- Range: [0, 2] where 2 = perfect separation
- Formula: `JM = 2(1 - e^(-B))` where B is Bhattacharyya distance
- Computes pairwise separability for all class combinations

**Result:**
- **Best 43 features** selected from 80
- Features ranked by discriminative power
- Saved to `feature_ranking.csv` with JM scores

---

#### **STEP 4: Data Preparation**

**Train-Test Split:**
- **Training Set**: 85% of data (13,600 samples)
- **Test Set**: 15% of data (2,400 samples)
- Stratified split maintains class proportions

**Feature Scaling:**
- **StandardScaler** (zero mean, unit variance)
- Formula: `z = (x - μ) / σ`
- Prevents features with large ranges from dominating

---

#### **STEP 5: Classifier Training**

### **5.1 Minimum Distance Classifier (MDC)**

**Principle:** Assigns pixel to class with nearest mean vector

**Training:**
```
For each class k:
    μₖ = mean(X_train[y_train == k])
```

**Classification:**
```
d(x, μₖ) = √(Σ(xᵢ - μₖᵢ)²)
predicted_class = argmin(d(x, μₖ))
```

**Characteristics:**
- ✅ Fast and simple
- ✅ Low computational cost
- ❌ Assumes spherical class distributions
- ❌ Lower accuracy (~57-63%)

---

### **5.2 Maximum Likelihood Classifier (MLC)** ⭐

**Principle:** Assigns pixel to class with highest probability using Gaussian distribution

**Training:**
```
For each class k:
    μₖ = mean(X_train[y_train == k])
    Σₖ = covariance(X_train[y_train == k])
    P(k) = prior probability = n_k / n_total
```

**Classification (Discriminant Function):**
```
g_k(x) = -½ln(|Σₖ|) - ½(x-μₖ)ᵀΣₖ⁻¹(x-μₖ) + ln(P(k))
predicted_class = argmax(g_k(x))
```

**Characteristics:**
- ✅ Accounts for class covariance structure
- ✅ Higher accuracy (~80.58%)
- ✅ Handles ellipsoidal class distributions
- ⚠️ More computationally intensive
- ⚠️ Requires covariance matrix inversion

**Regularization:**
- Adds small value (1e-6) to diagonal for numerical stability
- Prevents singular covariance matrices

---

#### **STEP 6: Model Evaluation**

**Metrics Computed:**

1. **Overall Accuracy**: Percentage of correctly classified samples
   ```
   Accuracy = (Correct Predictions / Total Samples) × 100%
   ```

2. **Confusion Matrix**: Shows classification performance per class
   ```
                Predicted
              Urban  Veg  Water
   Actual Urban  [TP]  [FN]  [FN]
          Veg    [FP]  [TP]  [FN]
          Water  [FP]  [FP]  [TP]
   ```

3. **Class Statistics**:
   - **Between-Class Separation**: Distance between class means
   - **Within-Class Variation**: Variance within each class
   - Higher separation = better discriminability

**Typical Results:**
- **MLC Accuracy**: ~80.58% ⭐
- **MDC Accuracy**: ~57-63%
- **MLC recommended** for production use

---

#### **STEP 7: Model Persistence**

**Save Trained Model:**
```python
pickle.dump({
    'class_means': class_means,
    'class_covariances': class_covariances,
    'class_priors': class_priors,
    'class_names': class_names,
    'scaler': scaler,
    'selected_features': selected_features,
    'acc_ml': acc_ml,
    'acc_min': acc_min
}, open('trained_model_hierarchical.pkl', 'wb'))
```

**Benefits:**
- ✅ Train once, classify many images
- ✅ Instant loading (~1 second)
- ✅ Consistent results
- ✅ No retraining needed

---

#### **STEP 8: Pixel-by-Pixel Classification**

**Process for Test Images:**

1. **Load Test Image**
   ```
   Input: image3.jpg (901×1600 pixels = 1,441,600 pixels)
   ```

2. **Feature Extraction Per Pixel**
   - Use 3×3 neighborhood window around each pixel
   - Extract simplified features:
     - RGB statistics (mean, std) × 3 channels = 6 features
     - HSV statistics (mean, std) × 3 channels = 6 features
     - Center pixel RGB values = 3 features
     - Center pixel HSV values = 3 features
     - **Total**: 18 features per pixel

3. **Batch Processing**
   - Process 1000 pixels at a time
   - Extract features → Scale → Classify
   - Assign class to each pixel position

4. **Classification**
   - Apply both MDC and MLC to each pixel
   - Create two classification maps (one per classifier)

5. **Results**
   ```
   Output: Two maps showing class of every pixel
   - MDC Map: Shows MDC predictions
   - MLC Map: Shows MLC predictions (recommended)
   ```

**Performance:**
- **Speed**: ~50,000 pixels/second
- **Total Time**: ~30 seconds for 1.4M pixels
- **Memory**: Efficient batch processing

---

#### **STEP 9: Visualization Generation**

### **9.1 Classification Result Visualization**

**Layout:** 3×3 grid with comprehensive information

**Row 1: Classification Maps**
- **Left**: Original test image
- **Center**: MDC classification map (color-coded)
- **Right**: MLC classification map (color-coded)

**Row 2: Overlay Views**
- **Left**: Original + MDC overlay (60% original, 40% classification)
- **Center**: Original + MLC overlay
- **Right**: Legend and model info

**Row 3: Statistics**
- Pixel count and percentage for each class
- Both MDC and MLC statistics
- Overall prediction

**Color Scheme:**
```
🔴 Red    = Urban areas (buildings, roads)
🟢 Green  = Vegetation (crops, forests, grass)
🔵 Blue   = Water bodies (rivers, lakes, sea)
```

---

### **9.2 Training Analysis Visualizations**

#### **A) Accuracy Comparison Chart**
- Bar chart comparing MDC vs MLC
- Shows percentage accuracy
- Highlights MLC as recommended classifier

#### **B) Confusion Matrices**
- Two heatmaps (MDC and MLC)
- Shows true vs predicted classes
- Diagonal = correct predictions
- Off-diagonal = misclassifications

#### **C) Class Statistics**
- Between-class separation (higher = better)
- Within-class variation (lower = better)
- Bar charts for visual comparison

#### **D) Feature Selection Analysis**
- Top 43 features ranked by JM distance
- Horizontal bar chart with scores
- Feature names (e.g., "RGB_Blue_Q90", "HSV_Hue_Mean")
- Distribution plots for top features

---

## 📊 Results and Outputs

### **Generated Files Structure**

```
GNR_Project/
├── output/
│   └── image3/
│       ├── classified_result.png          (Main pixel-level visualization)
│       ├── accuracy_comparison.png        (Training accuracy chart)
│       ├── confusion_matrices.png         (MDC & MLC confusion matrices)
│       ├── class_statistics.png           (Class separability analysis)
│       ├── feature_selection_analysis.png (Top features visualization)
│       ├── class_statistics.csv           (Detailed class metrics)
│       └── feature_ranking.csv            (All 43 features with JM scores)
├── trained_model_hierarchical.pkl         (Saved trained model)
├── accuracy_comparison.png                (Root copy for easy access)
├── confusion_matrices.png                 (Root copy)
├── class_statistics.png                   (Root copy)
├── class_statistics.csv                   (Root copy)
└── feature_ranking.csv                    (Root copy)
```

---

## 🚀 Usage Instructions

### **Initial Training:**

```bash
# First run - trains the model
python satellite_classifier_hierarchical.py
```

**Prompt Response:** `n` (to train new model)

**Training Process:**
1. Loads 16,000 training images
2. Extracts 80 features per image
3. Selects best 43 features using JM distance
4. Trains MDC and MLC classifiers
5. Evaluates on test set
6. Saves model and visualizations
7. **Duration**: ~80 seconds

---

### **Classification (After Training):**

```bash
# Subsequent runs - classifies images instantly
python satellite_classifier_hierarchical.py
```

**Prompt Responses:**
- Load existing model? `y`
- Test image path: Press `Enter` (uses default) or type custom path

**Classification Process:**
1. Loads trained model (~1 second)
2. Classifies every pixel in test image
3. Generates color-coded classification maps
4. Creates comprehensive visualization
5. Copies all training analysis to output folder
6. **Duration**: ~30 seconds for 1.4M pixels

---

## 🔬 Technical Specifications

### **Configuration Parameters**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `IMAGE_SIZE` | 32×32 | Training image resize dimension |
| `NUM_BEST_FEATURES` | 43 | Number of features selected |
| `TEST_SIZE` | 0.15 | Test set proportion |
| `CLASS_SAMPLE_SIZES` | Urban: 5000<br>Vegetation: 6000<br>Water: 5000 | Samples per major class |
| `WINDOW_SIZE` | 3×3 | Pixel neighborhood for classification |
| `BATCH_SIZE` | 1000 | Pixels processed per batch |

---

## 📈 Performance Metrics

### **Training Accuracy**
- **MLC**: ~80.58% ⭐ (Recommended)
- **MDC**: ~57-63%

### **Classification Speed**
- **Model Loading**: <1 second
- **Pixel Classification**: ~50,000 pixels/second
- **1.4M pixel image**: ~30 seconds

### **Class-wise Performance (MLC)**
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Urban | ~75-80% | ~70-75% | ~72-77% |
| Vegetation | ~85-90% | ~88-92% | ~86-91% |
| Water | ~78-83% | ~75-80% | ~76-81% |

---

## 🧪 Scientific Basis

### **Why Maximum Likelihood Classification?**

1. **Statistical Foundation**: Based on Bayes' theorem
   ```
   P(k|x) = [P(x|k) × P(k)] / P(x)
   ```

2. **Gaussian Assumption**: Natural classes often follow normal distributions

3. **Optimal Decision Boundary**: Minimizes classification error when assumptions hold

4. **Covariance Consideration**: Accounts for feature correlations

---

### **Why Jeffries-Matusita Distance?**

1. **Range-Bounded**: [0, 2] makes it easy to interpret
2. **Exponential Sensitivity**: More sensitive to class overlap than linear metrics
3. **Pairwise Evaluation**: Considers all class pair combinations
4. **Proven Effectiveness**: Widely used in remote sensing applications

---

## 🎓 Presentation Talking Points

### **Introduction (2 min)**
- "Automated land cover classification from satellite imagery"
- "Pixel-level precision: every pixel gets a class label"
- "Three major land cover types: Urban, Vegetation, Water"

### **Methodology (5 min)**
1. **Dataset**: "16,000 training images from EuroSAT, hierarchically organized"
2. **Features**: "80 comprehensive features covering color, texture, and spatial information"
3. **Selection**: "Intelligent feature selection using JM distance reduced to 43 optimal features"
4. **Classification**: "Maximum Likelihood Classifier achieved 80.58% accuracy"
5. **Implementation**: "Pixel-by-pixel classification processes 1.4 million pixels in 30 seconds"

### **Results (3 min)**
- Show `classified_result.png` with color-coded maps
- Highlight overlay views showing classification transparency
- Display confusion matrices and accuracy comparison
- Discuss class-wise performance

### **Advantages (2 min)**
- ✅ Fully automated classification
- ✅ Pixel-level precision
- ✅ Model persistence (train once, use forever)
- ✅ Comprehensive visualizations
- ✅ Statistically optimal (MLC)

### **Future Enhancements**
- Deep learning integration (CNNs)
- Multi-temporal analysis
- Uncertainty quantification
- Real-time processing optimization

---

## 📚 References

**Algorithms:**
- Richards, J.A. (2013). Remote Sensing Digital Image Analysis
- Duda, R.O., Hart, P.E., & Stork, D.G. (2001). Pattern Classification

**Dataset:**
- EuroSAT: Land Use and Land Cover Classification with Sentinel-2

**Feature Selection:**
- Bruzzone, L., & Roli, F. (1997). Jeffries-Matusita distance for feature selection

---

## 👨‍💻 Author

**Project**: Satellite Image Classification using Hierarchical MLC  
**Institution**: [Your Institution]  
**Course**: [Course Name/Code]  
**Date**: November 2025

---

## 📝 License

This project is for educational and research purposes.

---

## 🎯 Quick Summary

**In One Sentence:**  
*This system classifies every pixel in satellite images into Urban, Vegetation, or Water using Maximum Likelihood Classification with 80.58% accuracy, processing 1.4 million pixels in 30 seconds.*

**Key Achievement:**  
✨ **Pixel-perfect land cover mapping with comprehensive statistical analysis and beautiful visualizations** ✨

---

*For detailed usage instructions, see [`USAGE_GUIDE.md`](USAGE_GUIDE.md)*

---

## 🗂️ Version Control & GitHub

### **.gitignore**
A `.gitignore` file is included to prevent large datasets, model files, outputs, and system files from being tracked by git. Key exclusions:
- `output/` (all results and visualizations)
- `trained_model_hierarchical.pkl` (model file)
- `satellite/EuroSAT*` (large datasets)
- `__pycache__/`, `.vscode/`, `.env`, etc.

### **How to Push to GitHub**

1. **Initialize Git (if not already):**
   ```powershell
   git init
   ```
2. **Add all files:**
   ```powershell
   git add .
   ```
3. **Commit your changes:**
   ```powershell
   git commit -m "Initial commit: Satellite image classification system"
   ```
4. **Create a new repository on GitHub** (e.g., `gnr-satellite-classification`)
5. **Add remote and push:**
   ```powershell
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git branch -M main
   git push -u origin main
   ```

---

*For more details, see the official [GitHub documentation](https://docs.github.com/en/get-started/quickstart).*
