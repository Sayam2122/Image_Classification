import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import scipy
import scipy.stats
from scipy.stats import multivariate_normal, entropy
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import time
from pathlib import Path
import pickle

"""
🛰️ SATELLITE IMAGE CLASSIFICATION SYSTEM - HIERARCHICAL
========================================================

WORKFLOW:
1. FIRST RUN - Training:
   - Loads training data from hierarchical folders (Urban/Vegetation/Water)
   - Extracts 50 fast features from each image
   - Selects top 20 features using Jeffries-Matusita distance
   - Trains classifiers (Minimum Distance & Maximum Likelihood)
   - Saves trained model to 'trained_model_hierarchical.pkl'
   
2. SUBSEQUENT RUNS - Classification:
   - Detects existing trained model
   - Loads model instantly (no retraining!)
   - Prompts for test image path
   - Classifies image using both methods
   - Fast classification in seconds!

FEATURES:
✓ Hierarchical dataset support (equal samples from subfolders)
✓ Fast feature extraction (~50 features)
✓ Jeffries-Matusita feature selection (best for satellite imagery)
✓ Model persistence (train once, use forever!)
✓ Comprehensive visualizations
✓ Class statistics and confusion matrices
"""

print("🛰️ SATELLITE IMAGE CLASSIFICATION SYSTEM - HIERARCHICAL")
print("=" * 60)

# ============================================================
# CONFIGURATION
# ============================================================
SATELLITE_DATASET_PATH = "satellite/EuroSAT"
TEST_IMAGE_PATH = "image3.jpg"
OUTPUT_FOLDER = "output/image3"
TRAINING_OUTPUT_FOLDER = "output/training_visualizations"  # For confusion matrix, feature selection, etc.
MODEL_SAVE_PATH = "trained_model_hierarchical.pkl"
# Class-specific sample sizes: 16K total for better accuracy
CLASS_SAMPLE_SIZES = {
    'Urban': 5000,     
    'Vegetation': 6000,  
    'Water': 5000       
}
NUM_BEST_FEATURES = 43 
IMAGE_SIZE = 32 
TEST_SIZE = 0.15  

# ============================================================
# SAVE/LOAD MODEL FUNCTIONS
# ============================================================
def save_model(model_data, filepath=MODEL_SAVE_PATH):
    """Save trained model to disk"""
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n💾 Model saved to: {filepath}")

def load_model(filepath=MODEL_SAVE_PATH):
    """Load trained model from disk"""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    print(f"\n📂 Model loaded from: {filepath}")
    return model_data

# ============================================================
# ENHANCED FEATURE EXTRACTION (~70+ features for better accuracy!)
# ============================================================
def extract_comprehensive_features(image_path):
    """
    Extract ~70+ discriminative features (optimized for accuracy):
    - RGB color statistics (16 features - enhanced)
    - HSV color space (16 features - enhanced)
    - LAB color space (12 features - NEW for better color separation!)
    - Texture - Sobel gradients (8 features - enhanced)
    - Edge features (6 features - enhanced)
    - Spatial statistics (12 features)
    - Statistical moments (6 features - enhanced)
    
    Total: ~76 features (better discrimination + still fast!)
    """
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Resize to larger size for MORE DETAIL (accuracy boost!)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)
    
    features = []
    
    # 1. RGB COLOR STATISTICS (16 features - ENHANCED!)
    for i, color in enumerate(['B', 'G', 'R']):
        channel = img[:,:,i]
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.percentile(channel, 10),  # NEW: Lower percentile
            np.percentile(channel, 25),
            np.percentile(channel, 75),
            np.percentile(channel, 90),  # NEW: Upper percentile
        ])
    
    # 2. HSV COLOR SPACE (16 features - ENHANCED!)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for i in range(3):
        channel = hsv[:,:,i]
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.percentile(channel, 10),  # NEW
            np.percentile(channel, 25),
            np.percentile(channel, 75),
            np.percentile(channel, 90),  # NEW
        ])
    
    # 3. LAB COLOR SPACE (12 features - NEW for better color discrimination!)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    for i in range(3):
        channel = lab[:,:,i]
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.percentile(channel, 25),
            np.percentile(channel, 75),
        ])
    
    # 4. TEXTURE FEATURES - SOBEL GRADIENTS (8 features - ENHANCED!)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_direction = np.arctan2(sobely, sobelx)
    
    features.extend([
        np.mean(gradient_magnitude),
        np.std(gradient_magnitude),
        np.percentile(gradient_magnitude, 75),
        np.percentile(gradient_magnitude, 25),
        np.mean(np.abs(sobelx)),
        np.mean(np.abs(sobely)),
        np.mean(gradient_direction),  # NEW: Texture direction
        np.std(gradient_direction),   # NEW: Direction variance
    ])
    
    # 5. EDGE FEATURES (6 features - ENHANCED!)
    edges = cv2.Canny(gray, 50, 150)
    features.extend([
        np.sum(edges > 0) / edges.size,  # Edge density
        np.mean(edges),
        np.std(edges),
        np.percentile(edges, 75),  # NEW
        np.percentile(edges, 90),
        np.percentile(edges, 95),  # NEW: Very strong edges
    ])
    
    # 6. SPATIAL STATISTICS - QUADRANTS (12 features)
    h, w = gray.shape
    quadrants = [
        gray[0:h//2, 0:w//2],
        gray[0:h//2, w//2:w],
        gray[h//2:h, 0:w//2],
        gray[h//2:h, w//2:w],
    ]
    
    for quad in quadrants:
        features.extend([
            np.mean(quad),
            np.std(quad),
            np.percentile(quad, 75),
        ])
    
    # 7. STATISTICAL MOMENTS (6 features - ENHANCED!)
    features.extend([
        scipy.stats.kurtosis(gray.flatten()),
        scipy.stats.skew(gray.flatten()),
        np.var(gray),
        np.ptp(gray),  # Peak-to-peak (range)
        np.median(gray.flatten()),  # NEW: Median
        scipy.stats.entropy(np.histogram(gray, bins=32)[0] + 1e-10),  # NEW: Entropy
    ])
    
    return np.array(features)

# Total features: 18 + 18 + 12 + 8 + 6 + 12 + 6 = 80 features (ACCURACY BOOST!)

def get_feature_names():
    """
    Generate descriptive names for all 80 features (ENHANCED!)
    """
    feature_names = []
    
    # 1. RGB COLOR STATISTICS (18 features - ENHANCED!)
    for color in ['Blue', 'Green', 'Red']:
        feature_names.extend([
            f'RGB_{color}_Mean',
            f'RGB_{color}_Std',
            f'RGB_{color}_Q10',
            f'RGB_{color}_Q25',
            f'RGB_{color}_Q75',
            f'RGB_{color}_Q90',
        ])
    
    # 2. HSV COLOR SPACE (18 features - ENHANCED!)
    for component in ['Hue', 'Saturation', 'Value']:
        feature_names.extend([
            f'HSV_{component}_Mean',
            f'HSV_{component}_Std',
            f'HSV_{component}_Q10',
            f'HSV_{component}_Q25',
            f'HSV_{component}_Q75',
            f'HSV_{component}_Q90',
        ])
    
    # 3. LAB COLOR SPACE (12 features - NEW!)
    for component in ['L', 'A', 'B']:
        feature_names.extend([
            f'LAB_{component}_Mean',
            f'LAB_{component}_Std',
            f'LAB_{component}_Q25',
            f'LAB_{component}_Q75',
        ])
    
    # 4. TEXTURE FEATURES - SOBEL GRADIENTS (8 features - ENHANCED!)
    feature_names.extend([
        'Sobel_Gradient_Mean',
        'Sobel_Gradient_Std',
        'Sobel_Gradient_Q75',
        'Sobel_Gradient_Q25',
        'Sobel_X_Mean',
        'Sobel_Y_Mean',
        'Sobel_Direction_Mean',
        'Sobel_Direction_Std',
    ])
    
    # 5. EDGE FEATURES (6 features - ENHANCED!)
    feature_names.extend([
        'Edge_Density',
        'Edge_Mean',
        'Edge_Std',
        'Edge_Q75',
        'Edge_Q90',
        'Edge_Q95',
    ])
    
    # 6. SPATIAL STATISTICS - QUADRANTS (12 features)
    for quad in ['TopLeft', 'TopRight', 'BottomLeft', 'BottomRight']:
        feature_names.extend([
            f'Quad_{quad}_Mean',
            f'Quad_{quad}_Std',
            f'Quad_{quad}_Q75',
        ])
    
    # 7. STATISTICAL MOMENTS (6 features - ENHANCED!)
    feature_names.extend([
        'Gray_Kurtosis',
        'Gray_Skewness',
        'Gray_Variance',
        'Gray_Range',
        'Gray_Median',
        'Gray_Entropy',
    ])
    
    return feature_names

# ============================================================
# HIERARCHICAL DATA LOADING
# ============================================================
def load_hierarchical_training_data(dataset_path, samples_per_class=None):
    """
    Load training data from hierarchical folder structure.
    Each main class (Urban, Vegetation, Water) contains multiple subfolders.
    Loads equal number of samples from each subfolder.
    
    Args:
        dataset_path: Path to dataset
        samples_per_class: Dict with class-specific sample sizes or int for all classes
    """
    
    # Default sample sizes if not provided
    if samples_per_class is None:
        samples_per_class = {
            'Urban': 3000,
            'Vegetation': 4000,
            'Water': 3000
        }
    elif isinstance(samples_per_class, int):
        # Convert int to dict for backward compatibility
        samples_per_class = {
            'Urban': samples_per_class,
            'Vegetation': samples_per_class,
            'Water': samples_per_class
        }
    
    # Define hierarchical class structure
    class_hierarchy = {
        'Urban': ['Highway', 'Industrial', 'Residential'],
        'Vegetation': ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Pasture', 'PermanentCrop'],
        'Water': ['River', 'SeaLake']
    }
    
    print("\n📋 Hierarchical Class Structure:")
    for main_class, subclasses in class_hierarchy.items():
        print(f"  {main_class}: {', '.join(subclasses)}")
    
    # Get user input for which main classes to use
    available_classes = list(class_hierarchy.keys())
    print(f"\n🎯 Available main classes: {available_classes}")
    print("Enter class indices (comma-separated, e.g., 0,1,2 for all):")
    for i, cls in enumerate(available_classes):
        print(f"  {i}: {cls}")
    
    while True:
        try:
            user_input = input("Select classes (e.g., 0,1,2): ").strip()
            class_indices = [int(x.strip()) for x in user_input.split(',')]
            if all(0 <= i < len(available_classes) for i in class_indices):
                break
            print(f"Please enter valid indices between 0 and {len(available_classes)-1}")
        except ValueError:
            print("Please enter comma-separated numbers")
    
    selected_main_classes = [available_classes[i] for i in class_indices]
    K = len(selected_main_classes)
    
    # Use the provided sample sizes dictionary
    if not isinstance(samples_per_class, dict):
        raise ValueError("samples_per_class must be a dictionary")
    
    class_sample_sizes = samples_per_class
    
    print(f"\n🎯 Selected {K} classes: {selected_main_classes}")
    print(f"📊 Loading samples per class:")
    for cls in selected_main_classes:
        print(f"   • {cls}: {class_sample_sizes[cls]} samples")
    print(f"   (Equally distributed among subfolders)")
    
    start_time = time.time()
    all_features = []
    all_labels = []
    class_names = selected_main_classes
    
    for label, main_class in enumerate(selected_main_classes):
        subfolders = class_hierarchy[main_class]
        samples_for_this_class = class_sample_sizes[main_class]
        samples_per_subfolder = samples_for_this_class // len(subfolders)
        
        print(f"\n   📂 Processing {main_class} (Class {label}):")
        print(f"      Total samples: {samples_for_this_class}")
        print(f"      Subfolders: {len(subfolders)}, Samples per subfolder: {samples_per_subfolder}")
        
        class_features = []
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(dataset_path, main_class, subfolder)
            
            if not os.path.exists(subfolder_path):
                print(f"      ⚠️ Warning: {subfolder_path} not found")
                continue
            
            # Get all image files from subfolder
            image_files = [f for f in os.listdir(subfolder_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
            
            # Limit to samples_per_subfolder
            image_files = image_files[:samples_per_subfolder]
            
            print(f"      Loading {subfolder}: {len(image_files)} samples", end=" ")
            
            subfolder_features = []
            for img_file in image_files:
                img_path = os.path.join(subfolder_path, img_file)
                features = extract_comprehensive_features(img_path)
                if features is not None:
                    subfolder_features.append(features)
            
            class_features.extend(subfolder_features)
            print(f"✅ ({len(subfolder_features)} loaded)")
        
        if len(class_features) > 0:
            all_features.extend(class_features)
            all_labels.extend([label] * len(class_features))
            print(f"   ✅ Total for {main_class}: {len(class_features)} samples")
        else:
            print(f"   ⚠️ No valid samples found for {main_class}")
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ Training data loaded in {elapsed_time:.1f} seconds")
    print(f"📊 Total samples: {len(all_features)}")
    print(f"📏 Feature vector size: {len(all_features[0]) if all_features else 0}")
    
    return np.array(all_features), np.array(all_labels), class_names, K

# ============================================================
# FAST FEATURE SELECTION - JEFFRIES-MATUSITA DISTANCE ONLY
# ============================================================
def compute_jeffries_matusita_distance_fast(mean1, var1, mean2, var2):
    """
    Compute Jeffries-Matusita distance for 1D feature (SIMPLIFIED & FAST).
    Range: [0, 2], where 2 indicates perfect separability.
    
    For 1D case:
    B = 0.125 * (mean1-mean2)^2 / avg_var + 0.5 * ln(avg_var / sqrt(var1*var2))
    JM = 2(1 - e^(-B))
    """
    # Average variance
    avg_var = (var1 + var2) / 2.0 + 1e-10  # Add small value for stability
    
    # Bhattacharyya distance (simplified for 1D)
    mean_diff = mean1 - mean2
    term1 = 0.125 * (mean_diff ** 2) / avg_var
    term2 = 0.5 * np.log(avg_var / (np.sqrt(var1 * var2) + 1e-10) + 1e-10)
    
    B = term1 + term2
    B = max(0, B)  # Ensure non-negative
    
    # Jeffries-Matusita distance
    JM = 2.0 * (1.0 - np.exp(-B))
    return JM

def compute_jeffries_matusita_scores_fast(X, y, n_classes):
    """
    FAST feature selection using ONLY Jeffries-Matusita Distance.
    JM is the BEST method for satellite imagery - no need for others!
    """
    n_features = X.shape[1]
    
    print("\n🎯 Computing Jeffries-Matusita scores (FAST method)...")
    
    jm_scores = np.zeros(n_features)
    
    # Compute class statistics for each feature
    for feat_idx in range(n_features):
        if feat_idx % 10 == 0:
            print(f"   Feature {feat_idx}/{n_features}...")
        
        feature_data = X[:, feat_idx]
        
        # Collect class means and variances for this feature
        class_means_list = []
        class_vars_list = []
        
        for k in range(n_classes):
            class_samples = feature_data[y == k]
            if len(class_samples) > 1:
                mean = np.mean(class_samples)
                var = np.var(class_samples) + 1e-10  # Add stability
                
                class_means_list.append(mean)
                class_vars_list.append(var)
        
        if len(class_means_list) < 2:
            continue
        
        # Compute pairwise JM distances (SIMPLIFIED - no matrix operations!)
        jm_sum = 0
        pair_count = 0
        
        for i in range(len(class_means_list)):
            for j in range(i + 1, len(class_means_list)):
                # Jeffries-Matusita Distance (fast 1D version)
                jm = compute_jeffries_matusita_distance_fast(
                    class_means_list[i], class_vars_list[i],
                    class_means_list[j], class_vars_list[j]
                )
                jm_sum += jm
                pair_count += 1
        
        # Average pairwise separability
        if pair_count > 0:
            jm_scores[feat_idx] = jm_sum / pair_count
    
    print("   ✅ Jeffries-Matusita computation complete")
    return jm_scores


def select_best_features(X, y, n_features=20):
    """
    FAST feature selection using ONLY Jeffries-Matusita Distance.
    This is much faster than using 4 different methods!
    """
    print(f"\n🎯 FAST FEATURE SELECTION (Jeffries-Matusita only)")
    print(f"   Target: Top {n_features} from {X.shape[1]} features")
    
    n_classes = len(np.unique(y))
    
    # Compute JM scores (FAST!)
    jm_scores = compute_jeffries_matusita_scores_fast(X, y, n_classes)
    
    print(f"\n📈 Jeffries-Matusita Statistics:")
    print(f"   Mean: {np.mean(jm_scores):.4f}")
    print(f"   Max:  {np.max(jm_scores):.4f}")
    print(f"   Min:  {np.min(jm_scores):.4f}")
    
    # Select top n_features based on JM distance
    selected_indices = np.argsort(jm_scores)[-n_features:]
    selected_indices = np.sort(selected_indices)  # Keep original order
    
    # Display top features
    top_scores = jm_scores[selected_indices]
    print(f"\n✅ Selected {len(selected_indices)} best features")
    print(f"   Top 5 feature indices: {selected_indices[-5:]}")
    print(f"   Top 5 JM scores: {[f'{s:.4f}' for s in top_scores[-5:]]}")
    print(f"   Average JM score: {np.mean(top_scores):.4f}")
    
    # Get feature names
    feature_names = get_feature_names()
    
    # Save detailed feature ranking with names
    feature_ranking = pd.DataFrame({
        'Feature_Index': range(X.shape[1]),
        'Feature_Name': feature_names,
        'Jeffries_Matusita': jm_scores,
        'Selected': ['Yes' if i in selected_indices else 'No' for i in range(X.shape[1])]
    })
    
    feature_ranking = feature_ranking.sort_values('Jeffries_Matusita', ascending=False)
    
    return selected_indices, jm_scores, feature_ranking

# ============================================================
# CLASSIFIER IMPLEMENTATIONS
# ============================================================
def classify_minimum_distance(features, class_means):
    """Enhanced minimum distance classifier"""
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    distances = np.array([np.linalg.norm(features - mean, axis=1) for mean in class_means])
    predictions = np.argmin(distances, axis=0)
    return predictions

def classify_maximum_likelihood(features, class_means, class_covariances, class_priors):
    """Robust maximum likelihood classifier with numerical stability"""
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    n_samples = features.shape[0]
    K = len(class_means)
    likelihoods = np.zeros((K, n_samples))
    
    for k in range(K):
        mean = class_means[k]
        cov = class_covariances[k]
        prior = class_priors[k]
        
        # Regularization for numerical stability
        regularization = 1e-6
        cov_stable = cov + regularization * np.eye(cov.shape[0])
        
        try:
            mv_normal = multivariate_normal(mean=mean, cov=cov_stable, allow_singular=True)
            log_likelihood = mv_normal.logpdf(features)
            likelihoods[k] = log_likelihood + np.log(prior + 1e-10)
        except:
            likelihoods[k] = -np.inf
    
    predictions = np.argmax(likelihoods, axis=0)
    return predictions

# ============================================================
# TRAINING AND EVALUATION
## ============================================================
# TRAIN CLASSIFIERS
# ============================================================
def train_classifiers(X_train, y_train, K):
    """Train both classifiers and compute parameters"""
    print("\n🎓 Training Classifiers...")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Number of classes: {K}")
    
    # Initialize storage
    class_means = []
    class_covariances = []
    class_priors = []
    
    # Compute parameters for each class
    for k in range(K):
        class_samples = X_train[y_train == k]
        
        if len(class_samples) == 0:
            print(f"   ⚠️ Warning: No samples for class {k}")
            continue
        
        # Compute mean
        mean = np.mean(class_samples, axis=0)
        class_means.append(mean)
        
        # Compute covariance with regularization
        if len(class_samples) > 1:
            cov = np.cov(class_samples.T)
            cov += 1e-6 * np.eye(cov.shape[0])  # regularization
        else:
            cov = np.eye(X_train.shape[1]) * 0.01
        
        class_covariances.append(cov)
        
        # Compute prior
        prior = len(class_samples) / len(X_train)
        class_priors.append(prior)
    
    return class_means, class_covariances, class_priors


# ============================================================
# EVALUATE CLASSIFIERS
# ============================================================
def evaluate_classifiers(X_test, y_test, class_means, class_covariances, class_priors):
    """Evaluate both classifiers"""
    print("\n📊 Evaluating Classifiers...")
    
    # Minimum Distance
    y_pred_min = classify_minimum_distance(X_test, class_means)
    acc_min = accuracy_score(y_test, y_pred_min)
    
    # Maximum Likelihood
    y_pred_ml = classify_maximum_likelihood(X_test, class_means, class_covariances, class_priors)
    acc_ml = accuracy_score(y_test, y_pred_ml)
    
    print(f"   🎯 Minimum Distance Accuracy: {acc_min*100:.2f}%")
    print(f"   🎯 Maximum Likelihood Accuracy: {acc_ml*100:.2f}%")
    
    return acc_min, acc_ml, y_pred_min, y_pred_ml


# ============================================================
# CORRECT CLASS STATISTICS FUNCTION
# ============================================================
def compute_class_statistics(X_train, y_train, class_names, K):
    """Compute correct between-class separation and within-class variation"""
    print("\n📈 Computing Class Statistics...")

    # ---- first compute all class means ----
    class_means = []
    for k in range(K):
        samples = X_train[y_train == k]
        mean_k = np.mean(samples, axis=0)
        class_means.append(mean_k)

    within_class_vars = []
    between_class_seps = []

    # ---- now compute both metrics for each class ----
    for k in range(K):
        samples = X_train[y_train == k]

        # Within-class variation
        within_var = np.mean(np.var(samples, axis=0))
        within_class_vars.append(within_var)

        # Between-class separation (correct formula)
        dists = [
            np.linalg.norm(class_means[k] - class_means[j])
            for j in range(K) if j != k
        ]
        between_sep = np.mean(dists)
        between_class_seps.append(between_sep)

        print(f"   Class {k} ({class_names[k]}):")
        print(f"      Within-class variation: {within_var:.4f}")
        print(f"      Between-class separation: {between_sep:.4f}")

    return between_class_seps, within_class_vars

# ============================================================
# VISUALIZATION
# ============================================================
def create_visualizations(acc_min, acc_ml, y_test, y_pred_min, y_pred_ml, 
                         class_names, between_class_seps, within_class_vars, output_dir):
    """Create comprehensive visualizations"""
    print("\n📊 Creating Visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(10, 6))
    methods = ['Minimum Distance', 'Maximum Likelihood']
    accuracies = [acc_min * 100, acc_ml * 100]
    colors = ['#3498db', '#e74c3c']
    
    bars = plt.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Classifier Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylim([0, 100])
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{acc:.2f}%', ha='center', fontsize=12, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=150)
    plt.close()
    
    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Minimum Distance
    cm_min = confusion_matrix(y_test, y_pred_min)
    im1 = axes[0].imshow(cm_min, cmap='Blues', aspect='auto')
    axes[0].set_title('Minimum Distance', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_yticks(range(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_yticklabels(class_names)
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            axes[0].text(j, i, str(cm_min[i, j]), ha='center', va='center')
    
    # Maximum Likelihood
    cm_ml = confusion_matrix(y_test, y_pred_ml)
    im2 = axes[1].imshow(cm_ml, cmap='Reds', aspect='auto')
    axes[1].set_title('Maximum Likelihood', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_yticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_yticklabels(class_names)
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            axes[1].text(j, i, str(cm_ml[i, j]), ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=150)
    plt.close()
    
    # 3. Class Statistics
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(class_names))
    width = 0.35
    
    axes[0].bar(x, between_class_seps, width, color='#2ecc71', alpha=0.7)
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Between-Class Separation')
    axes[0].set_title('Between-Class Separation', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].bar(x, within_class_vars, width, color='#e67e22', alpha=0.7)
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Within-Class Variation')
    axes[1].set_title('Within-Class Variation', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_statistics.png'), dpi=150)
    plt.close()
    
    print(f"   ✅ Visualizations saved to: {output_dir}")

def visualize_feature_selection(feature_ranking, output_dir, n_top=20):
    """Create visualization for Jeffries-Matusita feature selection"""
    print("\n📊 Creating Feature Selection Visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature names
    feature_names = get_feature_names()
    
    # Get top features
    top_features = feature_ranking.head(n_top)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Jeffries-Matusita Score (Top features)
    axes[0].barh(range(n_top), top_features['Jeffries_Matusita'].values, color='#3498db', alpha=0.7)
    axes[0].set_yticks(range(n_top))
    # Use actual feature names instead of "Feature X"
    feature_labels = [feature_names[idx] for idx in top_features['Feature_Index'].values]
    axes[0].set_yticklabels(feature_labels, fontsize=9)
    axes[0].set_xlabel('Jeffries-Matusita Distance', fontsize=11)
    axes[0].set_title(f'Top {n_top} Features - Jeffries-Matusita Scores', fontweight='bold', fontsize=13)
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # 2. Jeffries-Matusita Distribution with selection threshold
    axes[1].hist(feature_ranking['Jeffries_Matusita'].values, bins=25, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=top_features['Jeffries_Matusita'].min(), color='red', linestyle='--', linewidth=2, 
                    label=f'Selection Threshold ({top_features["Jeffries_Matusita"].min():.3f})')
    axes[1].set_xlabel('Jeffries-Matusita Distance', fontsize=10)
    axes[1].set_ylabel('Frequency', fontsize=10)
    axes[1].set_title('All Features - JM Distribution', fontweight='bold', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_selection_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Feature selection visualization saved")

# ============================================================
# MAIN FUNCTION
# ============================================================
def main(auto_mode=False, test_image=None):
    """Main execution function
    
    Args:
        auto_mode (bool): If True, automatically load model and classify without prompts
        test_image (str): Path to test image (only used in auto_mode)
    """
    
    # Check for existing model
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"\n📂 Found existing trained model: {MODEL_SAVE_PATH}")
        
        if auto_mode:
            user_choice = 'y'
        else:
            user_choice = input("Load existing model and classify new image? (y/n): ").strip().lower()
        
        if user_choice == 'y':
            model_data = load_model()
            if model_data is not None:
                print("✅ Model loaded successfully!")
                print(f"   Classes: {model_data['class_names']}")
                print(f"   Training accuracy (ML): {model_data['acc_ml']*100:.2f}%")
                
                # Ask for test image path
                if auto_mode and test_image:
                    test_img = test_image
                elif auto_mode:
                    test_img = TEST_IMAGE_PATH
                else:
                    test_img = input(f"\n📷 Enter test image path (press Enter for default '{TEST_IMAGE_PATH}'): ").strip()
                    if not test_img:
                        test_img = TEST_IMAGE_PATH
                
                if os.path.exists(test_img):
                    print(f"\n🔍 Classifying: {test_img}")
                    
                    # Load the original image for display
                    original_img = cv2.imread(test_img)
                    
                    # Extract features
                    features = extract_comprehensive_features(test_img)
                    if features is not None:
                        # Select features
                        selected_features_indices = model_data['selected_features']
                        features_selected = features[selected_features_indices].reshape(1, -1)
                        
                        # Scale features
                        features_scaled = model_data['scaler'].transform(features_selected)
                        
                        # Classify with both methods
                        pred_min = classify_minimum_distance(features_scaled, model_data['class_means'])
                        pred_ml = classify_maximum_likelihood(
                            features_scaled, 
                            model_data['class_means'], 
                            model_data['class_covariances'],
                            model_data['class_priors']
                        )
                        
                        class_min = model_data['class_names'][pred_min[0]]
                        class_ml = model_data['class_names'][pred_ml[0]]
                        
                        print(f"\n🎯 Classification Results:")
                        print(f"   Minimum Distance Classifier (MDC): {class_min}")
                        print(f"   Maximum Likelihood Classifier (MLC): {class_ml} ⭐")
                        
                        # TRUE PIXEL-BY-PIXEL CLASSIFICATION (Using Patches)
                        print(f"\n🎨 Performing patch-based pixel classification...")
                        h, w = original_img.shape[:2]
                        print(f"   Image size: {h}x{w} = {h*w:,} pixels")
                        print(f"   Using overlapping patches for accurate pixel classification...")
                        
                        # Create classification maps for EACH pixel
                        classification_map_mdc_full = np.zeros((h, w), dtype=np.uint8)
                        classification_map_mlc_full = np.zeros((h, w), dtype=np.uint8)
                        
                        # Use larger patches (16x16) and resize to IMAGE_SIZE for feature extraction
                        patch_size = 16
                        stride = 8  # Overlapping patches for smoother results
                        
                        print(f"   Using {patch_size}x{patch_size} patches with stride {stride}...")
                        
                        # Temporary directory for patch processing
                        import tempfile
                        temp_dir = tempfile.mkdtemp()
                        
                        # Calculate number of patches
                        num_patches_h = (h - patch_size) // stride + 1
                        num_patches_w = (w - patch_size) // stride + 1
                        total_patches = num_patches_h * num_patches_w
                        
                        print(f"   Processing {total_patches:,} patches...")
                        
                        # Arrays to accumulate votes for each pixel
                        mdc_votes = np.zeros((h, w, 3), dtype=np.int32)  # 3 classes
                        mlc_votes = np.zeros((h, w, 3), dtype=np.int32)
                        
                        patch_count = 0
                        for i in range(0, h - patch_size + 1, stride):
                            for j in range(0, w - patch_size + 1, stride):
                                # Extract patch
                                patch = original_img[i:i+patch_size, j:j+patch_size]
                                
                                # Resize patch to IMAGE_SIZE and save temporarily
                                patch_resized = cv2.resize(patch, (IMAGE_SIZE, IMAGE_SIZE))
                                temp_patch_path = os.path.join(temp_dir, 'temp_patch.jpg')
                                cv2.imwrite(temp_patch_path, patch_resized)
                                
                                # Extract full 80 features from patch
                                try:
                                    patch_features = extract_comprehensive_features(temp_patch_path)
                                    
                                    if patch_features is not None:
                                        # Select and scale features
                                        patch_features_selected = patch_features[selected_features_indices].reshape(1, -1)
                                        patch_features_scaled = model_data['scaler'].transform(patch_features_selected)
                                        
                                        # Classify patch
                                        pred_mdc = classify_minimum_distance(patch_features_scaled, model_data['class_means'])[0]
                                        pred_mlc = classify_maximum_likelihood(
                                            patch_features_scaled,
                                            model_data['class_means'],
                                            model_data['class_covariances'],
                                            model_data['class_priors']
                                        )[0]
                                        
                                        # Vote for all pixels in this patch
                                        mdc_votes[i:i+patch_size, j:j+patch_size, pred_mdc] += 1
                                        mlc_votes[i:i+patch_size, j:j+patch_size, pred_mlc] += 1
                                    
                                except Exception as e:
                                    # If feature extraction fails, skip this patch
                                    pass
                                
                                patch_count += 1
                                if patch_count % 100 == 0:
                                    print(f"   Progress: {patch_count:,}/{total_patches:,} patches ({100*patch_count/total_patches:.1f}%)...", end='\r')
                        
                        # Clean up temp directory
                        import shutil
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        
                        print(f"\n   ✅ Patch processing complete! Processed {patch_count:,} patches")
                        
                        # Assign final classification based on majority vote
                        print(f"   Computing final pixel classifications from votes...")
                        for i in range(h):
                            for j in range(w):
                                classification_map_mdc_full[i, j] = np.argmax(mdc_votes[i, j])
                                classification_map_mlc_full[i, j] = np.argmax(mlc_votes[i, j])
                        
                        print(f"   ✅ Pixel-level classification complete! Each pixel classified based on its local patch")
                        
                        # Define colors for each class
                        class_colors = {
                            0: [255, 0, 0],      # Urban - Red
                            1: [0, 255, 0],      # Vegetation - Green
                            2: [0, 0, 255]       # Water - Blue
                        }
                        
                        class_colors_normalized = {
                            0: [1.0, 0.0, 0.0],      # Urban - Red
                            1: [0.0, 1.0, 0.0],      # Vegetation - Green
                            2: [0.0, 0.0, 1.0]       # Water - Blue
                        }
                        
                        # Create colored classification maps
                        colored_map_mdc = np.zeros((h, w, 3), dtype=np.uint8)
                        colored_map_mlc = np.zeros((h, w, 3), dtype=np.uint8)
                        
                        for class_id, color in class_colors.items():
                            colored_map_mdc[classification_map_mdc_full == class_id] = color
                            colored_map_mlc[classification_map_mlc_full == class_id] = color
                        
                        # Calculate TEST IMAGE class statistics (not training stats!)
                        print(f"\n📊 Computing test image statistics...")
                        
                        # Convert image to feature space for better analysis
                        img_hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
                        
                        # Extract features from each classified region (test image)
                        test_class_stats = []
                        class_feature_vectors = []  # Store feature vectors (RGB means) for each class

                        # We'll compute per-class mean RGB vectors for the test image and use them
                        # to compute between-class separations and within-class variances.
                        for class_id, class_name in enumerate(model_data['class_names']):
                            # Get pixels belonging to this class (using MLC)
                            class_mask = (classification_map_mlc_full == class_id)
                            pixel_count = int(np.sum(class_mask))

                            if pixel_count > 0:
                                # Get pixel values for this class
                                class_pixels_bgr = original_img[class_mask].astype(np.float64)
                                class_pixels_hsv = img_hsv[class_mask].astype(np.float64)

                                # Calculate RGB statistics
                                mean_rgb = np.mean(class_pixels_bgr, axis=0)
                                std_rgb = np.std(class_pixels_bgr, axis=0)

                                # Calculate HSV statistics for better color analysis
                                mean_hsv = np.mean(class_pixels_hsv, axis=0)
                                std_hsv = np.std(class_pixels_hsv, axis=0)

                                # Create feature vector for this class (RGB + HSV)
                                feature_vector = np.concatenate([mean_rgb, std_rgb, mean_hsv, std_hsv])
                                class_feature_vectors.append(feature_vector)
                                
                                # Calculate within-class variation (average std of all channels)
                                within_class_var = np.mean(np.concatenate([std_rgb, std_hsv]))
                                
                                test_class_stats.append({
                                    'class': class_name,
                                    'class_id': class_id,
                                    'pixel_count': pixel_count,
                                    'percentage': (pixel_count / (h*w)) * 100,
                                    'mean_rgb': mean_rgb.tolist(),
                                    'std_rgb': std_rgb.tolist(),
                                    'mean_hsv': mean_hsv.tolist(),
                                    'std_hsv': std_hsv.tolist(),
                                    'feature_vector': feature_vector,
                                    'within_class_variation': within_class_var
                                })
                            else:
                                # No pixels predicted for this class in test image
                                # Use training statistics as fallback
                                print(f"   ⚠️ No pixels for class '{class_name}' in test image — using training stats fallback.")
                                
                                # Use training class means as fallback
                                if hasattr(model_data['class_means'], '__iter__'):
                                    fallback_mean = model_data['class_means'][class_id][:3] if len(model_data['class_means'][class_id]) >= 3 else np.array([128, 128, 128])
                                else:
                                    fallback_mean = np.array([128, 128, 128])
                                
                                feature_vector = np.concatenate([fallback_mean, np.zeros(3), np.zeros(3), np.zeros(3)])
                                class_feature_vectors.append(feature_vector)
                                
                                test_class_stats.append({
                                    'class': class_name,
                                    'class_id': class_id,
                                    'pixel_count': 0,
                                    'percentage': 0.0,
                                    'mean_rgb': fallback_mean.tolist(),
                                    'std_rgb': np.zeros(3).tolist(),
                                    'mean_hsv': np.zeros(3).tolist(),
                                    'std_hsv': np.zeros(3).tolist(),
                                    'feature_vector': feature_vector,
                                    'within_class_variation': 0.0
                                })
                        
                        # Calculate between-class separation
                        between_class_separations = []
                        class_feature_array = np.array([s['feature_vector'] for s in test_class_stats])
                        
                        if len(class_feature_array) > 1:
                            # Calculate pairwise distances between class means
                            for i in range(len(class_feature_array)):
                                for j in range(i+1, len(class_feature_array)):
                                    dist = np.linalg.norm(class_feature_array[i] - class_feature_array[j])
                                    between_class_separations.append(dist)
                            
                            avg_between_class_sep = np.mean(between_class_separations) if between_class_separations else 0.0
                        else:
                            avg_between_class_sep = 0.0
                        
                        # Assign between-class separation to each class
                        for stat in test_class_stats:
                            stat['between_class_separation'] = avg_between_class_sep
                        
                        print(f"   ✅ Test image statistics computed!")
                        print(f"      Between-class separation: {avg_between_class_sep:.2f}")
                        avg_within = np.mean([s['within_class_variation'] for s in test_class_stats if s['pixel_count'] > 0])
                        print(f"      Average within-class variation: {avg_within:.2f}" if avg_within > 0 else "      Average within-class variation: N/A")
                        
                        # Compute between-class separations and within-class variances (TEST IMAGE)
                        print(f"\n🔎 Extracting per-class statistics for visualization...")
                        class_feature_vectors = np.array(class_feature_vectors, dtype=np.float64)

                        # Between-class separations: pairwise Euclidean distance between class mean RGB vectors
                        K = len(model_data['class_names'])
                        between_seps = np.zeros((K, K), dtype=np.float64)
                        for i in range(K):
                            for j in range(K):
                                between_seps[i, j] = np.linalg.norm(class_feature_vectors[i] - class_feature_vectors[j])

                        # Summary between-class separation (mean of pairwise distances for each class)
                        between_class_summary = np.mean(between_seps, axis=1)

                        # Within-class variation: approximate by mean RGB channel variance within predicted pixels
                        within_class_vars = []
                        for idx, stat in enumerate(test_class_stats):
                            if stat['pixel_count'] > 0:
                                # Use std_rgb values
                                within_var = float(np.mean(stat['std_rgb']))
                            else:
                                # Fallback to training within-class variation if available
                                fallback_cov = model_data.get('class_covariances')
                                if fallback_cov is not None:
                                    # approximate by mean of diagonal of covariance matrix
                                    within_var = float(np.mean(np.diag(fallback_cov[idx])))
                                else:
                                    within_var = 0.0
                            within_class_vars.append(within_var)

                        # Save test statistics into a dedicated folder under OUTPUT_FOLDER
                        test_stats_dir = os.path.join(OUTPUT_FOLDER, 'test_stats')
                        os.makedirs(test_stats_dir, exist_ok=True)

                        # Save per-class test stats CSV
                        stats_out_path = os.path.join(test_stats_dir, 'test_image_class_stats.csv')
                        rows = []
                        for s, sep, wvar in zip(test_class_stats, between_class_summary, within_class_vars):
                            rows.append({
                                'Class': s['class'],
                                'Pixel_Count': int(s['pixel_count']),
                                'Percentage': float(s['percentage']),
                                'Mean_R_RGB': float(s['mean_rgb'][0]) if len(s['mean_rgb'])>=1 else 0.0,
                                'Mean_G_RGB': float(s['mean_rgb'][1]) if len(s['mean_rgb'])>=2 else 0.0,
                                'Mean_B_RGB': float(s['mean_rgb'][2]) if len(s['mean_rgb'])>=3 else 0.0,
                                'Within_Class_Var': float(wvar),
                                'Between_Class_Sep': float(sep)
                            })

                        stats_df_test = pd.DataFrame(rows)
                        stats_df_test.to_csv(stats_out_path, index=False)
                        print(f"   ✅ Test image statistics saved to: {stats_out_path}")

                        # Create visualization for test image class statistics
                        fig_stats = plt.figure(figsize=(16, 10))
                        gs_stats = fig_stats.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
                        
                        # 1. Class Distribution (Percentage)
                        ax_dist = fig_stats.add_subplot(gs_stats[0, 0])
                        class_names_plot = [s['class'] for s in test_class_stats]
                        percentages = [s['percentage'] for s in test_class_stats]
                        colors_bars = [class_colors_normalized[i] for i in range(len(class_names_plot))]
                        
                        bars1 = ax_dist.bar(class_names_plot, percentages, color=colors_bars, alpha=0.7, edgecolor='black', linewidth=2)
                        ax_dist.set_ylabel('Percentage of Image (%)', fontsize=12, fontweight='bold')
                        ax_dist.set_title('Test Image: Class Distribution', fontsize=14, fontweight='bold')
                        ax_dist.set_ylim(0, max(percentages) * 1.2 if max(percentages) > 0 else 100)
                        ax_dist.grid(axis='y', alpha=0.3)
                        
                        for bar, pct in zip(bars1, percentages):
                            height = bar.get_height()
                            ax_dist.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
                        
                        # 2. Pixel Counts
                        ax_pixels = fig_stats.add_subplot(gs_stats[0, 1])
                        pixel_counts = [s['pixel_count'] for s in test_class_stats]
                        
                        bars2 = ax_pixels.bar(class_names_plot, pixel_counts, color=colors_bars, alpha=0.7, edgecolor='black', linewidth=2)
                        ax_pixels.set_ylabel('Number of Pixels', fontsize=12, fontweight='bold')
                        ax_pixels.set_title('Test Image: Pixel Count per Class', fontsize=14, fontweight='bold')
                        ax_pixels.grid(axis='y', alpha=0.3)
                        
                        for bar, count in zip(bars2, pixel_counts):
                            height = bar.get_height()
                            ax_pixels.text(bar.get_x() + bar.get_width()/2., height,
                                          f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
                        
                        # 3. Between-Class Separation
                        ax_between = fig_stats.add_subplot(gs_stats[1, 0])
                        
                        bars3 = ax_between.bar(class_names_plot, between_class_summary, color='steelblue', alpha=0.7, edgecolor='black', linewidth=2)
                        ax_between.set_ylabel('Separation Distance', fontsize=12, fontweight='bold')
                        ax_between.set_title('Between-Class Separation (Higher = Better)', fontsize=14, fontweight='bold')
                        ax_between.grid(axis='y', alpha=0.3)
                        
                        for bar, sep in zip(bars3, between_class_summary):
                            height = bar.get_height()
                            ax_between.text(bar.get_x() + bar.get_width()/2., height,
                                           f'{sep:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
                        
                        # 4. Within-Class Variation
                        ax_within = fig_stats.add_subplot(gs_stats[1, 1])
                        
                        bars4 = ax_within.bar(class_names_plot, within_class_vars, color='coral', alpha=0.7, edgecolor='black', linewidth=2)
                        ax_within.set_ylabel('Variation (Std Dev)', fontsize=12, fontweight='bold')
                        ax_within.set_title('Within-Class Variation (Lower = Better)', fontsize=14, fontweight='bold')
                        ax_within.grid(axis='y', alpha=0.3)
                        
                        for bar, var in zip(bars4, within_class_vars):
                            height = bar.get_height()
                            ax_within.text(bar.get_x() + bar.get_width()/2., height,
                                          f'{var:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
                        
                        fig_stats.suptitle('Test Image: Class Statistics Analysis', fontsize=16, fontweight='bold', y=0.98)
                        
                        # Save the test statistics visualization
                        stats_png_path = os.path.join(test_stats_dir, 'test_image_class_stats.png')
                        fig_stats.savefig(stats_png_path, dpi=150, bbox_inches='tight')
                        plt.close(fig_stats)
                        print(f"   ✅ Test statistics visualization saved to: {stats_png_path}")

                        # Create comprehensive visualization
                        fig = plt.figure(figsize=(24, 16))
                        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
                        
                        # Create comprehensive visualization
                        fig = plt.figure(figsize=(24, 16))
                        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
                        
                        # Row 1: Original and Classification Maps
                        ax1 = fig.add_subplot(gs[0, 0:2])
                        ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
                        ax1.set_title('Original Test Image', fontsize=14, fontweight='bold')
                        ax1.axis('off')
                        
                        ax2 = fig.add_subplot(gs[0, 2])
                        ax2.imshow(colored_map_mdc)
                        ax2.set_title(f'MDC: Pixel-by-Pixel\nClassification', 
                                     fontsize=12, fontweight='bold', color='blue')
                        ax2.axis('off')
                        
                        ax3 = fig.add_subplot(gs[0, 3])
                        ax3.imshow(colored_map_mlc)
                        ax3.set_title(f'MLC: Pixel-by-Pixel\nClassification ⭐', 
                                     fontsize=12, fontweight='bold', color='green')
                        ax3.axis('off')
                        
                        # Row 2: Detailed Classification Views with Zoomed Sections
                        # Show a zoomed region to see pixel-level detail
                        zoom_h, zoom_w = h // 4, w // 4
                        zoom_y, zoom_x = h // 3, w // 3
                        
                        ax4 = fig.add_subplot(gs[1, 0])
                        zoom_original = original_img[zoom_y:zoom_y+zoom_h, zoom_x:zoom_x+zoom_w]
                        ax4.imshow(cv2.cvtColor(zoom_original, cv2.COLOR_BGR2RGB))
                        ax4.set_title('Zoomed Region (Original)', fontsize=11, fontweight='bold')
                        ax4.axis('off')
                        
                        ax5 = fig.add_subplot(gs[1, 1])
                        zoom_mdc = colored_map_mdc[zoom_y:zoom_y+zoom_h, zoom_x:zoom_x+zoom_w]
                        ax5.imshow(zoom_mdc)
                        ax5.set_title('Zoomed Region (MDC)\nEach pixel classified', fontsize=11, fontweight='bold', color='blue')
                        ax5.axis('off')
                        
                        ax6 = fig.add_subplot(gs[1, 2])
                        zoom_mlc = colored_map_mlc[zoom_y:zoom_y+zoom_h, zoom_x:zoom_x+zoom_w]
                        ax6.imshow(zoom_mlc)
                        ax6.set_title('Zoomed Region (MLC)\nEach pixel classified', fontsize=11, fontweight='bold', color='green')
                        ax6.axis('off')
                        
                        # Legend and Class Info
                        ax7 = fig.add_subplot(gs[1, 3])
                        ax7.axis('off')
                        legend_text = "CLASSIFICATION KEY\n" + "="*35 + "\n\n"
                        legend_text += "Pixel Colors:\n"
                        for i, class_name in enumerate(model_data['class_names']):
                            color_name = ['🔴 Red', '🟢 Green', '🔵 Blue'][i]
                            legend_text += f"{color_name} = {class_name}\n"
                        legend_text += "\n" + "="*35 + "\n"
                        legend_text += f"Model Accuracy: {model_data['acc_ml']*100:.2f}%\n"
                        legend_text += f"Features Used: {len(selected_features_indices)}\n"
                        legend_text += f"Total Pixels: {h*w:,}\n"
                        legend_text += f"Image Size: {h}×{w}"
                        
                        ax7.text(0.5, 0.5, legend_text, ha='center', va='center',
                                fontsize=10, family='monospace',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                        
                        # Row 3: Test Image Statistics
                        ax8 = fig.add_subplot(gs[2, 0])
                        
                        # Create bar chart for test image class distribution
                        class_names_list = [s['class'] for s in test_class_stats]
                        class_percentages = [s['percentage'] for s in test_class_stats]
                        colors_for_bars = [class_colors_normalized[i] for i in range(len(class_names_list))]
                        
                        bars = ax8.bar(class_names_list, class_percentages, color=colors_for_bars, alpha=0.7, edgecolor='black')
                        ax8.set_ylabel('Percentage of Image (%)', fontsize=11, fontweight='bold')
                        ax8.set_title('Test Image: Class Distribution', fontsize=12, fontweight='bold')
                        ax8.set_ylim(0, max(class_percentages) * 1.2 if max(class_percentages) > 0 else 100)
                        ax8.grid(axis='y', alpha=0.3)
                        
                        # Add percentage labels on bars
                        for bar, pct in zip(bars, class_percentages):
                            height = bar.get_height()
                            ax8.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{pct:.1f}%',
                                    ha='center', va='bottom', fontweight='bold', fontsize=10)
                        
                        # NEW: Between/Within Class Variation Chart
                        ax8b = fig.add_subplot(gs[2, 1])
                        
                        # Get classes that actually have pixels
                        classes_with_pixels = [s for s in test_class_stats if s['pixel_count'] > 0]
                        
                        if len(classes_with_pixels) > 0:
                            class_names_present = [s['class'] for s in classes_with_pixels]
                            between_sep = [s['between_class_separation'] for s in classes_with_pixels]
                            within_var = [s['within_class_variation'] for s in classes_with_pixels]
                            
                            x = np.arange(len(class_names_present))
                            width = 0.35
                            
                            bars1 = ax8b.bar(x - width/2, between_sep, width, label='Between-Class Sep', 
                                           color='steelblue', alpha=0.7, edgecolor='black')
                            bars2 = ax8b.bar(x + width/2, within_var, width, label='Within-Class Var', 
                                           color='coral', alpha=0.7, edgecolor='black')
                            
                            ax8b.set_ylabel('Value', fontsize=11, fontweight='bold')
                            ax8b.set_title('Test Image: Class Separability', fontsize=12, fontweight='bold')
                            ax8b.set_xticks(x)
                            ax8b.set_xticklabels(class_names_present)
                            ax8b.legend(fontsize=9)
                            ax8b.grid(axis='y', alpha=0.3)
                            
                            # Add value labels on bars
                            for bar in bars1:
                                height = bar.get_height()
                                ax8b.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{height:.1f}',
                                        ha='center', va='bottom', fontsize=8)
                            for bar in bars2:
                                height = bar.get_height()
                                ax8b.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{height:.1f}',
                                        ha='center', va='bottom', fontsize=8)
                        else:
                            ax8b.text(0.5, 0.5, 'No classified pixels', ha='center', va='center')
                            ax8b.axis('off')
                            height = bar.get_height()
                            ax8.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{pct:.1f}%',
                                    ha='center', va='bottom', fontweight='bold', fontsize=10)
                        
                        # Pixel count details
                        ax9 = fig.add_subplot(gs[2, 2:4])
                        ax9.axis('off')
                        
                        stats_text = "TEST IMAGE PIXEL STATISTICS\n" + "="*50 + "\n\n"
                        stats_text += "MLC Classification Results:\n"
                        for stat in test_class_stats:
                            stats_text += f"  {stat['class']:12s}: {stat['pixel_count']:8,} pixels ({stat['percentage']:5.1f}%)\n"
                        
                        stats_text += "\n" + "="*50 + "\n"
                        stats_text += f"Total Pixels Classified: {h*w:,}\n"
                        stats_text += f"Overall Prediction: {class_ml} ⭐\n"
                        stats_text += f"MDC Prediction: {class_min}"
                        
                        ax9.text(0.5, 0.5, stats_text, ha='center', va='center',
                                fontsize=10, family='monospace',
                                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
                        
                        # Row 4: Feature Ranking
                        ax10 = fig.add_subplot(gs[3, :])
                        
                        # Load feature ranking if available (from OUTPUT_FOLDER)
                        feature_ranking_path = os.path.join(OUTPUT_FOLDER, 'feature_ranking.csv')
                        if os.path.exists(feature_ranking_path):
                            feature_df = pd.read_csv(feature_ranking_path)
                            top_n = min(15, len(feature_df))  # Show top 15 features
                            
                            top_features = feature_df.head(top_n)
                            feature_names = top_features['Feature_Name'].values
                            jm_scores = top_features['Jeffries_Matusita'].values
                            
                            # Create horizontal bar chart
                            y_pos = np.arange(len(feature_names))
                            bars = ax10.barh(y_pos, jm_scores, color='steelblue', alpha=0.7, edgecolor='black')
                            
                            # Color code bars based on JM score
                            for i, (bar, score) in enumerate(zip(bars, jm_scores)):
                                if score > 1.8:
                                    bar.set_color('darkgreen')
                                elif score > 1.5:
                                    bar.set_color('forestgreen')
                                elif score > 1.2:
                                    bar.set_color('orange')
                                else:
                                    bar.set_color('coral')
                            
                            ax10.set_yticks(y_pos)
                            ax10.set_yticklabels(feature_names, fontsize=9)
                            ax10.set_xlabel('Jeffries-Matusita Distance', fontsize=11, fontweight='bold')
                            ax10.set_title(f'Top {top_n} Most Discriminative Features (Used in Classification)', 
                                         fontsize=12, fontweight='bold')
                            ax10.set_xlim(0, 2.0)
                            ax10.grid(axis='x', alpha=0.3)
                            ax10.invert_yaxis()
                            
                            # Add JM score labels
                            for i, (bar, score) in enumerate(zip(bars, jm_scores)):
                                ax10.text(score + 0.05, i, f'{score:.3f}', 
                                        va='center', fontsize=8, fontweight='bold')
                            
                            # Add legend for color coding
                            from matplotlib.patches import Patch
                            legend_elements = [
                                Patch(facecolor='darkgreen', label='Excellent (>1.8)'),
                                Patch(facecolor='forestgreen', label='Very Good (1.5-1.8)'),
                                Patch(facecolor='orange', label='Good (1.2-1.5)'),
                                Patch(facecolor='coral', label='Moderate (<1.2)')
                            ]
                            ax10.legend(handles=legend_elements, loc='lower right', fontsize=9)
                        else:
                            ax10.text(0.5, 0.5, 'Feature ranking data not available', 
                                     ha='center', va='center', fontsize=12)
                            ax10.axis('off')
                        
                        # Main title
                        fig.suptitle('Complete Satellite Image Classification Analysis', 
                                   fontsize=16, fontweight='bold', y=0.98)
                        
                        # Create output directory if it doesn't exist
                        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
                        
                        # Save the comprehensive result
                        output_path = os.path.join(OUTPUT_FOLDER, 'classified_result.png')
                        plt.savefig(output_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        print(f"\n💾 Classification result saved to: {output_path}")
                        print(f"💡 Comprehensive visualization includes:")
                        print(f"   - Original image with pixel-by-pixel classification maps")
                        print(f"   - Zoomed regions showing individual pixel classifications")
                        print(f"   - Test image class distribution statistics")
                        print(f"   - Top 15 discriminative features with JM scores")
                        print(f"   - Color-coded: Red=Urban, Green=Vegetation, Blue=Water")
                        
                        # Save test image statistics to CSV
                        print(f"\n💾 Saving test image statistics...")
                        test_stats_df = pd.DataFrame({
                            'Class_Name': [s['class'] for s in test_class_stats],
                            'Pixel_Count': [s['pixel_count'] for s in test_class_stats],
                            'Percentage': [s['percentage'] for s in test_class_stats],
                            'Between_Class_Separation': [s['between_class_separation'] for s in test_class_stats],
                            'Within_Class_Variation': [s['within_class_variation'] for s in test_class_stats],
                            'Mean_R': [s['mean_rgb'][2] for s in test_class_stats],
                            'Mean_G': [s['mean_rgb'][1] for s in test_class_stats],
                            'Mean_B': [s['mean_rgb'][0] for s in test_class_stats],
                            'Std_R': [s['std_rgb'][2] for s in test_class_stats],
                            'Std_G': [s['std_rgb'][1] for s in test_class_stats],
                            'Std_B': [s['std_rgb'][0] for s in test_class_stats]
                        })
                        
                        test_stats_path = os.path.join(OUTPUT_FOLDER, 'test_image_statistics.csv')
                        test_stats_df.to_csv(test_stats_path, index=False)
                        print(f"   ✅ Test statistics saved to: test_image_statistics.csv")
                        
                        # Save classification maps as separate images
                        print(f"\n💾 Saving individual classification maps...")
                        cv2.imwrite(os.path.join(OUTPUT_FOLDER, 'mdc_classification_map.png'), colored_map_mdc)
                        cv2.imwrite(os.path.join(OUTPUT_FOLDER, 'mlc_classification_map.png'), colored_map_mlc)
                        print(f"   ✅ MDC map saved: mdc_classification_map.png")
                        print(f"   ✅ MLC map saved: mlc_classification_map.png")
                        
                        # Summary of all files
                        print(f"\n✨ Test Image Results saved in: {OUTPUT_FOLDER}")
                        print(f"   📊 Classification Maps:")
                        print(f"      - classified_result.png (comprehensive 3-panel visualization)")
                        print(f"      - mdc_classification_map.png (MDC pixel classification)")
                        print(f"      - mlc_classification_map.png (MLC pixel classification)")
                        print(f"   📈 Statistics:")
                        print(f"      - test_image_statistics.csv (summary stats)")
                        print(f"      - test_stats/test_image_class_stats.csv (detailed per-class stats)")
                        print(f"      - test_stats/test_image_class_stats.png (bar chart visualization)")
                        
                        print(f"\n✨ Training Analysis Files (from when model was trained):")
                        print(f"   📂 Location: {TRAINING_OUTPUT_FOLDER}")
                        print(f"   📊 Visualizations:")
                        print(f"      - confusion_matrices.png (MDC & MLC confusion matrices)")
                        print(f"      - accuracy_comparison.png (model performance)")
                        print(f"      - class_statistics.png (training class statistics)")
                        print(f"      - feature_selection_analysis.png (JM distance ranking)")
                        print(f"   📄 Data:")
                        print(f"      - class_statistics.csv (training stats)")
                        print(f"      - feature_ranking.csv (all features ranked by JM distance)")
                        
                        print(f"\n💡 All test results saved to: {OUTPUT_FOLDER}")
                        print(f"💡 Model is ready! Change TEST_IMAGE_PATH and run again.")
                    else:
                        print("❌ Could not extract features from image")
                else:
                    print(f"❌ Image not found: {test_img}")
                
                return
    
    # STEP 1: Load Training Data
    print("\n" + "="*60)
    print("STEP 1: LOADING TRAINING DATA")
    print("="*60)
    
    X, y, class_names, K = load_hierarchical_training_data(
        SATELLITE_DATASET_PATH,
        samples_per_class=CLASS_SAMPLE_SIZES
    )
    
    if X is None or len(X) == 0:
        print("❌ Error: No training data loaded")
        return
    
    # STEP 2: Feature Selection
    print("\n" + "="*60)
    print("STEP 2: ADVANCED FEATURE SELECTION")
    print("="*60)
    
    selected_features, feature_scores, feature_ranking = select_best_features(X, y, n_features=NUM_BEST_FEATURES)
    X_selected = X[:, selected_features]
    
    print(f"\n   Original features: {X.shape[1]}")
    print(f"   Selected features: {X_selected.shape[1]}")
    
    # Save feature ranking
    os.makedirs(TRAINING_OUTPUT_FOLDER, exist_ok=True)
    ranking_path = os.path.join(TRAINING_OUTPUT_FOLDER, 'feature_ranking.csv')
    feature_ranking.to_csv(ranking_path, index=False)
    print(f"   📊 Feature ranking saved to: {ranking_path}")
    
    # STEP 3: Split Data
    print("\n" + "="*60)
    print("STEP 3: SPLITTING DATA")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    print(f"   Training/Test split: {100*(1-TEST_SIZE):.0f}%/{100*TEST_SIZE:.0f}%")
    
    # STEP 4: Preprocessing
    print("\n" + "="*60)
    print("STEP 4: PREPROCESSING")
    print("="*60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("   ✅ Features standardized (zero mean, unit variance)")
    
    # STEP 5: Train Classifiers
    print("\n" + "="*60)
    print("STEP 5: TRAINING CLASSIFIERS")
    print("="*60)
    
    class_means, class_covariances, class_priors = train_classifiers(X_train_scaled, y_train, K)
    
    # STEP 6: Evaluate
    print("\n" + "="*60)
    print("STEP 6: EVALUATION")
    print("="*60)
    
    acc_min, acc_ml, y_pred_min, y_pred_ml = evaluate_classifiers(
        X_test_scaled, y_test, class_means, class_covariances, class_priors
    )
    
    # STEP 7: Compute Statistics
    print("\n" + "="*60)
    print("STEP 7: CLASS STATISTICS")
    print("="*60)
    
    between_class_seps, within_class_vars = compute_class_statistics(
        X_train_scaled, y_train, class_names, K
    )
    
    # STEP 8: Visualizations
    print("\n" + "="*60)
    print("STEP 8: CREATING VISUALIZATIONS")
    print("="*60)
    
    create_visualizations(
        acc_min, acc_ml, y_test, y_pred_min, y_pred_ml,
        class_names, between_class_seps, within_class_vars, TRAINING_OUTPUT_FOLDER
    )
    
    # Visualize feature selection results
    visualize_feature_selection(feature_ranking, TRAINING_OUTPUT_FOLDER, n_top=NUM_BEST_FEATURES)
    
    # STEP 9: Save Statistics
    print("\n" + "="*60)
    print("STEP 9: SAVING RESULTS")
    print("="*60)
    
    os.makedirs(TRAINING_OUTPUT_FOLDER, exist_ok=True)
    
    stats_df = pd.DataFrame({
        'Class_Name': class_names,
        'Between_Class_Separation': between_class_seps,
        'Within_Class_Variation': within_class_vars,
        'Sample_Count': [np.sum(y_train == k) for k in range(K)]
    })
    
    stats_path = os.path.join(TRAINING_OUTPUT_FOLDER, 'class_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"   ✅ Statistics saved to: {stats_path}")
    
    # STEP 10: Save Model
    print("\n" + "="*60)
    print("STEP 10: SAVING MODEL")
    print("="*60)
    
    model_data = {
        'class_means': class_means,
        'class_covariances': class_covariances,
        'class_priors': class_priors,
        'class_names': class_names,
        'K': K,
        'scaler': scaler,
        'selected_features': selected_features,
        'feature_scores': feature_scores,
        'feature_ranking': feature_ranking,
        'acc_min': acc_min,
        'acc_ml': acc_ml,
        'between_class_seps': between_class_seps,
        'within_class_vars': within_class_vars,
        'samples_per_class': CLASS_SAMPLE_SIZES,
        'num_features': NUM_BEST_FEATURES
    }
    
    save_model(model_data)
    
    # Final Summary
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"✅ Classes trained: {K}")
    print(f"✅ Total samples: {len(X)}")
    print(f"✅ Features used: {NUM_BEST_FEATURES}/{X.shape[1]}")
    print(f"✅ Minimum Distance Accuracy: {acc_min*100:.2f}%")
    print(f"✅ Maximum Likelihood Accuracy: {acc_ml*100:.2f}%")
    print(f"✅ Model saved to: {MODEL_SAVE_PATH}")
    print(f"✅ Training visualizations saved to: {TRAINING_OUTPUT_FOLDER}")
    
    # AUTOMATICALLY PROCEED TO CLASSIFICATION AFTER TRAINING
    print("\n" + "="*60)
    print("🎯 NOW CLASSIFYING TEST IMAGE")
    print("="*60)
    
    # Recursive call - now load the saved model and classify
    # (This will execute the "model already exists" branch above)
    print("📂 Loading trained model for classification...")
    main(auto_mode=True, test_image=test_image if test_image else TEST_IMAGE_PATH)

if __name__ == "__main__":
    import sys
    
    # Check for command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--auto' or sys.argv[1] == '-a':
            # Auto mode: load model and classify default image
            test_image = sys.argv[2] if len(sys.argv) > 2 else None
            main(auto_mode=True, test_image=test_image)
        elif sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("\n🛰️ Satellite Image Classifier - Usage:")
            print("="*60)
            print("Interactive mode (default):")
            print("  python satellite_classifier_hierarchical.py")
            print("\nAuto mode (no prompts):")
            print("  python satellite_classifier_hierarchical.py --auto [image_path]")
            print("  python satellite_classifier_hierarchical.py -a [image_path]")
            print("\nExamples:")
            print("  python satellite_classifier_hierarchical.py --auto")
            print("  python satellite_classifier_hierarchical.py --auto image2.jpg")
            print("="*60)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help or -h for usage information")
    else:
        # Default: interactive mode
        main()
