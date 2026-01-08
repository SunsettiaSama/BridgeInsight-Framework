# Adaptive Spectral Clustering

This repository implements the **Adaptive Spectral Clustering** method proposed in the paper *"A Systematic Framework Enhancing Molecular Screening Efficiency in Drug Discovery via Scaffold-Driven Fuzzy Similarity and Adaptive Spectral Clustering"*. The approach introduces an adaptive Gaussian kernel to dynamically adjust similarity metrics, improving clustering performance on heterogeneous datasets.

---

## Method Overview

### Key Contributions
1. **Adaptive Gaussian Kernel**: Dynamically adjusts bandwidth parameter (σ) using cluster variances
2. **Sparse Coding Optimization**: Integrates dictionary learning for enhanced feature representation
3. **Symmetrized Similarity Matrix**: Ensures robustness through symmetric matrix transformation

---

## Technical Details

### 1. Pairwise Similarity Calculation
**Gaussian Kernel with Adaptive Bandwidth**:
$S_{ij} = \exp\left(-\frac{\|X_i - X_j\|_2}{|f|}\right)$
- $X_i, X_j$: Input feature vectors
- $f$: Cluster variance-based scaling factor
- Symmetrization: $S = \frac{S + S^T}{2}$

### 2. Spectral Embedding
1. **Degree Matrix**: $D = \text{diag}(d_i), d_i = \sum_j W_{ij}$
2. **Laplacian Matrices**:
   - Unnormalized: $L = D - W$
   - Normalized: $L_{\text{sym}} = D^{-1/2} (D - W) D^{-1/2}$

### 3. Cluster Assignment
1. Extract top-$c$ eigenvectors from $L_{\text{sym}}$
2. Apply k-means clustering on spectral embedding matrix

---

## Algorithm Implementation

### Adaptive Spectral Clustering Steps
```python
def adaptive_spectral_clustering(X, clusters, seed=42):
    # Initialize parameters
    f = 1.0  # Variance scaling factor
    
    # Iterative optimization
    while not converged:
        # Step 1: Construct similarity matrix
        distances = euclidean_distances(X)
        S = exp(-distances / abs(f))
        S = (S + S.T) / 2
        
        # Step 2: Sparse representation learning
        D, A = dictionary_learning(X)
        
        # Step 3: Cluster variance calculation
        labels = spectral_clustering(S)
        f = compute_cluster_variance(A, labels)
        
        # Step 4: Update similarity matrix
        S = exp(-distances / abs(f))
        S = (S + S.T) / 2
    
    return final_labels
```
---

## Key Parameters

The following parameters control the behavior of the adaptive spectral clustering algorithm:

| Parameter         | Description                                                                 | Default  | Type    |
|-------------------|-----------------------------------------------------------------------------|----------|---------|
| `n_clusters`      | Target number of clusters to form                                           | Required | int     |
| `eigen_tol`             | Convergence tolerance for variance changes between iterations               | auto     | float   |
| `random_state`    | Seed for random number generation (controls spectral initialization)        | None     | int     |
| `affinity`    | Precomputed affinity matrix (shape=(n_samples, n_samples)) <br/>or distance metric identifier (e.g., 'euclidean', 'precomputed')  | 'rbf'      | {array-like, sparse matrix, str}   |

---

## Algorithm Workflow

### Iterative Optimization Process
```mermaid
flowchart TD
    A[Start] --> B[Initialize Parameters]
    B --> C[Build Initial Similarity Matrix and Initial Spectral Clustering]
    C --> D[Learn Dictionary Representation]
    D --> E{Converged?}
    E -->|No| D
    E --> |Yes|G[Calculate Cluster Variance and Build Final Similarity Matrix]
    G --> H[Final Spectral Clustering]
    H --> I[Output Labels]
```
## Environmental configuration
The experimental setup in this study is as follows: Core i7-9750H 2.60GHz CPU, NVIDIA GeForce GTX 1650 4GB GPU, 16GB RAM, 1TB SSD, and Windows 10 operating system. The compound data used in the experiments were obtained from the publicly available PubChem database, with 1-naphthol selected as the target molecule.
The Python compiler version we used is Python 3.8.
