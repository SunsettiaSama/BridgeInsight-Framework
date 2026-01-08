import os
import random
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import psutil
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import DictionaryLearning, SparseCoder, PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def calculate_accuracy(y_true, y_pred):
    """Calculate clustering accuracy using Hungarian algorithm."""
    w = np.zeros((y_pred.max() + 1, y_true.max() + 1))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return w[ind].sum() / y_pred.size


def create_matrix(X, f, gamma, max_value=1e10):
    """Create Similarity matrix based on a distance matrix and pheromone matrix."""

    # Calculate the distance matrix (Euclidean distances)
    distance_matrix = cdist(X, X, 'euclidean')

    # Apply the transformation
    Transformed_matrix = np.exp(-distance_matrix / np.abs(f))

    # Handle potential overflow by clipping large values
    Transformed_matrix = np.clip(Transformed_matrix, None, max_value)

    # Handle NaNs or Infinities by replacing them with a large number or the mean value
    if np.any(np.isnan(Transformed_matrix)) or np.any(np.isinf(Transformed_matrix)):
        mean_value = np.nanmean(Transformed_matrix[np.isfinite(Transformed_matrix)])
        Transformed_matrix = np.nan_to_num(Transformed_matrix, nan=mean_value, posinf=max_value, neginf=max_value)

    # Ensure symmetry by averaging the matrix with its transpose
    return (Transformed_matrix + Transformed_matrix.T) / 2

def fitness_sparse(X, cluster_labels, dictionary):
    """Calculate fitness based on sparse representation."""
    coder = SparseCoder(dictionary=dictionary, transform_algorithm='lasso_lars')
    sparse_codes = coder.transform(X)

    # Compute within-cluster variance
    within_cluster_variance = sum(
        np.sum((sparse_codes[cluster_labels == cluster] - sparse_codes[cluster_labels == cluster].mean(axis=0)) ** 2)
        for cluster in np.unique(cluster_labels)
    )
    return -within_cluster_variance  # Negate to maximize fitness

def evaluate_clustering(X, labels):
    """Evaluate clustering results using various metrics."""
    return {
        'Silhouette': round(silhouette_score(X, labels),4),
        'DBI': round(davies_bouldin_score(X, labels),4),
        'CH Index': round(calinski_harabasz_score(X, labels),4),
    }

from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import DictionaryLearning, SparseCoder
from scipy.spatial.distance import cdist
import numpy as np
from collections import defaultdict

def adaptive_spectral_cluster(
    objects,
    k,
    feature_extractor=None,
    max_value=1e10,
    random_state=42
):
    """
    自适应谱聚类函数，将对象列表按簇分组，并返回聚类结果及对应索引。

    参数:
    - objects: 列表，每个元素为待聚类的对象。
    - k: 目标簇数。
    - feature_extractor: 可选函数，用于将对象映射为特征向量。
    - max_value: 用于防止数值溢出的阈值。
    - random_state: 随机种子。

    返回:
    - clusters: 列表，长度为k，每个元素为对应簇的原始对象列表。
    - cluster_indices: 列表，长度为k，每个元素为对应簇的原始对象索引列表。
    """
    # 1. 特征提取
    if feature_extractor is None:
        X = np.array([obj for obj in objects])
    else:
        X = np.array([feature_extractor(obj) for obj in objects])
    
    # 2. 数据标准化
    X_scaled = StandardScaler().fit_transform(X)
    
    # 3. 构建相似度矩阵（自适应参数 f）
    def create_similarity_matrix(X, f, gamma):
        distance_matrix = cdist(X, X, 'euclidean')
        transformed = np.exp(-distance_matrix / np.abs(f))
        transformed = np.clip(transformed, None, max_value)
        return (transformed + transformed.T) / 2  # 对称化
    
    # 4. 稀疏表示适应度函数
    def compute_fitness(X, labels, dictionary):
        coder = SparseCoder(dictionary=dictionary, transform_algorithm='lasso_lars')
        sparse_codes = coder.transform(X)
        within_variance = sum(
            np.sum((sparse_codes[labels == cluster] - sparse_codes[labels == cluster].mean(axis=0)) ** 2)
            for cluster in np.unique(labels)
        )
        return -within_variance  # 返回负值以最大化适应度
    
    # 5. 字典学习
    dict_learner = DictionaryLearning(n_components=X_scaled.shape[1], random_state=random_state)
    dict_matrix = dict_learner.fit(X_scaled).components_
    
    # 6. 自适应优化流程
    gamma = 1.0 / X_scaled.shape[1]
    f0 = 1.0  # 初始参数
    similarity_initial = create_similarity_matrix(X_scaled, f0, gamma)
    
    # 初始聚类
    clustering_initial = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=random_state)
    labels_initial = clustering_initial.fit_predict(similarity_initial)
    
    # 计算自适应参数 f
    f = compute_fitness(X_scaled, labels_initial, dict_matrix)
    
    # 更新相似度矩阵并最终聚类
    similarity_final = create_similarity_matrix(X_scaled, f, gamma)
    clustering_final = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=random_state)
    labels_final = clustering_final.fit_predict(similarity_final)
    
    # 7. 按标签分组对象及索引
    clusters = defaultdict(list)
    cluster_indices = defaultdict(list)
    for idx, (obj, label) in enumerate(zip(objects, labels_final)):
        clusters[label].append(obj)
        cluster_indices[label].append(idx)
    
    # 按标签排序返回结果
    clusters_sorted = [clusters[i] for i in sorted(clusters.keys())]
    indices_sorted = [cluster_indices[i] for i in sorted(cluster_indices.keys())]
    
    return clusters_sorted, indices_sorted

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import calinski_harabasz_score
from sklearn.neighbors import kneighbors_graph
from scipy.optimize import minimize
from collections import defaultdict

def KNN_adaptive_spectral_cluster(
    objects,
    k,
    feature_extractor=None,
    max_value=1e10,
    random_state=42,
    n_neighbors=10,
    gamma_initial=1.0
):
    """
    优化后的自适应谱聚类方法，使用稀疏相似度矩阵和参数优化。
    """
    print('=' * 20)
    print('Start Clustering...')
    # 1. 特征提取
    if feature_extractor is None:
        X = np.array([obj for obj in objects])
    else:
        X = np.array([feature_extractor(obj) for obj in objects])
    
    # 2. 标准化
    X_scaled = StandardScaler().fit_transform(X)
    
    # 3. 构建稀疏相似度矩阵
    def create_similarity_matrix(X, f, gamma, n_neighbors=n_neighbors):
        distances = kneighbors_graph(X, n_neighbors, mode='distance', include_self=False)
        similarities = np.exp(- (distances.data ** 2) / (2 * gamma ** 2))
        sparse_similarities = distances.copy()
        sparse_similarities.data = similarities
        sparse_similarities = sparse_similarities + sparse_similarities.T
        sparse_similarities.setdiag(0)
        sparse_similarities.data = np.clip(sparse_similarities.data, 0, max_value)
        return sparse_similarities

    # 4. 适应度函数
    def compute_fitness(X, labels):
        return calinski_harabasz_score(X, labels)

    # 5. 参数优化
    def objective(params):
        f, gamma = params
        similarity_matrix = create_similarity_matrix(X_scaled, f, gamma)
        clustering = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=random_state)
        labels = clustering.fit_predict(similarity_matrix)
        fitness = compute_fitness(X_scaled, labels)
        return -fitness

    # 初始参数
    initial_params = [1.0, gamma_initial]
    bounds = [(0.1, 10.0), (0.1, 10.0)]  # f 和 gamma 的范围

    print('=' * 20)
    print('Optimize Parameters...')

    # 执行优化
    result = minimize(objective, x0=initial_params, bounds=bounds, method='L-BFGS-B')
    optimal_f, optimal_gamma = result.x

    print('=' * 20)
    print('Final Clustering...')

    # 6. 使用最优参数进行最终聚类
    similarity_final = create_similarity_matrix(X_scaled, optimal_f, optimal_gamma)
    clustering_final = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=random_state)
    labels_final = clustering_final.fit_predict(similarity_final)

    print('=' * 20)
    print('Split Clusters...')
    # 7. 分组结果
    clusters = defaultdict(list)
    cluster_indices = defaultdict(list)
    for idx, (obj, label) in enumerate(zip(objects, labels_final)):
        clusters[label].append(obj)
        cluster_indices[label].append(idx)

    # 按标签排序返回
    clusters_sorted = [clusters[i] for i in sorted(clusters.keys())]
    indices_sorted = [cluster_indices[i] for i in sorted(cluster_indices.keys())]
    
    return clusters_sorted, indices_sorted








def main():
    # 读取数据
    df = pd.read_excel('molecule.xlsx')
    cid = df['CID']
    X = df.iloc[:, 1:]
    print(X)
    print(X.shape)
    print(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")
    # Standardize data
    X_scaled = StandardScaler().fit_transform(X)
    gamma = 1.0 / X_scaled.shape[1]
    start_time = time.time()
    # Initial Transformed_matrix
    f0 = 1
    Similarity_matrix_initial = create_matrix(X_scaled, f0, gamma)
    print(np.isinf(Similarity_matrix_initial))
    # Dictionary learning for sparse coding
    dictionary_learner = DictionaryLearning(n_components=X_scaled.shape[1], random_state=SEED)
    dictionary_matrix = dictionary_learner.fit(X_scaled).components_
    clustering_initial = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=SEED)
    cluster_labels_initial = clustering_initial.fit_predict(Similarity_matrix_initial)
    f = fitness_sparse(X_scaled, cluster_labels_initial, dictionary_matrix)
    Similarity_matrix = create_matrix(X_scaled, f, gamma)
    # Clustering after optimization
    clustering_after = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=SEED)
    labels_after = clustering_after.fit_predict(Similarity_matrix)
    runtime_after = time.time() - start_time
    print("\nAfter Optimization:")
    print("Cluster distribution:", Counter(labels_after))
    # Evaluate after optimization
    metrics_after = evaluate_clustering(Similarity_matrix, labels_after)
    print(f"Metrics: {metrics_after}, Runtime: {runtime_after:.4f} seconds")
    df['Cluster'] = labels_after
    output_file = 'outputClusteringLabel.xlsx'
    df.to_excel(output_file, index=False)
    print(f"Clustering results saved to {output_file}")
    # PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(X_scaled)

    # cluster_colors = {0: '#7F9FAF', 1: '#E2A2AC', 2 : "#284852"}
    cluster_colors = {0: '#F2B2AC', 1: '#8FC4D9', 2: "#CEB4A5"}
    highlight_color = 'red'  # Highlight CID 7005
    highlight_index = df[df['CID'] == 7005].index
    plt.rc('font', family='Times New Roman')
    plt.figure(figsize=(12, 8))
    for i in range(len(X_scaled)):
        if i not in highlight_index:
            color = cluster_colors[labels_after[i]]
            plt.scatter(data_pca[i, 0], data_pca[i, 1], color=color, s=50, alpha=0.7)
    for i in highlight_index:
        plt.scatter(data_pca[i, 0], data_pca[i, 1], color=highlight_color, s=150, alpha=1.0, marker='*',
                    edgecolors=highlight_color, linewidth=1.5, label='Target Molecule')
    plt.xlabel('PCA Component 1', fontsize=14)
    plt.ylabel('PCA Component 2', fontsize=14)
    plt.legend(loc='upper right', fontsize=11)
    plt.savefig('ASC_MS.png', dpi=600)
    plt.show()


