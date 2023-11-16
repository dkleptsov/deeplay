import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import hdbscan
from sklearn.cluster import KMeans, OPTICS, SpectralClustering, DBSCAN, MeanShift
from sklearn.cluster import AgglomerativeClustering, AffinityPropagation, Birch
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score


CLUSTERING_TRAIN_PATH = 'data/task2_v4.csv'
PREDICTIONS_FOLDER = 'answer/'
NUMBER_OF_CLUSTERS = 8
KMEANS_PARAM_GRID = {'n_clusters': [NUMBER_OF_CLUSTERS],
                     'init': ['k-means++', 'random'],
                     'n_init': [10, 20, 30, 40, 50],
                     'max_iter': [100, 200, 300, 400, 500],
                     'tol': [0.0001, 0.001, 0.01, 0.1, 1],
                     'algorithm': ['elkan', 'lloyd'],
                     'random_state': [0]}
KMEANS_BEST_PARAMS = {'algorithm': 'elkan',
                      'init': 'k-means++',
                      'max_iter': 100,
                      'n_clusters': NUMBER_OF_CLUSTERS,
                      'n_init': 10,
                      'random_state': 0,
                      'tol': 0.01}
ALGORITHMS = [KMeans(**KMEANS_BEST_PARAMS),
              OPTICS(min_samples=10),
              hdbscan.HDBSCAN(min_cluster_size=10), #, gen_min_span_tree=True),
              DBSCAN(min_samples=10),
              SpectralClustering(n_clusters=NUMBER_OF_CLUSTERS, assign_labels='discretize'),
              GaussianMixture(n_components=NUMBER_OF_CLUSTERS),
              AgglomerativeClustering(n_clusters=NUMBER_OF_CLUSTERS),
              AffinityPropagation(damping=0.9, preference=-200, max_iter=250, convergence_iter=15),
              Birch(n_clusters=NUMBER_OF_CLUSTERS)]
BEST_ALGORITHM = KMeans(**KMEANS_BEST_PARAMS)


def explore_data(data:pd.DataFrame) -> None:
    """ Проведем исследовательский анализ данных """
    print(f"Размерность данных:\n{data.shape}")
    print(f"\nТипы данных:\n{data.dtypes}")
    print(f"\nОписательная статистика:\n{data.describe()}")
    print(f"\nПервые строчки:\n{data.head()}")
    print(f"\nКоличество уникальных значений:\n{data.nunique()}")
    print(f"\nКоличество пропущенных значений:\n{data.isnull().sum()}")


def find_cluster_number_elbow(data:pd.DataFrame, max_clusters:int=10) -> None:
    """ Найдем оптимальное количество кластеров с помощью Elbow Method """
    wcss = []
    for n_clusters in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto",random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    if len(wcss) > 0:
    # Рассчитаем оптимальное количество кластеров
        diff = [wcss[i] - wcss[i-1] for i in range(1, len(wcss))]
        optimal_clusters = diff.index(max(diff))
        print(f"Оптимальное количество кластеров Elbow : {optimal_clusters}")

    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Количество кластеров')
    plt.ylabel('WCSS')
    plt.savefig(f"{PREDICTIONS_FOLDER}task2 оптимальное количество кластеров.png")
    plt.show()


def find_optimal_clusters_silhouette(data:pd.DataFrame, max_clusters:int=10) -> int:
    """ Найдем оптимальное количество кластеров с помощью Silhouette Method """
    scores = []
    for n_clusters in range(2, max_clusters+1):
        clusterer = KMeans(n_clusters=n_clusters, n_init="auto",random_state=0)
        preds = clusterer.fit_predict(data)
        score = silhouette_score(data, preds)
        scores.append(score)

    optimal_n_clusters = np.argmax(scores) + 2
    print(f"Оптимальное количество кластеров Silhouette : {optimal_n_clusters}")
    return optimal_n_clusters


def find_best_clustering_params(data:pd.DataFrame, param_grid:dict, algorithm)-> dict:
    """ Подбираем параметры для кластеризации c помощью GridSearch и кроссвалидации """
    grid_search = GridSearchCV(algorithm,
                               param_grid=param_grid,
                               cv=5,
                               n_jobs=-1)
    grid_search.fit(data)
    print(f"The best parameters for {algorithm.__class__.__name__}:\
                                    \n{grid_search.best_params_}\n")
    return grid_search.best_params_


def cluster_data(data:pd.DataFrame, algorithm) -> tuple:
    """ Кластеризуем данные с помощью заданного алгоритма"""
    clustering = algorithm.fit_predict(data)
    return clustering, algorithm.__class__.__name__


def reduce_dimensionality_tsne(X:pd.DataFrame) -> np.ndarray:
    """Уменьшим размерность с помощью метода t-SNE для визуализации результа """
    tsne = TSNE(n_components=2)
    X_reduced = tsne.fit_transform(X)
    return X_reduced


def reduce_dimensionality_pca(X:pd.DataFrame) -> np.ndarray:
    """ Уменьшим размерность с помощью метода PCA для визуализации результа """
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    return X_reduced


def visualize_results(data:pd.DataFrame, clustering:np.ndarray, title:str) -> None:
    """ Визуализируем результат кластеризации """
    data_2d = reduce_dimensionality_tsne(data)
    plt.figure(figsize=(8, 6))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clustering, s=50,cmap='viridis')
    plt.title(title)
    plt.savefig(f"{PREDICTIONS_FOLDER}task 2 {title}.png")
    plt.show()


def make_barplot(results:dict, title:str) -> None:
    """ Построим barplot c результата кластеризации """
    df = pd.Series(results).sort_values(ascending=True, inplace=False)
    plt.figure(figsize=(12, 6))
    plt.barh(df.index, df.values, align='center', alpha=0.8, left=0.17)
    plt.title(title)
    plt.savefig(f"{PREDICTIONS_FOLDER}task 2 {title}.png")
    plt.show()


def save_predictions(predictions:np.ndarray) -> None:
    """ Сохраняем предсказания в файл """
    df = pd.DataFrame(predictions, columns=['cluster'])
    df.to_csv(f"{PREDICTIONS_FOLDER}task2_predictions.csv", index=False)


def main():
    # Загрузим данные
    data = pd.read_csv(CLUSTERING_TRAIN_PATH, sep=";")

    # Проведем исследовательский анализ данных
    explore_data(data)
    """
    Основные выводы:
    - данные не содержат пропусков.
    - все признаки числовые.
    - признаки имеют разный масштаб, поэтому их нужно нормализовать.
    """

    # Нормализуем фичи
    scaled_data = StandardScaler().fit_transform(data)

    # Найдем оптимальное количество кластеров двумя способами 
    find_cluster_number_elbow(scaled_data)
    find_optimal_clusters_silhouette(scaled_data)

    # Оценим работы разных алгоритмов кластеризации графически.
    # И при помощи метрик Silhouette, Davies-Bouldin и Calinski-Harabasz
    silhouette_scores, davies_bouldin_scores, calinski_harabasz_scores={},{},{}
    for algorithm in ALGORITHMS:
        clustering, title = cluster_data(scaled_data, algorithm)
        visualize_results(scaled_data, clustering, title)
        silhouette_scores[title] = silhouette_score(scaled_data, clustering)
        davies_bouldin_scores[title] = davies_bouldin_score(scaled_data, clustering)
        calinski_harabasz_scores[title] = calinski_harabasz_score(scaled_data, clustering)

    make_barplot(silhouette_scores, "Silhouette scores")
    make_barplot(davies_bouldin_scores, "Davies-Bouldin scores")
    make_barplot(calinski_harabasz_scores, "Calinski-Harabasz scores")

    # Подберем параметры для алгоритма с помощью GridSearch и кроссвалидации
    find_best_clustering_params(scaled_data, KMEANS_PARAM_GRID, KMeans())

    # Кластеризуем данные с помощью лучшего алгоритма и сохраняем результаты
    clustering, title = cluster_data(scaled_data, BEST_ALGORITHM)
    save_predictions(clustering)


if __name__ == "__main__":
    main()