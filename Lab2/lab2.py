import numpy as np
import matplotlib.pyplot as plt
import time
import random
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage


def generate_random_cities(n, x_range=(0, 100), y_range=(0, 100)):
    """Генерирует n случайных городов в заданных пределах."""
    return [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(n)]

def distance(p1, p2):
    """Евклидово расстояние между двумя точками."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def compute_wcss(cities, labels, centroids):
    """Сумма квадратов расстояний от точек до центроидов их кластеров."""
    wcss = 0.0
    cities = np.array(cities)
    for i, city in enumerate(cities):
        centroid = centroids[labels[i]]
        wcss += distance(city, centroid) ** 2
    return wcss

def plot_dendrogram(cities, title="Дендрограмма иерархической кластеризации"):
    """Строит дендрограмму для иерархической кластеризации."""
    cities_array = np.array(cities)
    linkage_matrix = linkage(cities_array, method='ward')
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix)
    plt.title(title)
    plt.xlabel("Индекс города")
    plt.ylabel("Расстояние")
    plt.grid(True)
    plt.show()


def kmeans_clustering(cities, K, max_iters=100, tol=1e-4):
    """Классический K-средних со случайной инициализацией (ручная реализация)."""
    if K > len(cities):
        raise ValueError("K не может быть больше количества городов")
    centroids = np.array(random.sample(cities, K))
    cities = np.array(cities)
    labels = np.zeros(len(cities), dtype=int)
    iterations = 0
    for _ in range(max_iters):
        iterations += 1
        for i, city in enumerate(cities):
            dists = [distance(city, c) for c in centroids]
            labels[i] = np.argmin(dists)
        new_centroids = []
        for k in range(K):
            cluster_points = cities[labels == k]
            if len(cluster_points) > 0:
                new_centroids.append(np.mean(cluster_points, axis=0))
            else:
                new_centroids.append(centroids[k])
        new_centroids = np.array(new_centroids)
        if np.allclose(new_centroids, centroids, atol=tol):
            break
        centroids = new_centroids
    return labels, centroids, iterations

def hierarchical_clustering(cities, K):
    """Иерархическая (агломеративная) кластеризация с использованием sklearn."""
    cities_array = np.array(cities)
    clustering = AgglomerativeClustering(n_clusters=K, linkage='ward')
    labels = clustering.fit_predict(cities_array)
    centroids = []
    for i in range(K):
        cluster_points = cities_array[labels == i]
        if len(cluster_points) > 0:
            centroids.append(np.mean(cluster_points, axis=0))
        else:
            centroids.append([0, 0])
    return labels, np.array(centroids), None

def sklearn_kmeans_clustering(cities, K):
    """Использует реализацию KMeans из scikit-learn."""
    cities_array = np.array(cities)
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(cities_array)
    return kmeans.labels_, kmeans.cluster_centers_, kmeans.n_iter_

def run_test(cities, K, test_name):
    print(f"\n{'=' * 70}")
    print(f"{test_name}")
    print(f"Количество городов: {len(cities)}, K = {K}")
    print('=' * 70)

    methods = [
        ("Ручной K-средних (случ. иниц.)", kmeans_clustering),
        ("Sklearn KMeans (k-means++)", sklearn_kmeans_clustering),
        ("Иерархическая (агломеративная)", hierarchical_clustering)
    ]


    results = []
    for name, func in methods:
        start = time.perf_counter()
        try:
            if name == "Иерархическая (агломеративная)":
                labels, centroids, _ = func(cities, K)
                iterations = "—"
            else:
                labels, centroids, iterations = func(cities, K)
            elapsed = time.perf_counter() - start
            wcss = compute_wcss(cities, labels, centroids) if len(labels) > 0 else 0
            results.append((name, elapsed, wcss, iterations))
        except Exception as e:
            print(f"Ошибка в методе {name}: {e}")
            results.append((name, 0, 0, "ошибка"))

    print("\n{:<35} {:>12} {:>12} {:>15}".format("Метод", "Время (с)", "WCSS", "Итерации"))
    print("-" * 75)
    for name, t, wcss, it in results:
        print("{:<35} {:>12.6f} {:>12.2f} {:>15}".format(name, t, wcss, it))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(test_name, fontsize=16)

    cmap = plt.cm.tab10
    colors = [cmap(i / K) for i in range(K)] if K > 0 else []

    for ax, (name, elapsed, wcss, it) in zip(axes, results):
        if name == "Ручной K-средних (случ. иниц.)":
            labels, centroids, _ = kmeans_clustering(cities, K)
        elif name == "Sklearn KMeans (k-means++)":
            labels, centroids, _ = sklearn_kmeans_clustering(cities, K)
        else:
            labels, centroids, _ = hierarchical_clustering(cities, K)

        ax.set_title(f"{name}\nВремя: {elapsed:.4f} с, WCSS: {wcss:.2f}")
        for i in range(K):
            pts = [cities[j] for j in range(len(cities)) if labels[j] == i]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, color=colors[i % len(colors)], label=f'Кластер {i+1}', alpha=0.6)
        if len(centroids) > 0:
            ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Центроиды')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()

    plt.tight_layout()
    plt.show()

def find_optimal_k(cities, max_K=10):
    """Определяет оптимальное число кластеров методом локтя (максимальное расстояние до прямой)."""
    wcss = []
    K_range = range(1, max_K + 1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(np.array(cities))
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, wcss, 'bo-')
    plt.xlabel('Количество кластеров K')
    plt.ylabel('WCSS')
    plt.title('Метод локтя для выбора K')
    plt.grid(True)
    plt.show()

    first_point = (1, wcss[0])
    last_point = (max_K, wcss[-1])
    max_dist = -1
    optimal_k = 1
    for i, k in enumerate(K_range, start=1):
        x0, y0 = k, wcss[i-1]
        x1, y1 = first_point
        x2, y2 = last_point
        area = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
        base = np.hypot(x2 - x1, y2 - y1)
        dist = area / base if base != 0 else 0
        if dist > max_dist:
            max_dist = dist
            optimal_k = k
    return optimal_k


def main():
    tests = [
        {
            'name': 'Тест 1: Случайные города',
            'cities': generate_random_cities(int(input("Количество городов N = "))),
            'K': int(input("Количество кластеров K = "))
        },
        {
            'name': 'Тест 2: Три разнесённых кластера',
            'cities': [
                (0, 0), (1, 0), (2, 0),
                (5, 5), (6, 5), (5, 6),
                (10, 10), (11, 10), (10, 11)
            ],
            'K': 3
        },
        {
            'name': 'Тест 3: Один кластер (все точки рядом)',
            'cities': [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
            'K': 1
        },
        {
            'name': 'Тест 4 (белый ящик): Все точки одинаковы',
            'cities': [(5, 5)] * 10,
            'K': 2
        },
        {
            'name': 'Тест 5 (белый ящик): Пустой кластер после инициализации',
            'cities': [(0, 0), (1, 0), (0, 1), (1, 1), (10, 10)],
            'K': 3
        },
        {
            'name': 'Тест 6 (чёрный ящик): Пересекающиеся кластеры',
            'cities': [
                (2, 2), (3, 2), (2, 3), (3, 3),
                (2.5, 2.5), (3.5, 2.5), (2.5, 3.5), (3.5, 3.5)
            ],
            'K': 2
        },
        {
            'name': 'Тест 7 (чёрный ящик): Кластеры разной плотности',
            'cities': [
                (0, 0), (0.1, 0), (0, 0.1), (0.1, 0.1), (0.2, 0),
                (0, 0.2), (0.2, 0.2), (0.3, 0.1), (0.1, 0.3), (0.3, 0.3),
                (8, 8), (9, 9), (8, 9), (9, 8)
            ],
            'K': 2
        },
        {
            'name': 'Тест 8 (чёрный ящик): Кластеры + шум',
            'cities': [
                (0, 0), (1, 0), (0, 1), (1, 1),
                (5, 5), (6, 5), (5, 6), (6, 6),
                (2, 2), (3, 4), (4, 2), (3, 3), (7, 2), (8, 3)
            ],
            'K': 2
        },
        {
            'name': 'Тест 9 (чёрный ящик): Один выброс',
            'cities': [(0, 0)] * 8 + [(100, 100)],
            'K': 2
        },
        {
            'name': 'Тест 10 (для метода локтя): Три сгенерированных кластера',
            'cities': [tuple(pt) for pt in make_blobs(n_samples=100, centers=3, cluster_std=1.5, random_state=42)[0]],
            'K': 3
        }
    ]

    for test in tests:
        run_test(test['cities'], test['K'], test['name'])

    print("\nДемонстрация метода локтя для набора 'Тест 10'")
    optimal_k = find_optimal_k(tests[-1]['cities'], max_K=10)
    print(f"Оптимальное число кластеров по методу локтя: {optimal_k}")

    print("\nДендрограмма для иерархической кластеризации (Тест 10)")
    plot_dendrogram(tests[-1]['cities'], title="Дендрограмма (Тест 10)")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
