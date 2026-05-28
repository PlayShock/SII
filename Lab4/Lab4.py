import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Perceptron
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

print("=" * 60)
print("ЗАДАЧА КЛАССИФИКАЦИИ")
print("=" * 60)

train_data = pd.read_csv('disease_train.csv')
test_public = pd.read_csv('disease_public_test.csv')
sample_submission = pd.read_csv('disease_sample_submission.csv')

print("\nРазмер обучающей выборки:", train_data.shape)
print("Размер тестовой выборки:", test_public.shape)
print("\nПервые 5 строк обучающей выборки:")
print(train_data.head())

target = 'Y'
features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']

print("\n" + "=" * 60)
print("АНАЛИЗ ТИПОВ ДАННЫХ")
print("=" * 60)
print("\nТипы данных в обучающей выборке:")
print(train_data[features].dtypes)

print("\nКоличество пропущенных значений:")
print(train_data[features].isnull().sum())

print("\nСтатистика признаков:")
print(train_data[features].describe())

print("\nРаспределение классов в обучающей выборке:")
print(train_data[target].value_counts())
print(f"Процент больных: {train_data[target].mean() * 100:.2f}%")

X_train_raw = train_data[features].copy()
y_train = train_data[target].copy()

X_test = test_public[features].copy()
y_test = sample_submission[target].values

print("\n" + "=" * 60)
print("ПОДГОТОВКА ДАННЫХ")
print("=" * 60)
print(f"Размер обучающей выборки: {X_train_raw.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

print("\n" + "=" * 60)
print("ОБУЧЕНИЕ НА СЫРЫХ ДАННЫХ")
print("=" * 60)

base_clf = DecisionTreeClassifier(max_depth=5, random_state=42)

bagging_raw = BaggingClassifier(
    estimator=base_clf,
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

perceptron_raw = Perceptron(random_state=42, max_iter=1000, tol=1e-3)

bagging_raw.fit(X_train_raw, y_train)
y_pred_bagging_raw = bagging_raw.predict(X_test)
acc_bagging_raw = accuracy_score(y_test, y_pred_bagging_raw)

perceptron_raw.fit(X_train_raw, y_train)
y_pred_perc_raw = perceptron_raw.predict(X_test)
acc_perc_raw = accuracy_score(y_test, y_pred_perc_raw)

print(f"\nТочность на сырых данных:")
print(f"BaggingClassifier (сырые):     {acc_bagging_raw:.4f}")
print(f"Perceptron (сырые):            {acc_perc_raw:.4f}")

print("\n" + "=" * 60)
print("ПРЕДОБРАБОТКА ДАННЫХ")
print("=" * 60)

scaler_bagging = MinMaxScaler()
X_train_bagging = scaler_bagging.fit_transform(X_train_raw)
X_test_bagging = scaler_bagging.transform(X_test)

scaler_perc = StandardScaler()
X_train_perc = scaler_perc.fit_transform(X_train_raw)
X_test_perc = scaler_perc.transform(X_test)

print("Примененные преобразования:")
print("- BaggingClassifier: MinMaxScaler (нормализация в диапазон [0,1])")
print("- Perceptron: StandardScaler (стандартизация с нулевым средним)")
print(f"\nСтатистика после преобразования (Bagging):")
print(f"  Min: {X_train_bagging.min():.3f}, Max: {X_train_bagging.max():.3f}")
print(f"Статистика после преобразования (Perceptron):")
print(f"  Mean: {X_train_perc.mean():.3f}, Std: {X_train_perc.std():.3f}")

print("\n" + "=" * 60)
print("ОБУЧЕНИЕ НА ОБРАБОТАННЫХ ДАННЫХ")
print("=" * 60)

bagging_processed = BaggingClassifier(
    estimator=base_clf,
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)
bagging_processed.fit(X_train_bagging, y_train)
y_pred_bagging = bagging_processed.predict(X_test_bagging)
acc_bagging = accuracy_score(y_test, y_pred_bagging)

perceptron_processed = Perceptron(random_state=42, max_iter=1000, tol=1e-3)
perceptron_processed.fit(X_train_perc, y_train)
y_pred_perc = perceptron_processed.predict(X_test_perc)
acc_perc = accuracy_score(y_test, y_pred_perc)

print(f"\nТочность на обработанных данных:")
print(f"BaggingClassifier (обраб.):     {acc_bagging:.4f}")
print(f"Perceptron (обраб.):            {acc_perc:.4f}")

print("\n" + "=" * 60)
print("КРОСС-ВАЛИДАЦИЯ (5-кратная)")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores_bagging = cross_val_score(bagging_processed, X_train_bagging, y_train, cv=skf)
cv_scores_perc = cross_val_score(perceptron_processed, X_train_perc, y_train, cv=skf)

print(f"\nBaggingClassifier:")
print(f"  Средняя точность: {cv_scores_bagging.mean():.4f}")
print(f"  Стандартное отклонение: {cv_scores_bagging.std():.4f}")
print(f"  Оценки по folds: {cv_scores_bagging}")

print(f"\nPerceptron:")
print(f"  Средняя точность: {cv_scores_perc.mean():.4f}")
print(f"  Стандартное отклонение: {cv_scores_perc.std():.4f}")
print(f"  Оценки по folds: {cv_scores_perc}")

print("\n" + "=" * 60)
print("ДЕТАЛЬНЫЙ ОТЧЕТ О КЛАССИФИКАЦИИ")
print("=" * 60)

print("\nClassification Report для BaggingClassifier:")
print(classification_report(y_test, y_pred_bagging, target_names=['Здоров (0)', 'Болен (1)']))

print("\nClassification Report для Perceptron:")
print(classification_report(y_test, y_pred_perc, target_names=['Здоров (0)', 'Болен (1)']))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_bagging, ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('BaggingClassifier (обработанные данные)', fontsize=12)
axes[0].set_xlabel('Предсказанный класс')
axes[0].set_ylabel('Истинный класс')

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_perc, ax=axes[1], colorbar=False, cmap='Oranges')
axes[1].set_title('Perceptron (обработанные данные)', fontsize=12)
axes[1].set_xlabel('Предсказанный класс')
axes[1].set_ylabel('Истинный класс')

plt.tight_layout()
plt.show()

methods = ['Bagging (сырые)', 'Perceptron (сырые)', 'Bagging (обраб.)', 'Perceptron (обраб.)']
accuracies = [acc_bagging_raw, acc_perc_raw, acc_bagging, acc_perc]

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, accuracies, color=['#2E86AB', '#A23B72', '#2E86AB', '#A23B72'])
plt.ylabel('Точность', fontsize=12)
plt.title('Сравнение точности классификаторов до и после предобработки', fontsize=14)
plt.ylim(0, 1)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Случайное угадывание')

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=11)

plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
data_to_plot = [cv_scores_bagging, cv_scores_perc]
bp = plt.boxplot(data_to_plot, tick_labels=['BaggingClassifier', 'Perceptron'],
                  patch_artist=True, boxprops=dict(facecolor='#2E86AB'),
                  medianprops=dict(color='#F18F01', linewidth=2))
plt.ylabel('Точность', fontsize=12)
plt.title('Распределение точности при 5-кратной кросс-валидации', fontsize=14)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Случайное угадывание')
plt.grid(axis='y', linestyle='--', alpha=0.3)

for i, scores in enumerate([cv_scores_bagging, cv_scores_perc], 1):
    median_val = np.median(scores)
    plt.text(i, median_val + 0.02, f'медиана={median_val:.3f}',
             ha='center', va='bottom', fontsize=9, style='italic')

plt.legend()
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("ЗАДАЧА КЛАСТЕРИЗАЦИИ")
print("=" * 60)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_train_perc)

ari = adjusted_rand_score(y_train, clusters)
silhouette_avg = silhouette_score(X_train_perc, clusters)

print(f"\nОценка качества кластеризации KMeans (k=2):")
print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
print(f"  Silhouette Score: {silhouette_avg:.4f}")
print(f"\nРаспределение кластеров:")
print(pd.Series(clusters).value_counts())

print(f"\nСоответствие кластеров истинным меткам:")
cross_tab = pd.crosstab(y_train, clusters, rownames=['Истинный класс'], colnames=['Кластер'])
print(cross_tab)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_train_perc)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='plasma',
                            edgecolor='k', alpha=0.7)
axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='lime', marker='X', s=200, edgecolor='black', linewidth=2, label='Центры кластеров')
axes[0].set_xlabel('Главная компонента 1', fontsize=11)
axes[0].set_ylabel('Главная компонента 2', fontsize=11)
axes[0].set_title('Кластеры KMeans (k=2)', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='RdYlGn',
                            edgecolor='k', alpha=0.7)
axes[1].set_xlabel('Главная компонента 1', fontsize=11)
axes[1].set_ylabel('Главная компонента 2', fontsize=11)
axes[1].set_title('Истинные метки (0 - здоров, 1 - болен)', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cluster_counts = pd.Series(clusters).value_counts().sort_index()
axes[0].bar(['Кластер 0', 'Кластер 1'], cluster_counts,
            color=['#1B998B', '#C73E1D'])
axes[0].set_ylabel('Количество образцов', fontsize=11)
axes[0].set_title('Распределение объектов по кластерам', fontsize=12)
axes[0].grid(axis='y', linestyle='--', alpha=0.3)

metrics = ['Adjusted Rand Index', 'Silhouette Score']
values = [ari, silhouette_avg]
axes[1].bar(metrics, values, color=['#3D5A80', '#EE6C4D'])
axes[1].set_ylabel('Значение метрики', fontsize=11)
axes[1].set_title('Оценка качества кластеризации', fontsize=12)
axes[1].set_ylim(0, 1)
for i, v in enumerate(values):
    axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
axes[1].grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 60)

results_df = pd.DataFrame({
    'Модель': ['BaggingClassifier', 'BaggingClassifier', 'Perceptron', 'Perceptron'],
    'Тип данных': ['Сырые', 'Обработанные', 'Сырые', 'Обработанные'],
    'Точность на тесте': [acc_bagging_raw, acc_bagging, acc_perc_raw, acc_perc],
    'CV средняя точность': [cv_scores_bagging.mean(), cv_scores_bagging.mean(),
                            cv_scores_perc.mean(), cv_scores_perc.mean()],
    'CV std': [cv_scores_bagging.std(), cv_scores_bagging.std(),
               cv_scores_perc.std(), cv_scores_perc.std()]
})

print("\n", results_df.to_string(index=False))

clustering_results = pd.DataFrame({
    'Метод': ['KMeans (k=2)'],
    'Adjusted Rand Index': [ari],
    'Silhouette Score': [silhouette_avg],
    'Количество кластеров': [2],
    'Количество итераций': [10]
})

print("\nРезультаты кластеризации:")
print(clustering_results.to_string(index=False))