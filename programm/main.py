import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
def load_data(file_path):
    """Загружает данные из CSV."""
    try:
        data = pd.read_csv(file_path)
        print(f"Файл '{file_path}' успешно загружен.")
        return data
    except FileNotFoundError:
        print(f"Ошибка: Файл '{file_path}' не найден. Проверьте путь.")
        exit()

# Предобработка данных
def preprocess_data(data):
    """Проверка пропусков, типов данных и удаление дубликатов."""
    # Проверка на пропуски
    if data.isnull().sum().sum() > 0:
        data = data.dropna()
    
    # Убедимся, что типы данных корректны
    data['user_id'] = data['user_id'].astype(str)
    data['test_group'] = data['test_group'].astype('category')
    data['converted'] = data['converted'].astype(bool)
    
    # Исключение пользователей, которые оказались в обеих группах
    duplicates = data.groupby('user_id')['test_group'].nunique()
    exclude_users = duplicates[duplicates > 1].index
    data = data[~data['user_id'].isin(exclude_users)]
    
    return data

# Расчет метрик
def calculate_metrics(data):
    """Рассчитывает основные метрики для каждой группы."""
    metrics = data.groupby('test_group').agg(
        total_users=('user_id', 'nunique'),
        total_converted=('converted', 'sum'),
        total_ads=('total_ads', 'sum')
    )
    
    # Конверсия и другие метрики
    metrics['conversion_rate'] = metrics['total_converted'] / metrics['total_users']
    metrics['avg_ads'] = metrics['total_ads'] / metrics['total_users']
    metrics['conversion_per_user'] = metrics['total_converted'] / metrics['total_users']  # Конверсии на пользователя
    metrics['conversion_per_ad'] = metrics['total_converted'] / metrics['total_ads']  # Конверсии на одну рекламу
    metrics['total_conversion_percentage'] = (metrics['total_converted'] / metrics['total_users']) * 100  # % конверсий
    metrics['users_per_ad'] = metrics['total_users'] / metrics['total_ads']  # Пользователи на одну рекламу
    metrics['active_user_ratio'] = metrics['total_converted'] / metrics['total_users']  # Доля активных пользователей
    
    return metrics

# Статистический анализ
def statistical_analysis(data):
    """Проводит статистический анализ различий между группами."""
    # Конверсия
    group_data = data.groupby('test_group')['converted']
    ad_converted = group_data.get_group('ad')
    psa_converted = group_data.get_group('psa')
    
    # Проверка нормальности
    ad_shapiro = shapiro(ad_converted)[1]
    psa_shapiro = shapiro(psa_converted)[1]
    
    # Тесты на равенство пропорций
    z_stat, p_value_conversion = proportions_ztest(
        [ad_converted.sum(), psa_converted.sum()],
        [len(ad_converted), len(psa_converted)]
    )
    
    # Количество рекламы
    ad_ads = data[data['test_group'] == 'ad']['total_ads']
    psa_ads = data[data['test_group'] == 'psa']['total_ads']
    
    if ad_shapiro > 0.05 and psa_shapiro > 0.05:
        # Нормальное распределение
        t_stat, p_value_ads = ttest_ind(ad_ads, psa_ads, equal_var=False)
    else:
        # Ненормальное распределение
        t_stat, p_value_ads = mannwhitneyu(ad_ads, psa_ads)
    
    # Доверительные интервалы
    ci_ad = proportion_confint(ad_converted.sum(), len(ad_converted), alpha=0.05, method='normal')
    ci_psa = proportion_confint(psa_converted.sum(), len(psa_converted), alpha=0.05, method='normal')
    
    return {
        "conversion_test": {"z_stat": z_stat, "p_value": p_value_conversion},
        "ads_test": {"t_stat": t_stat, "p_value": p_value_ads},
        "confidence_intervals": {
            "ad_conversion": ci_ad,
            "psa_conversion": ci_psa
        }
    }

# Формирование выводов
def generate_insights(metrics, analysis):
    """Генерация выводов и рекомендаций."""
    insights = []
    
    # Конверсия
    if analysis['conversion_test']['p_value'] < 0.05:
        insights.append("Конверсия в группах 'ad' и 'psa' статистически различается.")
    else:
        insights.append("Конверсии в группах 'ad' и 'psa' статистически не различаются.")
    
    # Количество рекламы
    if analysis['ads_test']['p_value'] < 0.05:
        insights.append("Среднее количество увиденной рекламы в группах 'ad' и 'psa' статистически различается.")
    else:
        insights.append("Среднее количество увиденной рекламы в группах 'ad' и 'psa' статистически не различается.")
    
    # Рекомендации
    if metrics.loc['ad', 'conversion_rate'] > metrics.loc['psa', 'conversion_rate']:
        insights.append("Реклама положительно влияет на конверсию. Рекомендуется продолжать кампанию.")
    else:
        insights.append("Реклама не оказывает значительного влияния на конверсию. Возможно, стоит пересмотреть стратегию.")
    
    return insights

# Визуализация
def plot_metrics(metrics):
    """Графическое представление метрик."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    
    sns.barplot(x=metrics.index, y='conversion_rate', data=metrics, ax=axes[0, 0])
    axes[0, 0].set_title('Конверсия по группам')
    
    sns.barplot(x=metrics.index, y='avg_ads', data=metrics, ax=axes[0, 1])
    axes[0, 1].set_title('Среднее количество рекламы')
    
    sns.barplot(x=metrics.index, y='conversion_per_user', data=metrics, ax=axes[1, 0])
    axes[1, 0].set_title('Конверсии на пользователя')
    
    sns.barplot(x=metrics.index, y='conversion_per_ad', data=metrics, ax=axes[1, 1])
    axes[1, 1].set_title('Конверсии на одну рекламу')
    
    sns.barplot(x=metrics.index, y='total_conversion_percentage', data=metrics, ax=axes[2, 0])
    axes[2, 0].set_title('% Конверсий на группу')
    
    sns.barplot(x=metrics.index, y='users_per_ad', data=metrics, ax=axes[2, 1])
    axes[2, 1].set_title('Пользователи на одну рекламу')
    
    plt.tight_layout()
    plt.show()

# Основная функция
def main(file_path):
    """Основной рабочий процесс."""
    # Шаг 1: Загрузка данных
    data = load_data(file_path)
    
    # Шаг 2: Предобработка
    data = preprocess_data(data)
    
    # Шаг 3: Расчет метрик
    metrics = calculate_metrics(data)
    
    # Шаг 4: Статистический анализ
    analysis = statistical_analysis(data)
    
    # Шаг 5: Формирование выводов
    insights = generate_insights(metrics, analysis)
    
    # Вывод результатов
    print("=== Таблица с метриками ===")
    print(metrics)
    print("\n=== Анализ статистической значимости ===")
    print(analysis)
    print("\n=== Выводы и рекомендации ===")
    for insight in insights:
        print("- " + insight)
    
    # Визуализация
    plot_metrics(metrics)

# Запуск программы
if __name__ == "__main__":
    # Задайте путь к файлу
    file_path = input("Введите путь к файлу data.csv: ").strip()
    main(file_path)
