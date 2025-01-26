import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import os

# Загрузка данных
def load_data(file_path):
    """Загружает данные из CSV."""
    try:
        # Загружаем данные с использованием переданного пути к файлу
        data = pd.read_csv(file_path)
        print(f"Файл '{file_path}' успешно загружен.")
        return data
    except FileNotFoundError:
        print(f"Ошибка: Файл '{file_path}' не найден. Проверьте путь.")
        exit()
    except Exception as e:
        # Обработка других возможных ошибок (например, проблемы с форматом файла)
        print(f"Произошла ошибка при загрузке файла: {e}")
        exit()

# Пример использования функции:
file_path = '/Users/mikhail/Desktop/Business:site/programm/data.csv'
data = load_data(file_path)


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
    """Рассчитывает метрики для каждой группы."""
    metrics = data.groupby('test_group').agg(
        total_users=('user_id', 'nunique'),  # Общее количество пользователей в группе
        total_converted=('converted', 'sum'),  # Общее количество конверсий в группе
        total_ads=('total_ads', 'sum'),  # Общее количество рекламы в группе
        session_duration=('session_duration', 'sum'),  # Общее время, проведенное пользователями на сайте
        total_sessions=('session_id', 'nunique'),  # Общее количество сессий в группе
        pages_viewed=('pages_viewed', 'sum'),  # Общее количество просмотренных страниц
        avg_age=('age', 'mean'),  # Средний возраст пользователей в группе
        returning_users=('returning_user', 'sum'),  # Количество возвращающихся пользователей
        target_page_reached=('target_page_reached', 'sum')  # Количество пользователей, достигших целевой страницы
    )

    # Основные и дополнительные метрики
    metrics['Конверсия'] = metrics['total_converted'] / metrics['total_users']  # Конверсия: количество конверсий на общее количество пользователей
    metrics['Среднее количество рекламы'] = metrics['total_ads'] / metrics['total_users']  # Среднее количество рекламы на пользователя
    metrics['Среднее время на сайте'] = metrics['session_duration'] / metrics['total_users']  # Среднее время на сайте на одного пользователя
    metrics['Процент неактивных пользователей'] = (metrics['total_users'] - metrics['total_converted']) / metrics['total_users'] * 100  # Процент неактивных пользователей (не сделавших конверсию)
    metrics['Вовлеченность'] = metrics['pages_viewed'] / metrics['total_users']  # Вовлеченность: количество просмотренных страниц на одного пользователя
    metrics['Среднее количество сессий на пользователя'] = metrics['total_sessions'] / metrics['total_users']  # Среднее количество сессий на пользователя
    metrics['Конверсии на пользователя'] = metrics['total_converted'] / metrics['total_users']  # Конверсии на пользователя
    metrics['Конверсии на одну рекламу'] = metrics['total_converted'] / metrics['total_ads']  # Конверсии на одну рекламу
    metrics['Среднее количество страниц на сессию'] = metrics['pages_viewed'] / metrics['total_sessions']  # Среднее количество страниц, просмотренных за одну сессию
    metrics['Коэффициент активности пользователей'] = metrics['total_converted'] / metrics['total_users']  # Коэффициент активности пользователей (отношение конверсий к общему числу пользователей)
    metrics['Процент возвращающихся пользователей'] = metrics['returning_users'] / metrics['total_users'] * 100  # Процент возвращающихся пользователей
    metrics['Процент достижения целевой страницы'] = metrics['target_page_reached'] / metrics['total_users'] * 100  # Процент пользователей, достигших целевой страницы
    metrics['Процент конверсий от общего числа пользователей'] = metrics['total_converted'] / metrics['total_users'] * 100  # Процент конверсий от общего числа пользователей
    metrics['Среднее время на сессию'] = metrics['session_duration'] / metrics['total_sessions']  # Среднее время на сессию
    metrics['Конверсии на сессию'] = metrics['total_converted'] / metrics['total_sessions']  # Конверсии на сессию
    metrics['Среднее количество страниц на пользователя'] = metrics['pages_viewed'] / metrics['total_users']  # Среднее количество страниц на пользователя
    metrics['Коэффициент активности пользователей (сессии)'] = metrics['total_sessions'] / metrics['total_users']  # Коэффициент активности пользователей (отношение сессий к общему числу пользователей)

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

# Визуализация
def plot_dynamic_metrics(metrics):
    """Динамичный график с ответвлениями, улучшенное отображение изменений и чисел внутри графиков."""
    
    # Преобразуем индекс в строки, чтобы можно было работать с ними
    metrics = metrics.reset_index()

    # Создание графиков с Plotly
    fig = go.Figure()

    # Линейный график для 'Конверсии на пользователя'
    fig.add_trace(go.Scatter(
        x=metrics.index, y=metrics['Конверсии на пользователя'],
        mode='lines+markers+text',
        name='Конверсии на пользователя',
        line=dict(color='blue', width=2),
        text=metrics['Конверсии на пользователя'].apply(lambda x: f"{x:.1f}"),  # Числа внутри графика
        textposition='top center',
        hovertemplate='Конверсии на пользователя: %{y}<extra></extra>',
    ))

    # Линейный график для 'Среднее время на сессию'
    fig.add_trace(go.Scatter(
        x=metrics.index, y=metrics['Среднее время на сессию'],
        mode='lines+markers+text',
        name='Среднее время на сессию',
        line=dict(color='green', width=2, dash='dot'),
        text=metrics['Среднее время на сессию'].apply(lambda x: f"{x:.1f}"),  # Числа внутри графика
        textposition='top center',
        hovertemplate='Среднее время на сессию: %{y}<extra></extra>',
    ))

    # Добавление изменений под графиком
    conversion_change = (metrics['Конверсии на пользователя'].iloc[-1] - metrics['Конверсии на пользователя'].iloc[0]) / metrics['Конверсии на пользователя'].iloc[0] * 100
    time_change = (metrics['Среднее время на сессию'].iloc[-1] - metrics['Среднее время на сессию'].iloc[0]) / metrics['Среднее время на сессию'].iloc[0] * 100

    fig.add_annotation(
        x=0.5, y=-0.2, xref="paper", yref="paper",
        text=f"Изменение конверсий на пользователя: {conversion_change:.1f}%",
        showarrow=False, font=dict(size=12, color="red"),
        align="center"
    )

    fig.add_annotation(
        x=0.5, y=-0.25, xref="paper", yref="paper",
        text=f"Изменение времени на сессию: {time_change:.1f}%",
        showarrow=False, font=dict(size=12, color="red"),
        align="center"
    )

    # Настройка оформления
    fig.update_layout(
        title="Динамика конверсий и времени на сессию",
        xaxis_title="Группы",
        yaxis_title="Значения метрик",
        showlegend=True,
        hovermode="closest",
        plot_bgcolor='white',
        margin=dict(t=50, b=100, l=50, r=50),
        height=600,
    )

    fig.show()

def plot_metrics(metrics):
    """Графическое представление метрик с разными типами диаграмм, добавлены аннотации изменений."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 15))  # Увеличено пространство для графиков

    # Столбчатая диаграмма для конверсии
    sns.barplot(x=metrics.index, y='Конверсия', data=metrics, ax=axes[0, 0])
    axes[0, 0].set_title('Конверсия по группам')
    for i in range(len(metrics)):
        axes[0, 0].text(i, metrics['Конверсия'][i] + 0.01, f"{metrics['Конверсия'][i]:.1f}", ha='center')

    # Круговая диаграмма для средней рекламы
    axes[0, 1].pie(metrics['Среднее количество рекламы'], labels=metrics.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Среднее количество рекламы')

    # Линейная диаграмма для конверсий на пользователя (Тренд)
    sns.lineplot(x=metrics.index, y='Конверсии на пользователя', data=metrics, ax=axes[0, 2], marker='o')
    axes[0, 2].set_title('Конверсии на пользователя (Тренд)')
    for i in range(len(metrics)):
        axes[0, 2].text(i, metrics['Конверсии на пользователя'][i] + 0.01, f"{metrics['Конверсии на пользователя'][i]:.1f}", ha='center')

    # Столбчатая диаграмма для конверсий на рекламу
    sns.barplot(x=metrics.index, y='Конверсии на одну рекламу', data=metrics, ax=axes[1, 0])
    axes[1, 0].set_title('Конверсии на одну рекламу')
    for i in range(len(metrics)):
        axes[1, 0].text(i, metrics['Конверсии на одну рекламу'][i] + 0.01, f"{metrics['Конверсии на одну рекламу'][i]:.1f}", ha='center')

    # Гистограмма для вовлеченности пользователей
    sns.barplot(x=metrics.index, y='Вовлеченность', data=metrics, ax=axes[1, 1])
    axes[1, 1].set_title('Вовлеченность пользователей')
    for i in range(len(metrics)):
        axes[1, 1].text(i, metrics['Вовлеченность'][i] + 0.01, f"{metrics['Вовлеченность'][i]:.1f}", ha='center')

    # Круговая диаграмма для процента активных пользователей
    axes[1, 2].pie(metrics['Коэффициент активности пользователей'], labels=metrics.index, autopct='%1.1f%%', startangle=90)
    axes[1, 2].set_title('Процент активных пользователей')

    # Линейная диаграмма для среднего времени на сессию (Динамика)
    sns.lineplot(x=metrics.index, y='Среднее время на сессию', data=metrics, ax=axes[2, 0], marker='o')
    axes[2, 0].set_title('Среднее время на сессию (Динамика)')
    for i in range(len(metrics)):
        axes[2, 0].text(i, metrics['Среднее время на сессию'][i] + 0.01, f"{metrics['Среднее время на сессию'][i]:.1f}", ha='center')

    # Столбчатая диаграмма для процента неактивных пользователей
    sns.barplot(x=metrics.index, y='Процент неактивных пользователей', data=metrics, ax=axes[2, 1])
    axes[2, 1].set_title('Процент неактивных пользователей')
    for i in range(len(metrics)):
        axes[2, 1].text(i, metrics['Процент неактивных пользователей'][i] + 0.01, f"{metrics['Процент неактивных пользователей'][i]:.1f}", ha='center')

    # Столбчатая диаграмма для конверсий на сессию
    sns.barplot(x=metrics.index, y='Конверсии на сессию', data=metrics, ax=axes[2, 2])
    axes[2, 2].set_title('Конверсии на сессию')
    for i in range(len(metrics)):
        axes[2, 2].text(i, metrics['Конверсии на сессию'][i] + 0.01, f"{metrics['Конверсии на сессию'][i]:.1f}", ha='center')

    # Круговая диаграмма для процента возвращающихся пользователей
    axes[3, 0].pie(metrics['Процент возвращающихся пользователей'], labels=metrics.index, autopct='%1.1f%%', startangle=90)
    axes[3, 0].set_title('Процент возвращающихся пользователей')

    # Гистограмма для средней страницы на пользователя
    sns.barplot(x=metrics.index, y='Среднее количество страниц на пользователя', data=metrics, ax=axes[3, 1])
    axes[3, 1].set_title('Среднее количество страниц на пользователя')
    for i in range(len(metrics)):
        axes[3, 1].text(i, metrics['Среднее количество страниц на пользователя'][i] + 0.01, f"{metrics['Среднее количество страниц на пользователя'][i]:.1f}", ha='center')

    # Столбчатая диаграмма для процента достижения целевой страницы (Доля столбца)
    sns.barplot(x=metrics.index, y='Процент достижения целевой страницы', data=metrics, ax=axes[3, 2])
    axes[3, 2].set_title('Процент достижения целевой страницы (Доля столбца)')
    for i in range(len(metrics)):
        axes[3, 2].text(i, metrics['Процент достижения целевой страницы'][i] + 0.01, f"{metrics['Процент достижения целевой страницы'][i]:.1f}", ha='center')

    plt.tight_layout()
    plt.show()

# Пример использования с метриками
metrics = pd.DataFrame({
    'index': ['ad', 'psa', 'grp1', 'grp2'],
    'Конверсии на пользователя': [0.25, 0.28, 0.30, 0.35],
    'Среднее время на сессию': [2.5, 2.7, 3.0, 3.2],
    'Конверсия': [0.15, 0.18, 0.17, 0.20],
    'Среднее количество рекламы': [2, 3, 4, 5],
    'Конверсии на одну рекламу': [0.1, 0.12, 0.15, 0.13],
    'Вовлеченность': [0.75, 0.80, 0.78, 0.82],
    'Коэффициент активности пользователей': [0.60, 0.65, 0.68, 0.70],
    'Среднее время на сессию': [2.5, 2.7, 3.0, 3.2],
    'Процент неактивных пользователей': [0.40, 0.35, 0.30, 0.25],
    'Конверсии на сессию': [0.05, 0.06, 0.07, 0.08],
    'Процент возвращающихся пользователей': [0.30, 0.33, 0.35, 0.40],
    'Среднее количество страниц на пользователя': [1.5, 1.6, 1.7, 1.8],
    'Процент достижения целевой страницы': [0.20, 0.18, 0.15, 0.17]
})

# Устанавливаем индекс на 'index'
metrics.set_index('index', inplace=True)

# Визуализация
plot_metrics(metrics)
plot_dynamic_metrics(metrics)

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
    
    # Вывод результатов
    print("=== Таблица с метриками ===")
    print(metrics)
    print("\n=== Анализ статистической значимости ===")
    print(analysis)
    
    # Визуализация
    plot_metrics(metrics)

# Запуск программы
if __name__ == "__main__":
    # Задайте путь к файлу
    file_path = input("Введите путь к файлу data.csv: ").strip()
    main(file_path)
