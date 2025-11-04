import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

STATION_FILES = {
    2812717: {'name': 'Mamyr-3', 'file': "C:\\Users\\erbos\\Downloads\\openaq_location_2812717_measurments.csv"},
    2812784: {'name': 'Respublika 4', 'file': "C:\\Users\\erbos\\Downloads\\openaq_location_2812784_measurments (1) — копия.csv"}
}
TARGET_PARAMETER = 'pm25' # Какой параметр прогнозируем
TEST_SET_SIZE = 0.2 # Доля данных для тестовой выборки (20%)


# Шаг 1: Загрузка данных из отдельных CSV файлов
print("--- Шаг 1: Загрузка данных из отдельных CSV файлов ---")

raw_dataframes = {}

for station_id, info in STATION_FILES.items():
    file_path = info['file']
    station_name = info['name']
    print(f"Загрузка данных для станции '{station_name}' из файла: {file_path}")

    try:
        df_station = pd.read_csv(file_path)
        print(f"  > Успешно загружено {len(df_station)} строк.")

        # Преобразуем datetimeUtc в datetime объекты как можно раньше
        if 'datetimeUtc' in df_station.columns:
             df_station['datetimeUtc'] = pd.to_datetime(df_station['datetimeUtc'])
        else:
             print(f"  > Внимание: Колонка 'datetimeUtc' не найдена в файле {file_path}. Проверьте формат.")
             continue # Пропускаем станцию, если нет колонки времени

        raw_dataframes[station_id] = df_station

    except FileNotFoundError:
        print(f"  > Ошибка: Файл '{file_path}' не найден.")
        # Продолжаем выполнение для других файлов, не завершаем скрипт
    except Exception as e:
        print(f"  > Произошла ошибка при загрузке файла {file_path}: {e}")
        # Продолжаем выполнение для других файлов


print("\n--- Шаг 1 завершен ---")


# Шаг 2, 3, 4, 5: Предобработка, Feature Engineering, Моделирование и Оценка для КАЖДОЙ станции
print("\n--- Шаги 2-5: Обработка и моделирование для каждой станции ---")

models = {} # Словарь для хранения обученных моделей
evaluation_metrics = {} # Словарь для хранения метрик оценки

# Проходим только по успешно загруженным данным
for station_id, df_raw_station in raw_dataframes.items():
    # Проверяем, что DataFrame не пустой после загрузки
    if df_raw_station.empty:
        print(f"\n--- Пропускаем обработку для станции ID: {station_id}. Нет загруженных данных. ---")
        continue

    # Проверяем, что в данных есть колонка 'location_name', прежде чем к ней обращаться
    if 'location_name' in df_raw_station.columns and not df_raw_station['location_name'].empty:
        station_name = df_raw_station['location_name'].iloc[0]
    else:
        station_name = f"Unknown Station (ID: {station_id})"
        print(f"\n--- Обработка и моделирование для станции с неизвестным именем (ID: {station_id}) ---")
        print(f"  > Внимание: Колонка 'location_name' отсутствует или пуста.")


    print(f"\n--- Обработка и моделирование для станции: {station_name} ---")


    # --- Шаг 2: Предобработка данных для текущей станции ---
    print(f"  > Шаг 2: Предобработка данных...")

    # Фильтруем по выбранному параметру (на случай, если в файле есть другие параметры)
    # Убедимся, что колонка 'parameter' существует
    if 'parameter' in df_raw_station.columns:
        df_param_station = df_raw_station[df_raw_station['parameter'] == TARGET_PARAMETER].copy()
        if df_param_station.empty:
             print(f"    > Внимание: Нет данных по параметру '{TARGET_PARAMETER}' для станции {station_name}. Пропускаем моделирование.")
             continue # Пропускаем станцию, если нет нужного параметра
    else:
        print(f"    > Внимание: Колонка 'parameter' отсутствует для станции {station_name}. Пропускаем моделирование.")
        continue # Пропускаем станцию, если нет колонки параметра


    # Удалим ненужные колонки
    cols_to_drop = ['parameter', 'unit', 'datetimeLocal', 'timezone', 'country_iso', 'isMobile', 'isMonitor', 'owner_name', 'provider']
    df_param_station = df_param_station.drop(columns=[col for col in cols_to_drop if col in df_param_station.columns])

    # Сортировка данных по времени (уже отфильтровано по станции)
    df_param_station = df_param_station.sort_values(by='datetimeUtc')

    # Устанавливаем индекс по времени
    df_param_station = df_param_station.set_index('datetimeUtc')

    # Ресэмплинг до почасовых данных (среднее значение за час)
    # ИСПРАВЛЕНО: используем 'h' вместо 'H'
    hourly_values = df_param_station['value'].resample('h').mean()

    # Заполнение пропусков: ffill, затем bfill
    # ИСПРАВЛЕНО: используем .ffill() и .bfill() напрямую
    hourly_values_filled = hourly_values.ffill().bfill()

    processed_df_station = pd.DataFrame(hourly_values_filled, columns=['value'])

    # Проверка на оставшиеся пропуски (могут быть, если целый час или период без данных)
    if processed_df_station['value'].isnull().sum() > 0:
         print(f"    > Внимание: В станции {station_name} остались пропуски после заполнения ({processed_df_station['value'].isnull().sum()} штук).")


    # --- Шаг 3: Feature Engineering для текущей станции ---
    print(f"  > Шаг 3: Feature Engineering...")

    df_features_station = processed_df_station.copy()
    # Lagged Features (значения из прошлого в часах)
    lags = [1, 2, 3, 6, 12, 24] # Значения 1, 2, 3, 6, 12, 24 часа наза
    for lag in lags:
        # Добавлена проверка, что данных достаточно для создания лага
        if len(df_features_station) > lag:
             df_features_station[f'value_lag_{lag}h'] = df_features_station['value'].shift(lag)
        else:
             print(f"    > Внимание: Недостаточно данных ({len(df_features_station)} строк) для создания лага {lag}h для станции {station_name}. Признак не создан.")
             df_features_station[f'value_lag_{lag}h'] = np.nan # Заполним NaN


    # Time-based Features
    df_features_station['hour_of_day'] = df_features_station.index.hour
    df_features_station['day_of_week'] = df_features_station.index.dayofweek

    # Удаление строк с NaN, появившихся после shift() или из-за недостатка данных
    initial_rows = len(df_features_station)
    df_features_station = df_features_station.dropna()
    rows_removed = initial_rows - len(df_features_station)

    if rows_removed > 0:
        print(f"    > Удалено {rows_removed} строк с пропусками после создания лагов.")

    print(f"  > Получено {len(df_features_station)} строк с признаками для моделирования.")


    # --- Шаг 4: Подготовка данных для XGBoost (разделение на train/test) ---
    print(f"  > Шаг 4: Подготовка данных для XGBoost...")

    if len(df_features_station) == 0:
        print(f"    > Нет данных с признаками для станции {station_name} после Feature Engineering. Пропускаем моделирование.")
        continue # Пропускаем эту станцию, если данных нет

    # Разделение данных на обучающую и тестовую выборки (ХРОНОЛОГИЧЕСКИ)
    # Используем iloc для разделения по индексу (времени), чтобы сохранить порядок
    train_size = int(len(df_features_station) * (1 - TEST_SET_SIZE))
    # Проверка, что обе выборки не пустые
    if train_size == 0 or train_size >= len(df_features_station):
         print(f"    > Недостаточно данных ({len(df_features_station)} строк) для разделения на обучающую и тестовую выборки для станции {station_name}. Пропускаем моделирование.")
         continue

    train_data = df_features_station.iloc[:train_size]
    test_data = df_features_station.iloc[train_size:]

    print(f"    > Размер обучающей выборки: {len(train_data)}")
    print(f"    > Размер тестовой выборки: {len(test_data)}")
    print("    > Дата начала тестовой выборки:", test_data.index.min())

    # Определение признаков (X) и целевой переменной (y)
    features = [col for col in train_data.columns if col != 'value'] # Все колонки, кроме 'value' - это признаки

    X_train, y_train = train_data[features], train_data['value']
    X_test, y_test = test_data[features], test_data['value']

    # Проверка, что в обучающей выборке есть признаки
    if X_train.empty or y_train.empty:
         print(f"    > Обучающая выборка пуста для станции {station_name}. Пропускаем моделирование.")
         continue


    # --- Шаг 5: Обучение, Прогнозирование и Оценка модели XGBoost ---
    print(f"  > Шаг 5: Обучение и оценка модели XGBoost...")

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,             # Количество деревьев
        learning_rate=0.05,           # Скорость обучения
        max_depth=6,                  # Глубина деревьев
        random_state=42,              # Для воспроизводимости
        n_jobs=-1                     # Использовать все ядра
    )

    # --- Обучение модели ---
    # Используем X_train и y_train для обучения
    model.fit(X_train, y_train)

    print(f"    > Обучение завершено для станции {station_name}.")

    # --- Прогнозирование на тестовой выборке ---
    print("\n    > Прогнозирование на тестовой выборке...")
    # Проверка, что тестовая выборка не пуста
    if X_test.empty or y_test.empty:
         print(f"    > Тестовая выборка пуста для станции {station_name}. Пропускаем оценку и визуализацию.")
         continue

    predictions = model.predict(X_test)

    # --- Оценка модели ---
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"    > Метрики оценки для станции {station_name}:")
    print(f"      - Mean Absolute Error (MAE): {mae:.2f} µg/m³")
    print(f"      - Root Mean Squared Error (RMSE): {rmse:.2f} µg/m³")

    # Сохраняем метрики и модель
    evaluation_metrics[station_id] = {'MAE': mae, 'RMSE': rmse, 'station_name': station_name}
    models[station_id] = model # Сохраняем обученную модель


    print(f"  > Визуализация результатов для станции {station_name}...")
    plt.figure(figsize=(15, 7))

    plt.plot(y_test.index, y_test, label='Фактические PM2.5', marker='.', linestyle='-', linewidth=1, markersize=5)
    plt.plot(y_test.index, predictions, label='Прогноз XGBoost', marker='.', linestyle='--', linewidth=1, markersize=5, alpha=0.7)

    plt.title(f'Прогноз PM2.5 vs Фактические значения: Станция "{station_name}" (Тестовая выборка)')
    plt.xlabel('Время (UTC)')
    plt.ylabel('PM2.5 (µg/m³)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


print("\n--- Обработка и моделирование для всех станций завершены ---")

print("\n--- Сводка метрик по станциям ---")
if evaluation_metrics:
    for station_id, metrics in evaluation_metrics.items():
        print(f"Станция '{metrics['station_name']}' (ID: {station_id}): MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")
else:
    print("Модели не были успешно обучены ни для одной станции.")
