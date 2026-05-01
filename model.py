import sqlite3
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
import lightgbm as lgb
import warnings
import os
warnings.filterwarnings('ignore')

class DataLoader:
    """Загружает данные для любого набора инструментов из БД"""
    
    def __init__(self, market_db_path, news_db_path=None):
        self.market_conn = sqlite3.connect(market_db_path)
        self.news_conn = sqlite3.connect(news_db_path) if news_db_path and os.path.exists(news_db_path) else None
        
    def list_available_tables(self):
        cursor = self.market_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        return [t[0] for t in tables]
    
    def load_instrument(self, table_name, start_date=None, end_date=None):
        query = f"SELECT datetime, open, high, low, close, volume FROM {table_name}"
        df = pd.read_sql_query(query, self.market_conn)
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        if start_date:
            df = df[df['datetime'] >= start_date]
        if end_date:
            df = df[df['datetime'] <= end_date]
        
        # Извлекаем имя инструмента (sber, gold, imoex, usd_rub)
        instrument_name = table_name.replace('_5min', '').replace('_1min', '')
        
        # Переименовываем колонки
        df = df.rename(columns={
            'open': f'{instrument_name}_open',
            'high': f'{instrument_name}_high',
            'low': f'{instrument_name}_low',
            'close': f'{instrument_name}_close',
            'volume': f'{instrument_name}_volume'
        })
        
        return df[['datetime', f'{instrument_name}_open', f'{instrument_name}_high', 
                   f'{instrument_name}_low', f'{instrument_name}_close', f'{instrument_name}_volume']]
    
    def load_all_instruments(self, table_names, start_date=None, end_date=None):
        all_data = {}
        
        for table in table_names:
            try:
                df = self.load_instrument(table, start_date, end_date)
                instrument_name = table.replace('_5min', '').replace('_1min', '')
                all_data[instrument_name] = df
                print(f" {instrument_name}: {len(df)} записей")
            except Exception as e:
                print(f" Ошибка загрузки {table}: {e}")
        
        if not all_data:
            raise ValueError("Не удалось загрузить ни один инструмент")
        
        # Объединяем по datetime
        merged = None
        for name, df in all_data.items():
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, on='datetime', how='inner')
        
        merged = merged.sort_values('datetime').reset_index(drop=True)
        
        print(f"  После объединения: {len(merged)} строк")
        print(f"  Колонки: {list(merged.columns)}")
        
        return merged, list(all_data.keys())
    
    def load_news(self, start_date=None, end_date=None):
        if self.news_conn is None:
            return pd.DataFrame()
        
        query = "SELECT id, title, published, full_text, tags FROM news"
        news_df = pd.read_sql_query(query, self.news_conn)
        
        news_df['published'] = pd.to_datetime(news_df['published'])
        news_df = news_df.sort_values('published').reset_index(drop=True)
        
        if start_date:
            news_df = news_df[news_df['published'] >= start_date]
        if end_date:
            news_df = news_df[news_df['published'] <= end_date]
        
        news_df['text'] = news_df['title'].fillna('') + '. ' + news_df['full_text'].fillna('')
        
        print(f" Новости: {len(news_df)} записей")
        return news_df
    
    def close(self):
        self.market_conn.close()
        if self.news_conn:
            self.news_conn.close()

class FeatureBuilder:
    """
    Строит признаки для набора инструментов
    """
    
    def __init__(self, target_name):
        """
        Parameters:
        -----------
        target_name : str
            Название инструмента, который предсказываем (например, 'sber')
        """
        self.target = target_name
        
    def add_technical_features(self, df, instruments):
        """
        Добавляет технические признаки для всех инструментов
        """
        result = df.copy()
        
        # Убеждаемся, что instruments - это список строк
        if isinstance(instruments, str):
            instruments = [instruments]
        
        for inst in instruments:
            # Проверяем, что inst - строка
            if not isinstance(inst, str):
                print(f" Пропускаем inst={inst}, тип={type(inst)}")
                continue
            
            close_col = f'{inst}_close'
            if close_col not in result.columns:
                print(f" Колонка {close_col} не найдена, пропускаем {inst}")
                continue
            
            # Принудительно преобразуем в float
            result[close_col] = pd.to_numeric(result[close_col], errors='coerce')
            
            # 1. Доходности
            result[f'{inst}_return_1'] = result[close_col].pct_change()
            result[f'{inst}_return_5'] = result[close_col].pct_change(5)
            result[f'{inst}_return_10'] = result[close_col].pct_change(10)
            result[f'{inst}_return_20'] = result[close_col].pct_change(20)
            
            # 2. Логарифмические доходности
            result[f'{inst}_log_return'] = np.log(result[close_col] / result[close_col].shift(1))
            
            # 3. Свечные паттерны
            open_col = f'{inst}_open'
            high_col = f'{inst}_high'
            low_col = f'{inst}_low'
            
            if open_col in result.columns and high_col in result.columns and low_col in result.columns:
                result[open_col] = pd.to_numeric(result[open_col], errors='coerce')
                result[high_col] = pd.to_numeric(result[high_col], errors='coerce')
                result[low_col] = pd.to_numeric(result[low_col], errors='coerce')
                
                result[f'{inst}_body'] = abs(result[close_col] - result[open_col])
                result[f'{inst}_range'] = result[high_col] - result[low_col]
                result[f'{inst}_body_ratio'] = result[f'{inst}_body'] / (result[f'{inst}_range'] + 1e-6)
            
            # 4. Скользящие средние
            for period in [5, 10, 20, 40]:
                sma = result[close_col].rolling(period).mean()
                result[f'{inst}_sma_{period}'] = sma
                result[f'{inst}_dist_to_sma_{period}'] = (result[close_col] / sma - 1) * 100
            
            # 5. Волатильность
            for period in [5, 10, 20]:
                result[f'{inst}_volatility_{period}'] = result[f'{inst}_return_1'].rolling(period).std() * 100
            
            # 6. RSI
            delta = result[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result[f'{inst}_rsi'] = 100 - (100 / (1 + rs))
            
            # 7. Объемные признаки
            volume_col = f'{inst}_volume'
            if volume_col in result.columns:
                result[volume_col] = pd.to_numeric(result[volume_col], errors='coerce').fillna(0)
                result[f'{inst}_volume_ma_10'] = result[volume_col].rolling(10).mean()
                result[f'{inst}_volume_ratio'] = result[volume_col] / (result[f'{inst}_volume_ma_10'] + 1)
                result[f'{inst}_volume_zscore'] = (result[volume_col] - result[volume_col].rolling(20).mean()) / (result[volume_col].rolling(20).std() + 1e-6)
            
            # 8. Лаги
            for lag in [1, 2, 3, 5, 10]:
                result[f'{inst}_return_lag_{lag}'] = result[f'{inst}_return_1'].shift(lag)
            
            # Временные признаки
            if 'datetime' in result.columns:
                result['hour'] = result['datetime'].dt.hour
                result['minute'] = result['datetime'].dt.minute
                result['day_of_week'] = result['datetime'].dt.dayofweek
                result['day_of_month'] = result['datetime'].dt.day
            
            # Удаляем строки с NaN
            result = result.dropna().reset_index(drop=True)
            
            return result
    
    def extract_news_features(self, news_df, bar_time, window_minutes):
        """
        Извлекает признаки из новостей за окно
        """
        if news_df.empty:
            return {
                'count': 0,
                'sentiment': 0,
                'minutes_since_last': window_minutes,
                'urgency': 0,
                'has_target_mention': 0,
                'has_russia': 0,
                'has_economics': 0,
                'avg_title_len': 0
            }
        
        window_start = bar_time - timedelta(minutes=window_minutes)
        mask = (news_df['published'] <= bar_time) & (news_df['published'] > window_start)
        news_window = news_df[mask]
        
        if len(news_window) == 0:
            return {
                'count': 0,
                'sentiment': 0,
                'minutes_since_last': window_minutes,
                'urgency': 0,
                'has_target_mention': 0,
                'has_economics': 0,
            }
        
        # Объединяем текст
        all_text = ' '.join(news_window['text'].tolist()).lower()
        all_tags = ' '.join(news_window['tags'].fillna('').tolist()).lower()
        
        # Сентимент по ключевым словам
        pos_words = ['рост', 'вырос', 'увеличился', 'прибыль', 'позитив', 'хороший', 'рекорд', 'дивиденд', 'успех']
        neg_words = ['падение', 'упал', 'снижение', 'убыток', 'негатив', 'плохой', 'кризис', 'штраф', 'проблема']
        
        pos_count = sum(all_text.count(w) for w in pos_words)
        neg_count = sum(all_text.count(w) for w in neg_words)
        sentiment = (pos_count - neg_count) / (pos_count + neg_count + 1)
        
        # Упоминание целевого инструмента
        has_target = 1 if (self.target in all_text) else 0
        
        # Экономика
        has_economics = 1 if ('экономика' in all_tags or 'рынок' in all_tags or 'финансы' in all_tags) else 0
        
        # Срочность (экспоненциальное затухание)
        latest_time = news_window['published'].max()
        minutes_since = (bar_time - latest_time).total_seconds() / 60
        urgency = np.exp(-minutes_since / 30)
        
        return {
            'count': len(news_window),
            'sentiment': sentiment,
            'minutes_since_last': minutes_since,
            'urgency': urgency,
            'has_target_mention': has_target,
            'has_economics': has_economics
        }
    
    def add_news_features(self, df, news_df, windows=[5, 15, 30, 60, 120]):
        """
        МАКСИМАЛЬНО БЫСТРАЯ ВЕРСИЯ
        Использует векторизацию и кумулятивные подсчёты
        """
        if news_df is None or news_df.empty:
            return df
    
        result = df.copy()
        news = news_df.copy()
        news['published'] = pd.to_datetime(news['published'])
        news = news.sort_values('published').reset_index(drop=True)
        
        # Предварительный расчёт сентимента (один раз)
        print("  Расчёт сентимента новостей...")
        pos_words = ['рост', 'вырос', 'увеличился', 'прибыль', 'позитив', 
                    'хороший', 'рекорд', 'дивиденд', 'успех', 'контракт']
        neg_words = ['падение', 'упал', 'снижение', 'убыток', 'негатив', 
                    'плохой', 'кризис', 'штраф', 'проблема', 'риск']
        
        texts = news['text'].str.lower().fillna('')
        pos_counts = texts.apply(lambda x: sum(x.count(w) for w in pos_words))
        neg_counts = texts.apply(lambda x: sum(x.count(w) for w in neg_words))
        news['sentiment'] = (pos_counts - neg_counts) / (pos_counts + neg_counts + 1)
        news['has_sber'] = texts.str.contains('сбер|сбербанк', na=False).astype(int)
        
        # Создаём индекс времени для быстрого поиска
        result = result.sort_values('datetime').reset_index(drop=True)
        news_times = news['published'].values
        news_sentiments = news['sentiment'].values
        news_has_sber = news['has_sber'].values
        
        import numpy as np
        from bisect import bisect_right
        
        print("  Агрегация новостей по окнам...")
        
        for window in windows:
            print(f"    Окно {window} мин...")
            prefix = f'news_{window}min'
            window_seconds = window * 60
            
            counts = np.zeros(len(result), dtype=int)
            sentiments = np.zeros(len(result))
            mins_since = np.full(len(result), float(window))
            has_sber = np.zeros(len(result), dtype=int)
            
            for i, row in enumerate(result.itertuples()):
                bar_time = row.datetime
                bar_timestamp = bar_time.timestamp()
                window_start = bar_timestamp - window_seconds
                
                # Бинарный поиск границ окна
                left_idx = bisect_right(news_times, pd.Timestamp.fromtimestamp(window_start)) - 1
                right_idx = bisect_right(news_times, bar_time) - 1
                
                if left_idx <= right_idx and left_idx >= 0:
                    window_news_count = right_idx - left_idx + 1
                    counts[i] = window_news_count
                    
                    # Агрегируем сентимент
                    window_sentiments = news_sentiments[left_idx:right_idx+1]
                    sentiments[i] = window_sentiments.mean()
                    window_has_sber = news_has_sber[left_idx:right_idx+1]
                    has_sber[i] = window_has_sber.max()
                    
                    # Время с последней новости
                    last_news_time = news_times[right_idx]
                    mins_since[i] = (bar_time - last_news_time).total_seconds() / 60
            
            result[f'{prefix}_count'] = counts
            result[f'{prefix}_sentiment'] = sentiments
            result[f'{prefix}_minutes_since_last'] = mins_since
            result[f'{prefix}_has_sber'] = has_sber
        
        return result
    
    def create_target(self, df, up_threshold_pct=0.1, down_threshold_pct=-0.1):
        """
        Создает целевую переменную для следующей свечи
        
        Parameters:
        -----------
        up_threshold_pct : float
            Порог для сигнала UP в процентах (0.1 = 0.1%)
        down_threshold_pct : float
            Порог для сигнала DOWN в процентах
        """
        df = df.copy()
        
        target_close = f'{self.target}_close'
        if target_close not in df.columns:
            raise ValueError(f"Колонка {target_close} не найдена")
        
        # Будущая доходность
        future_return = df[target_close].shift(-1) / df[target_close] - 1
        
        # Преобразуем проценты в десятичные
        up_threshold = up_threshold_pct / 100
        down_threshold = down_threshold_pct / 100
        
        # Целевые классы
        df['target'] = 0
        df.loc[future_return > up_threshold, 'target'] = 1
        df.loc[future_return < down_threshold, 'target'] = -1
        df['future_return'] = future_return
        
        # Убираем последнюю строку
        df = df[:-1].copy()
        
        print(f"\n Целевые классы для {self.target}:")
        for cls in [-1, 0, 1]:
            count = (df['target'] == cls).sum()
            pct = count / len(df) * 100
            name = "DOWN" if cls == -1 else "NEUTRAL" if cls == 0 else "UP"
            print(f"     {name} ({cls:2d}): {count:6d} ({pct:.2f}%)")
        
        return df

class Trainer:
    """Универсальный тренер для любого инструмента"""
    
    def __init__(self, model_params=None):
        default_params = {
            'n_estimators': 300,
            'max_depth': 7,
            'learning_rate': 0.02,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'class_weight': 'balanced',
            'random_state': 42,
            'verbose': -1
        }
        self.params = model_params or default_params
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, X, y, n_splits=5):
        """
        Обучение с временной кросс-валидацией
        """
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ МОДЕЛИ")
        print("="*60)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        X_scaled = self.scaler.fit_transform(X)
        
        results = {'macro_f1': [], 'accuracy': []}
        all_y_true, all_y_pred = [], []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**self.params)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            
            macro_f1 = f1_score(y_val, y_pred, average='macro')
            acc = accuracy_score(y_val, y_pred)
            
            results['macro_f1'].append(macro_f1)
            results['accuracy'].append(acc)
            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)
            
            print(f"  Fold {fold+1}: Macro F1 = {macro_f1:.4f}, Acc = {acc:.4f}")
        
        print(f"\n Средние результаты:")
        print(f"  Macro F1: {np.mean(results['macro_f1']):.4f} (+/- {np.std(results['macro_f1']):.4f})")
        print(f"  Accuracy: {np.mean(results['accuracy']):.4f} (+/- {np.std(results['accuracy']):.4f})")
        
        print("\n" + "="*60)
        print("КЛАССИФИКАЦИОННЫЙ ОТЧЕТ")
        print("="*60)
        print(classification_report(all_y_true, all_y_pred, 
                                     target_names=['DOWN', 'NEUTRAL', 'UP']))
        
        # Финальная модель
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X_scaled, y)
        
        # Важность признаков
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return results, importance
    
    def predict(self, X):
        """Предсказание для новых данных"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Вероятности классов"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

def run_for_target(target_name, 
                   market_db_path,
                   news_db_path=None,
                   table_names=None,
                   start_date='2025-01-01',
                   end_date='2025-12-31',
                   up_threshold=0.1,
                   down_threshold=-0.1):
    """
    Запуск модели для указанного инструмента
    
    Parameters:
    -----------
    target_name : str
        Какой инструмент предсказываем ('sber', 'gold', 'imoex', 'usd_rub')
    market_db_path : str
        Путь к БД с рыночными данными
    news_db_path : str or None
        Путь к БД с новостями
    table_names : list or None
        Список таблиц для загрузки (если None - используем стандартные)
    start_date, end_date : str
        Период данных
    up_threshold, down_threshold : float
        Пороги в процентах
    """
    
    print("\n" + "="*30)
    print(f"МОДЕЛЬ ДЛЯ {target_name.upper()}")
    print("="*30)
    
    # 1. Загрузка данных
    print("\n ЗАГРУЗКА ДАННЫХ")
    print("="*30)
    
    loader = DataLoader(market_db_path, news_db_path)
    
    # Автоматическое определение таблиц, если не указаны
    if table_names is None:
        available = loader.list_available_tables()
        # Ищем таблицы для известных инструментов
        possible_names = ['sber', 'gold', 'imoex', 'usdrub', 'moex', 'rts', 'usd_rub']
        table_names = [t for t in available if any(name in t.lower() for name in possible_names)]
        print(f"  Найдены таблицы: {table_names}")
    
    # Загружаем все инструменты
    merged_df, instruments = loader.load_all_instruments(table_names, start_date, end_date)
    news_df = loader.load_news(start_date, end_date) if news_db_path else pd.DataFrame()
    
    loader.close()
    
    # Проверяем, что целевой инструмент загружен
    if target_name not in instruments:
        raise ValueError(f"Инструмент {target_name} не найден в загруженных данных. Доступны: {instruments}")
    
    # 2. Построение признаков
    print("\n ПОСТРОЕНИЕ ПРИЗНАКОВ")
    print("="*60)
    
    engineer = FeatureBuilder(target_name)
    
    # Технические признаки
    print("  Технические индикаторы...")
    df = engineer.add_technical_features(merged_df, instruments)
    
    # Новостные признаки
    if not news_df.empty:
        df = engineer.add_news_features(df, news_df)
    
    # Целевая переменная
    df = engineer.create_target(df, up_threshold, down_threshold)
    
    # 3. Подготовка к обучению
    exclude_cols = ['datetime', 'target', 'future_return']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target']
    
    print(f"\n  Размер выборки: {len(X)} примеров")
    print(f"  Количество признаков: {len(feature_cols)}")
    
    # 4. Обучение
    trainer = Trainer()
    results, importance = trainer.train(X, y, n_splits=5)
    
    # 5. Топ признаков
    print("\n" + "="*60)
    print("ТОП-20 ВАЖНЫХ ПРИЗНАКОВ")
    print("="*60)
    
    for i, row in importance.head(20).iterrows():
        print(f"{i+1:2d}. {row['feature']:45s} {row['importance']:8.0f}")
    
    # 6. Пример предсказания
    print("\n" + "="*60)
    print("ПРИМЕР ПРЕДСКАЗАНИЯ")
    print("="*60)
    
    X_scaled = trainer.scaler.transform(X)
    last_features = X_scaled[-1:].reshape(1, -1)
    pred = trainer.model.predict(last_features)[0]
    proba = trainer.model.predict_proba(last_features)[0]
    
    signal_map = {-1: 'ПРОДАЖА (прогноз падения)', 
                  0: 'НЕЙТРАЛЬНО', 
                  1: 'ПОКУПКА (прогноз роста)'}
    
    print(f"Сигнал: {signal_map[pred]}")
    print(f"Уверенность: {max(proba):.2%}")
    print("\nВероятности:")
    print(f"  Падение (-1): {proba[0]:.2%}")
    print(f"  Нейтрально (0): {proba[1]:.2%}")
    print(f"  Рост (+1): {proba[2]:.2%}")
    
    return trainer, results, importance, df


if __name__ == "__main__":
    
    # ПУТИ К ФАЙЛАМ (ИЗМЕНИТЕ ПОД СЕБЯ)
    MARKET_DB = "data/market_data_clean.db"
    NEWS_DB = "data/ria_news_all_2025.db"
    
    # Можете запустить для любого инструмента:
    
    # 1. Для Сбера
    print("\n" + "="*30)
    print("ЗАПУСК ДЛЯ СБЕРА")
    print("="*30)
    trainer_sber, _, _, _ = run_for_target(
        target_name='sber',
        market_db_path=MARKET_DB,
        news_db_path=NEWS_DB,
        start_date='2025-01-01',
        end_date='2025-12-31'
    )
    
    # 2. Для Золота
    print("\n" + "="*30)
    print("ЗАПУСК ДЛЯ ЗОЛОТА")
    print("="*30)
    trainer_gold, _, _, _ = run_for_target(
        target_name='gold',
        market_db_path=MARKET_DB,
        news_db_path=NEWS_DB,
        start_date='2025-01-01',
        end_date='2025-12-31'
    )
    
    # 3. Для Индекса MOEX
    print("\n" + "="*30)
    print("ЗАПУСК ДЛЯ ИНДЕКСА MOEX")
    print("="*30)
    trainer_imoex, _, _, _ = run_for_target(
        target_name='imoex',
        market_db_path=MARKET_DB,
        news_db_path=NEWS_DB,
        start_date='2025-01-01',
        end_date='2025-12-31'
    )
    
    # 4. Для Доллара/Рубль
    print("\n" + "="*30)
    print("ЗАПУСК ДЛЯ ДОЛЛАР/РУБЛЬ")
    print("="*30)
    trainer_usdrub, _, _, _ = run_for_target(
        target_name='usd_rub',
        market_db_path=MARKET_DB,
        news_db_path=NEWS_DB,
        start_date='2025-01-01',
        end_date='2025-12-31'
    )