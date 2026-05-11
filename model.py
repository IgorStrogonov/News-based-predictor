"""
Полноценная модель с эмбеддингами новостей (SBERT) и взвешенной агрегацией
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer


# ЗАГРУЗКА ДАННЫХ
class DataLoader:
    def __init__(self, market_db_path, news_db_path):
        self.market_conn = sqlite3.connect(market_db_path)
        self.news_conn = sqlite3.connect(news_db_path)
    
    def load_market_data(self, ticker='sber'):
        df = pd.read_sql_query(f"""
            SELECT datetime, open, high, low, close, volume 
            FROM {ticker}_5min 
            ORDER BY datetime
        """, self.market_conn)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    
    def load_news(self, start_date=None, end_date=None):
        news = pd.read_sql_query("""
            SELECT id, title, published, full_text, tags 
            FROM news
        """, self.news_conn)
        news['published'] = pd.to_datetime(news['published'])
        news['text'] = news['title'] + '. ' + news['full_text'].fillna('')
        
        if start_date:
            news = news[news['published'] >= start_date]
        if end_date:
            news = news[news['published'] <= end_date]
        
        return news
    
    def close(self):
        self.market_conn.close()
        self.news_conn.close()


# ЭМБЕДДИНГИ НОВОСТЕЙ (SBERT)
class NewsEmbedder:
    """
    Преобразование новостей в эмбеддинги с помощью SBERT
    """
    
    def __init__(self, model_name='intfloat/multilingual-e5-large'):
        print(f" Загрузка модели {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("   Модель загружена")
    
    def get_embeddings(self, texts, batch_size=32):
        """
        Получает эмбеддинги для списка текстов
        """
        print(f"   Вычисление эмбеддингов для {len(texts)} новостей...")
        
        # Для модели e5 нужен префикс
        texts_with_prefix = [f"passage: {t[:512]}" for t in texts]
        
        embeddings = self.model.encode(
            texts_with_prefix, 
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def get_embeddings_for_news(self, news_df):
        """
        Добавляет колонку с эмбеддингами в DataFrame новостей
        """
        texts = news_df['text'].fillna('').tolist()
        embeddings = self.get_embeddings(texts)
        
        news_df = news_df.copy()
        news_df['embedding'] = list(embeddings)
        
        return news_df



# ВЗВЕШЕННАЯ АГРЕГАЦИЯ ЭМБЕДДИНГОВ
class WeightedNewsAggregator:
    """
    Агрегирует эмбеддинги новостей с экспоненциальным взвешиванием
    Более свежие новости имеют больший вес
    """
    
    def __init__(self, decay_minutes=30):
        """
        decay_minutes: период полураспада веса (в минутах)
        """
        self.decay_minutes = decay_minutes
    
    def _get_weights(self, news_times, current_time):
        """
        Рассчитывает веса для новостей на основе их возраста
        """
        time_diff = (current_time - news_times).dt.total_seconds() / 60
        decay_rate = np.log(2) / self.decay_minutes
        weights = np.exp(-decay_rate * np.maximum(time_diff, 0))
        return weights
    
    def aggregate_embeddings(self, news_batch, current_time):
        """
        Агрегирует эмбеддинги с экспоненциальным взвешиванием
        Возвращает: 
        - взвешенное среднее эмбеддинга
        - сумму весов (интенсивность)
        - максимальный вес (свежесть самой свежей новости)
        """
        if len(news_batch) == 0:
            # Пустой эмбеддинг (нулевой вектор)
            return np.zeros(768), 0.0, 0.0
        
        embeddings = np.vstack(news_batch['embedding'].values)
        weights = self._get_weights(news_batch['published'], current_time)
        
        total_weight = weights.sum()
        if total_weight > 0:
            weighted_avg = np.average(embeddings, weights=weights, axis=0)
            max_weight = weights.max()
        else:
            weighted_avg = np.mean(embeddings, axis=0)
            max_weight = 0.0
        
        return weighted_avg, total_weight, max_weight
    
    def aggregate_sentiment(self, news_batch, current_time):
        """
        Агрегирует сентимент с тем же взвешиванием
        """
        if len(news_batch) == 0:
            return 0.0, 0.0, 0.0
        
        # Простой сентимент на основе ключевых слов
        pos_words = ['рост', 'вырос', 'увеличился', 'прибыль', 'позитив', 'хороший', 'рекорд']
        neg_words = ['падение', 'упал', 'снижение', 'убыток', 'негатив', 'плохой', 'кризис']
        
        sentiments = []
        for _, row in news_batch.iterrows():
            text = row['text'].lower()
            pos_count = sum(text.count(w) for w in pos_words)
            neg_count = sum(text.count(w) for w in neg_words)
            sent = (pos_count - neg_count) / (pos_count + neg_count + 1)
            sentiments.append(sent)
        
        sentiments = np.array(sentiments)
        weights = self._get_weights(news_batch['published'], current_time)
        
        total_weight = weights.sum()
        if total_weight > 0:
            weighted_sentiment = np.average(sentiments, weights=weights)
        else:
            weighted_sentiment = np.mean(sentiments)
        
        return weighted_sentiment, total_weight, weights.max()



# ПОСТРОЕНИЕ ПРИЗНАКОВ ДЛЯ МОДЕЛИ
class FeatureBuilder:
    """
    Строит признаки для модели: рыночные + новостные (эмбеддинги)
    """
    
    def __init__(self, ticker='sber', embedding_dim=768):
        self.ticker = ticker
        self.embedding_dim = embedding_dim
        self.aggregator = WeightedNewsAggregator(decay_minutes=30)
        self.embedder = None  # будет инициализирован позже
    
    def add_market_features(self, df):
        """
        Добавляет технические признаки (упрощённая версия)
        """
        result = df.copy()
        
        # Доходности
        result['return_1'] = result['close'].pct_change()
        result['return_5'] = result['close'].pct_change(5)
        result['return_10'] = result['close'].pct_change(10)
        
        # Скользящие средние
        result['sma_10'] = result['close'].rolling(10, min_periods=5).mean()
        result['sma_20'] = result['close'].rolling(20, min_periods=10).mean()
        result['dist_to_sma_10'] = (result['close'] / result['sma_10'] - 1) * 100
        
        # Волатильность
        result['volatility_10'] = result['return_1'].rolling(10, min_periods=5).std() * 100
        
        # Лаги
        for lag in [1, 2, 3, 5]:
            result[f'close_lag_{lag}'] = result['close'].shift(lag)
            result[f'return_lag_{lag}'] = result['return_1'].shift(lag)
        
        # Объём
        result['volume_ma_10'] = result['volume'].rolling(10, min_periods=1).mean()
        result['volume_ratio'] = result['volume'] / (result['volume_ma_10'] + 1)
        
        # Временные признаки
        result['hour'] = result['datetime'].dt.hour
        result['minute'] = result['datetime'].dt.minute
        result['day_of_week'] = result['datetime'].dt.dayofweek
        
        return result
    
    def add_news_features(self, df, news_df, windows=[30, 60, 120, 240]):
        """
        Добавляет новостные признаки с эмбеддингами
        Для каждого окна: взвешенный эмбеддинг, интенсивность, сентимент
        """
        if news_df is None or len(news_df) == 0:
            return df
        
        result = df.copy()
        news = news_df.copy()
        news = news.sort_values('published').reset_index(drop=True)
        
        print(f"  Агрегация новостей по окнам {windows}...")
        
        for window in windows:
            print(f"    Окно {window} мин...")
            window_td = timedelta(minutes=window)
            prefix = f'news_{window}min'
            
            # Списки для признаков
            weighted_embeddings = []
            total_weights = []
            max_weights = []
            weighted_sentiments = []
            
            for idx, row in result.iterrows():
                bar_time = row['datetime']
                window_start = bar_time - window_td
                
                # Новости в окне
                mask = (news['published'] <= bar_time) & (news['published'] > window_start)
                news_window = news[mask]
                
                # Агрегация эмбеддингов
                w_emb, t_weight, m_weight = self.aggregator.aggregate_embeddings(
                    news_window, bar_time
                )
                weighted_embeddings.append(w_emb)
                total_weights.append(t_weight)
                max_weights.append(m_weight)
                
                # Агрегация сентимента
                w_sent, _, _ = self.aggregator.aggregate_sentiment(news_window, bar_time)
                weighted_sentiments.append(w_sent)
            
            # Добавляем эмбеддинги как отдельные признаки (первые 10 компонент PCA)
            embeddings_array = np.array(weighted_embeddings)
            
            # Упрощаем: берём первые 10 компонент через PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=10)
            embeddings_pca = pca.fit_transform(embeddings_array)
            
            for i in range(10):
                result[f'{prefix}_emb_pca_{i}'] = embeddings_pca[:, i]
            
            # Добавляем мета-признаки
            result[f'{prefix}_total_weight'] = total_weights
            result[f'{prefix}_max_weight'] = max_weights
            result[f'{prefix}_sentiment'] = weighted_sentiments
            result[f'{prefix}_count'] = [len(news[(news['published'] <= bar_time) & 
                                                   (news['published'] > bar_time - window_td)]) 
                                         for bar_time in result['datetime']]
        
        return result
    
    def create_target(self, df, up_threshold_pct=0.1, down_threshold_pct=-0.1):
        """
        Создаёт целевую переменную для следующей свечи
        """
        df = df.copy()
        
        up_threshold = up_threshold_pct / 100
        down_threshold = down_threshold_pct / 100
        
        df['future_return'] = df['close'].shift(-1) / df['close'] - 1
        df['target'] = 0
        df.loc[df['future_return'] > up_threshold, 'target'] = 1
        df.loc[df['future_return'] < down_threshold, 'target'] = -1
        
        # Убираем последнюю строку
        df = df[:-1].copy()
        
        print(f"\n  Распределение классов для {self.ticker}:")
        for cls in [-1, 0, 1]:
            count = (df['target'] == cls).sum()
            pct = count / len(df) * 100
            name = "DOWN" if cls == -1 else "NEUTRAL" if cls == 0 else "UP"
            print(f"    {name} ({cls:2d}): {count:6d} ({pct:.2f}%)")
        
        return df


# ОБУЧЕНИЕ МОДЕЛИ
class ModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, X, y, n_splits=5):
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ МОДЕЛИ")
        print("="*60)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        X_scaled = self.scaler.fit_transform(X)
        
        results = []
        all_y_true, all_y_pred = [], []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.02,
                class_weight='balanced',
                random_state=42,
                verbose=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            macro_f1 = f1_score(y_val, y_pred, average='macro')
            results.append(macro_f1)
            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)
            
            print(f"  Fold {fold+1}: Macro F1 = {macro_f1:.4f}")
        
        print(f"\n Средний Macro F1: {np.mean(results):.4f} (+/- {np.std(results):.4f})")
        
        print("\n" + "="*60)
        print("КЛАССИФИКАЦИОННЫЙ ОТЧЕТ")
        print("="*60)
        print(classification_report(all_y_true, all_y_pred, 
                                     target_names=['DOWN', 'NEUTRAL', 'UP']))
        
        # Финальная модель
        self.model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.02,
            class_weight='balanced', random_state=42, verbose=-1
        )
        self.model.fit(X_scaled, y)
        
        # Важность признаков
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return results, importance



# MAIN
def main():
    print("="*70)
    print("МОДЕЛЬ С ЭМБЕДДИНГАМИ НОВОСТЕЙ (SBERT)")
    print("="*70)
    
    # Пути к файлам
    MARKET_DB = "data/market_data_clean.db"
    NEWS_DB = "data/ria_news_all_2025.db"
    
    # 1. Загрузка данных
    print("\n ЗАГРУЗКА ДАННЫХ")
    print("-"*50)
    
    loader = DataLoader(MARKET_DB, NEWS_DB)
    market_df = loader.load_market_data('sber')
    news_df_raw = loader.load_news(start_date='2025-01-01', end_date='2025-12-31')
    loader.close()
    
    print(f"  Рыночные данные: {len(market_df)} свечей")
    print(f"  Новости: {len(news_df_raw)} записей")
    
    # 2. Эмбеддинги новостей
    print("\n ВЫЧИСЛЕНИЕ ЭМБЕДДИНГОВ НОВОСТЕЙ")
    print("-"*50)
    
    embedder = NewsEmbedder('intfloat/multilingual-e5-large')
    news_df = embedder.get_embeddings_for_news(news_df_raw)
    
    # 3. Построение признаков
    print("\n ПОСТРОЕНИЕ ПРИЗНАКОВ")
    print("-"*50)
    
    builder = FeatureBuilder(ticker='sber')
    builder.aggregator = WeightedNewsAggregator(decay_minutes=30)
    
    # Рыночные признаки
    df = builder.add_market_features(market_df)
    
    # Новостные признаки (с эмбеддингами)
    df = builder.add_news_features(df, news_df, windows=[30, 60, 120, 240])
    
    # Целевая переменная
    df = builder.create_target(df, up_threshold_pct=0.1, down_threshold_pct=-0.1)
    
    # 4. Подготовка к обучению
    exclude_cols = ['datetime', 'target', 'future_return', 'open', 'high', 'low', 'volume']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target']
    
    print(f"\n  Размер выборки: {len(X)} примеров")
    print(f"  Количество признаков: {len(feature_cols)}")
    
    # 5. Обучение
    trainer = ModelTrainer()
    results, importance = trainer.train(X, y, n_splits=5)
    
    # 6. Топ признаков
    print("\n" + "="*70)
    print(" ТОП-30 ВАЖНЫХ ПРИЗНАКОВ")
    print("="*70)
    
    for i, row in importance.head(30).iterrows():
        # Подсвечиваем новостные признаки
        is_news = 'news_' in row['feature']
        print(f" {row['feature']:40s} {row['importance']:6.0f}")
    
    # 7. Оценка вклада новостей
    print("\n" + "="*70)
    print(" ОЦЕНКА ВКЛАДА НОВОСТЕЙ")
    print("="*70)
    
    news_importance = importance[importance['feature'].str.contains('news_', na=False)]['importance'].sum()
    total_importance = importance['importance'].sum()
    news_share = news_importance / total_importance * 100
    
    print(f"  Суммарная важность новостных признаков: {news_share:.1f}%")
    print(f"  Суммарная важность рыночных признаков: {100 - news_share:.1f}%")
    
    if news_share > 5:
        print("\n   Новости дают значимый вклад в предсказания")
    else:
        print("\n   Вклад новостей незначителен")
        print("     Возможные причины:")
        print("     - Мало релевантных новостей о Сбере")
        print("     - Окна агрегации не подобраны оптимально")
        print("     - Нужно больше данных или другой энкодер")
    
    print("\n Модель готова!")


if __name__ == "__main__":
    main()