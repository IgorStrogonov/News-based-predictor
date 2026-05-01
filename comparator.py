"""
Сравнение трёх моделей:
1. Market-only (только рыночные данные)
2. News-only (только новости)
3. Fusion (рынок + новости)

Цель: доказать, что новости добавляют предсказательную силу
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

def load_market_data(db_path, ticker='sber'):
    """Загружает только рыночные данные"""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"""
        SELECT datetime, open, high, low, close, volume 
        FROM {ticker}_5min 
        ORDER BY datetime
    """, conn)
    conn.close()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def load_news_data(db_path, start_date=None, end_date=None):
    """Загружает новости"""
    conn = sqlite3.connect(db_path)
    news = pd.read_sql_query("""
        SELECT id, title, published, full_text, tags 
        FROM news
    """, conn)
    conn.close()
    news['published'] = pd.to_datetime(news['published'])
    news['text'] = news['title'] + '. ' + news['full_text'].fillna('')
    
    if start_date:
        news = news[news['published'] >= start_date]
    if end_date:
        news = news[news['published'] <= end_date]
    
    return news

# ============================================================
# 2. ПРИЗНАКИ (единые для всех моделей)
# ============================================================

def add_market_features(df):
    """Добавляет рыночные признаки"""
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

def add_news_features(df, news_df, windows=[60]):
    """Добавляет новостные признаки (упрощённые, без эмбеддингов)"""
    if news_df is None or len(news_df) == 0:
        return df
    
    result = df.copy()
    news = news_df.sort_values('published').reset_index(drop=True)
    
    # Простой сентимент для новостей (один раз)
    pos_words = ['рост', 'вырос', 'увеличился', 'прибыль', 'позитив', 'хороший', 'рекорд']
    neg_words = ['падение', 'упал', 'снижение', 'убыток', 'негатив', 'плохой', 'кризис']
    
    def get_sentiment(text):
        text_lower = text.lower()
        pos = sum(text_lower.count(w) for w in pos_words)
        neg = sum(text_lower.count(w) for w in neg_words)
        return (pos - neg) / (pos + neg + 1)
    
    news['sentiment'] = news['text'].apply(get_sentiment)
    news['has_sber'] = news['text'].str.lower().str.contains('сбер|сбербанк', na=False).astype(int)
    
    for window in windows:
        prefix = f'news_{window}min'
        window_td = pd.Timedelta(minutes=window)
        
        counts = []
        sentiments = []
        has_sbers = []
        
        for idx, row in tqdm(result.iterrows(), total=len(result), desc=f"Окно {window} мин"):
            bar_time = row['datetime']
            window_start = bar_time - window_td
            
            mask = (news['published'] <= bar_time) & (news['published'] > window_start)
            news_window = news[mask]
            
            counts.append(len(news_window))
            sentiments.append(news_window['sentiment'].mean() if len(news_window) > 0 else 0)
            has_sbers.append(news_window['has_sber'].max() if len(news_window) > 0 else 0)
        
        result[f'{prefix}_count'] = counts
        result[f'{prefix}_sentiment'] = sentiments
        result[f'{prefix}_has_sber'] = has_sbers
    
    return result

def create_target(df, up_threshold_pct=0.1, down_threshold_pct=-0.1):
    """Создаёт целевую переменную"""
    df = df.copy()
    up_th = up_threshold_pct / 100
    down_th = down_threshold_pct / 100
    
    df['future_return'] = df['close'].shift(-1) / df['close'] - 1
    df['target'] = 0
    df.loc[df['future_return'] > up_th, 'target'] = 1
    df.loc[df['future_return'] < down_th, 'target'] = -1
    
    # Убираем последнюю строку
    df = df[:-1].copy()
    
    return df

# ============================================================
# 3. ОБУЧЕНИЕ МОДЕЛИ
# ============================================================

def train_model(X, y, model_name="Model", n_splits=5):
    """Обучает модель и возвращает Macro F1"""
    print(f"\n{'='*60}")
    print(f"ОБУЧЕНИЕ: {model_name}")
    print(f"{'='*60}")
    print(f"  Признаков: {X.shape[1]}")
    print(f"  Примеров: {X.shape[0]}")
    
    # Удаляем строки с NaN
    mask = ~X.isna().any(axis=1)
    X = X[mask]
    y = y[mask]
    print(f"  После очистки: {X.shape[0]} примеров")
    
    if X.shape[0] == 0:
        print("  Нет данных для обучения!")
        return 0.0
    
    # Стандартизация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Временная кросс-валидация
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.02,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        macro_f1 = f1_score(y_val, y_pred, average='macro')
        scores.append(macro_f1)
        print(f"  Fold {fold+1}: Macro F1 = {macro_f1:.4f}")
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"\n  ИТОГ: Macro F1 = {mean_score:.4f} (+/- {std_score:.4f})")
    
    return mean_score

# ============================================================
# 4. ГЛАВНАЯ ФУНКЦИЯ СРАВНЕНИЯ
# ============================================================

def compare_models(market_db_path, news_db_path, ticker='sber'):
    """
    Сравнивает три модели:
    1. Market-only (только рыночные признаки)
    2. News-only (только новостные признаки)
    3. Fusion (рынок + новости)
    """
    print("="*70)
    print("СРАВНЕНИЕ ТРЁХ МОДЕЛЕЙ")
    print("="*70)
    
    # Загружаем данные
    print("\n ЗАГРУЗКА ДАННЫХ")
    print("-"*50)
    
    market_df = load_market_data(market_db_path, ticker)
    news_df = load_news_data(news_db_path, 
                             start_date=market_df['datetime'].min(),
                             end_date=market_df['datetime'].max())
    
    print(f"  Рыночные данные: {len(market_df)} свечей")
    print(f"  Новости: {len(news_df)} записей")
    
    # Добавляем признаки к рыночным данным
    print("\n ПОСТРОЕНИЕ ПРИЗНАКОВ")
    print("-"*50)
    
    df_with_market = add_market_features(market_df)
    df_with_market = create_target(df_with_market)
    
    df_with_news = add_news_features(market_df, news_df)
    df_with_news = create_target(df_with_news)
    
    df_fusion = add_market_features(market_df)
    df_fusion = add_news_features(df_fusion, news_df)
    df_fusion = create_target(df_fusion)
    
    # Определяем колонки для каждой модели
    market_features = [c for c in df_with_market.columns 
                       if c not in ['datetime', 'target', 'future_return', 'open', 'high', 'low', 'volume']]
    
    news_features = [c for c in df_with_news.columns 
                     if c not in ['datetime', 'target', 'future_return', 'open', 'high', 'low', 'close', 'volume'] 
                     and 'news_' in c]
    
    fusion_features = [c for c in df_fusion.columns 
                       if c not in ['datetime', 'target', 'future_return', 'open', 'high', 'low', 'volume']]
    
    print(f"  Market-only признаков: {len(market_features)}")
    print(f"  News-only признаков: {len(news_features)}")
    print(f"  Fusion признаков: {len(fusion_features)}")
    
    # Обучаем три модели
    results = {}
    
    # 1. Market-only
    X_market = df_with_market[market_features].copy()
    y_market = df_with_market['target'].copy()
    results['Market-only'] = train_model(X_market, y_market, "Market-only")
    
    # 2. News-only
    X_news = df_with_news[news_features].copy()
    y_news = df_with_news['target'].copy()
    results['News-only'] = train_model(X_news, y_news, "News-only")
    
    # 3. Fusion
    X_fusion = df_fusion[fusion_features].copy()
    y_fusion = df_fusion['target'].copy()
    results['Fusion'] = train_model(X_fusion, y_fusion, "Fusion")
    
    # Финальное сравнение
    print("\n" + "="*70)
    print(" ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*70)
    print(f"\n  {'Модель':<15} {'Macro F1':<10} {'Вывод'}")
    print(f"  {'-'*15} {'-'*10} {'-'*30}")
    
    market_score = results.get('Market-only', 0)
    news_score = results.get('News-only', 0)
    fusion_score = results.get('Fusion', 0)
    
    print(f"  {'Market-only':<15} {market_score:.4f}     {'Базовый уровень'}")
    print(f"  {'News-only':<15} {news_score:.4f}     {'Сигнал в текстах'}")
    print(f"  {'Fusion':<15} {fusion_score:.4f}     {'Рынок + новости'}")
    
    # Анализ
    print("\n" + "="*70)
    print(" ВЫВОДЫ")
    print("="*70)
    
    news_improvement = fusion_score - market_score
    
    if news_improvement > 0.02:
        print(f"\n  Новости ДАЮТ значимый вклад! (+{news_improvement:.4f} Macro F1)")
        print(f"     Fusion лучше Market-only на {news_improvement:.2%}")
    elif news_improvement > 0:
        print(f"\n  Новости дают небольшой вклад (+{news_improvement:.4f} Macro F1)")
        print(f"     Эффект есть, но небольшой")
    else:
        print(f"\n  Новости НЕ ДАЮТ вклада ({news_improvement:+.4f} Macro F1)")
        print(f"     Возможные причины:")
        print(f"     - Мало релевантных новостей о Сбере")
        print(f"     - Метод агрегации не подходит")
        print(f"     - Новости уже учтены в цене")
    
    if news_score > 0.35:
        print(f"\n  News-only показывает {news_score:.4f} — в текстах есть сигнал!")
    else:
        print(f"\n  News-only показывает {news_score:.4f} — сигнал в текстах слабый")
    
    return results


# ============================================================
# 5. ЗАПУСК
# ============================================================

if __name__ == "__main__":
    MARKET_DB = "data/market_data_clean.db"
    NEWS_DB = "data/ria_news_all_2025.db"
    
    results = compare_models(MARKET_DB, NEWS_DB, ticker='sber')