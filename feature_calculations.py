# feature_calculations.py

import numpy as np
import pandas as pd
import numba

def resample_data(data, timeframe):
    temp_data = data.copy()
    temp_data['Close time'] = pd.to_datetime(temp_data['Close time'])
    temp_data.set_index('Close time', inplace=True)
    resampled_data = temp_data.resample(timeframe).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    resampled_data.dropna(inplace=True)
    return resampled_data

def format_col_name(col_name, extra_str):
    return f"{col_name}_{extra_str}" if extra_str else col_name

# Include all feature calculation functions here
# For brevity, I'm showing one example function; include all others similarly

def calculate_MA_data(data, window, mode='MA', extra_str=None):
    new_cols = []
    relative_new_cols = []
    if mode == 'MA':  # Simple Moving Average
        col_name = format_col_name(f'SMA_{window}', extra_str)
        relative_col_name = format_col_name(f'SMA_{window}_rel', extra_str)
        data[col_name] = data['Close'].rolling(window=window).mean()
        data[relative_col_name] = (data['Close'] - data[col_name]) / data['Close'] * 100
    elif mode == 'EMA':  # Exponential Moving Average
        col_name = format_col_name(f'EMA_{window}', extra_str)
        relative_col_name = format_col_name(f'EMA_{window}_rel', extra_str)
        data[col_name] = data['Close'].ewm(span=window, adjust=False).mean()
        data[relative_col_name] = (data['Close'] - data[col_name]) / data['Close'] * 100
    elif mode == 'WMA':  # Weighted Moving Average
        col_name = format_col_name(f'WMA_{window}', extra_str)
        relative_col_name = format_col_name(f'WMA_{window}_rel', extra_str)
        weights = list(range(1, window + 1))
        data[col_name] = data['Close'].rolling(window=window).apply(
            lambda x: np.dot(x, weights) / sum(weights), raw=True
        )
        data[relative_col_name] = (data['Close'] - data[col_name]) / data['Close'] * 100
    else:
        raise ValueError(f"Mode {mode} not recognized. Available modes: 'MA', 'EMA', 'WMA'")
    new_cols.append(col_name)
    relative_new_cols.append(relative_col_name)
    return data, new_cols, relative_new_cols

def calculate_ema_bollinger_bands(df, window=20, num_std=2, extra_str=None):
    new_cols = []
    relative_new_cols = []
    
    # EMA 계산
    ma = df['Close'].ewm(span=window, adjust=False).mean()
    # 표준 편차 계산 (EMA를 기준으로)
    std = df['Close'].ewm(span=window, adjust=False).std()
    # 볼린저 밴드 계산
    upper_col = format_col_name(f'Upper_BB_{window}', extra_str)
    lower_col = format_col_name(f'Lower_BB_{window}', extra_str)
    upper_col_rel = format_col_name(f'Upper_BB_{window}_rel', extra_str)
    lower_col_rel = format_col_name(f'Lower_BB_{window}_rel', extra_str)
    
    df[upper_col] = ma + (std * num_std)
    df[lower_col] = ma - (std * num_std)
    df[upper_col_rel] = (df[upper_col] - df['Close']) / df['Close'] * 100
    df[lower_col_rel] = (df['Close'] - df[lower_col]) / df['Close'] * 100
    
    new_cols.extend([upper_col, lower_col])
    relative_new_cols.extend([upper_col_rel, lower_col_rel])
    
    return df, new_cols, relative_new_cols

def calculate_rsi(df, window=14, extra_str=None):
    new_cols = []
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_col = format_col_name(f'RSI_{window}', extra_str)
    df[rsi_col] = rsi - 50
    new_cols.append(rsi_col)
    return df, new_cols

def calculate_macd(df, short_window=12, long_window=26, signal_window=9, extra_str=None):
    new_cols = []
    
    short_ema = df['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long_window, adjust=False).mean()
    macd_col = format_col_name(f'MACD_{short_window}_{long_window}', extra_str)
    signal_col = format_col_name(f'Signal_Line_{signal_window}', extra_str)
    
    macd_data = short_ema - long_ema
    df[macd_col] = macd_data
    df[signal_col] = df[macd_col].ewm(span=signal_window, adjust=False).mean()
    df[macd_col] = df[macd_col] / df['Close'] * 100
    df[signal_col] = df[signal_col] / df['Close'] * 100
    
    new_cols.extend([macd_col, signal_col])
    
    return df, new_cols

def calculate_stochastic_oscillator(df, k_window=14, d_window=3, extra_str=None):
    new_cols = []
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    k_col = format_col_name(f'%K_{k_window}', extra_str)
    d_col = format_col_name(f'%D_{d_window}', extra_str)
    
    df[k_col] = 100 * ((df['Close'] - low_min) / (high_max - low_min)) - 50
    df[d_col] = df[k_col].rolling(window=d_window).mean()
    
    new_cols.extend([k_col, d_col])
    
    return df, new_cols

def calculate_adx(df, window=14, extra_str=None):
    new_cols = []
    
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(df['High'] - df['Low'])
    tr2 = pd.DataFrame(abs(df['High'] - df['Close'].shift()))
    tr3 = pd.DataFrame(abs(df['Low'] - df['Close'].shift()))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (abs(minus_dm).rolling(window=window).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=window).mean()
    
    adx_col = format_col_name(f'ADX_{window}', extra_str)
    df[adx_col] = adx
    new_cols.append(adx_col)
    
    return df, new_cols

def calculate_cci(df, window=20, extra_str=None):
    new_cols = []
    
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mean_tp = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: pd.Series(x).mad())
    cci = (tp - mean_tp) / (0.015 * mad)
    
    cci_col = format_col_name(f'CCI_{window}', extra_str)
    df[cci_col] = cci
    new_cols.append(cci_col)
    
    return df, new_cols

def calculate_atr(df, window=14, extra_str=None):
    new_cols = []
    
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = high_low.combine(high_close, max).combine(low_close, max)
    atr = tr.rolling(window=window).mean()
    
    atr_col = format_col_name(f'ATR_{window}', extra_str)
    df[atr_col] = atr
    new_cols.append(atr_col)
    
    return df, new_cols

def calculate_obv(df, extra_str=None):
    new_cols = []
    
    df['daily_ret'] = df['Close'].pct_change()
    df['direction'] = df['daily_ret'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    df['vol_direction'] = df['Volume'] * df['direction']
    
    obv_col = format_col_name('OBV', extra_str)
    df[obv_col] = df['vol_direction'].cumsum()
    new_cols.append(obv_col)
    
    return df, new_cols

def calculate_williams_r(df, window=14, extra_str=None):
    new_cols = []
    
    high_max = df['High'].rolling(window=window).max()
    low_min = df['Low'].rolling(window=window).min()
    williams_r = (high_max - df['Close']) / (high_max - low_min) * 100
    
    wr_col = format_col_name(f'Williams_%R_{window}', extra_str)
    df[wr_col] = williams_r - 50
    new_cols.append(wr_col)
    
    return df, new_cols

import numpy as np
import pandas as pd
import numba

# Numba를 이용해 각 윈도우의 지원/저항을 계산하는 함수
@numba.njit
def compute_support_resistance_numba(volumes, lows, highs, avg_vol, window):
    n = volumes.shape[0]
    support = np.empty(n)
    resistance = np.empty(n)
    
    # 초기 window-1 값은 NaN 처리
    for i in range(window - 1):
        support[i] = np.nan
        resistance[i] = np.nan
        
    # 각 인덱스 i에 대해, window 크기만큼의 데이터를 확인
    for i in range(window - 1, n):
        start = i - window + 1
        candidate_support = np.inf
        candidate_resistance = -np.inf
        found_candidate = False
        
        # 미리 계산해둔 rolling 평균값 (현재 윈도우의 avg)
        current_avg = avg_vol[i]
        fallback_support = np.inf
        fallback_resistance = -np.inf
        
        # window 내에서 순회하며 후보와 fallback 계산
        for j in range(start, i + 1):
            # fallback: 조건과 무관하게 window 전체의 min(low)와 max(high)
            if lows[j] < fallback_support:
                fallback_support = lows[j]
            if highs[j] > fallback_resistance:
                fallback_resistance = highs[j]
            # 후보 조건: 해당 캔들의 volume이 window의 평균 이상인 경우
            if volumes[j] >= current_avg:
                found_candidate = True
                if lows[j] < candidate_support:
                    candidate_support = lows[j]
                if highs[j] > candidate_resistance:
                    candidate_resistance = highs[j]
        # 후보가 존재하면 후보의 값 사용, 없으면 전체 window의 fallback 값 사용
        if found_candidate:
            support[i] = candidate_support
            resistance[i] = candidate_resistance
        else:
            support[i] = fallback_support
            resistance[i] = fallback_resistance
    return support, resistance

def calculate_support_resistance_optimized(df, window=20, extra_str=None):
    """
    OHLCV 데이터를 기반으로 rolling window 내에서 지지와 저항 레벨을 feature로 계산합니다.
    - 각 window 내에서 평균 거래량 이상인 캔들을 후보로 선정하고,
      * Support: 후보들의 최저 'Low'
      * Resistance: 후보들의 최고 'High'
    후보가 없으면 window 전체의 최솟값/최댓값을 사용합니다.
    또한, 현재 'Close' 가격과의 백분율 차이를 feature로 추가합니다.
    """
    # 컬럼 이름 생성
    def format_col_name(col_name, extra_str):
        return f"{col_name}_{extra_str}" if extra_str else col_name

    support_col = format_col_name(f"Support_{window}", extra_str)
    resistance_col = format_col_name(f"Resistance_{window}", extra_str)
    support_rel_col = format_col_name(f"Support_{window}_rel", extra_str)
    resistance_rel_col = format_col_name(f"Resistance_{window}_rel", extra_str)
    
    n = len(df)
    if n < window:
        df[support_col] = np.nan
        df[resistance_col] = np.nan
        df[support_rel_col] = np.nan
        df[resistance_rel_col] = np.nan
        return df, [support_col, resistance_col, support_rel_col, resistance_rel_col]
    
    # 필요한 배열 추출
    volumes = df['Volume'].to_numpy()
    lows = df['Low'].to_numpy()
    highs = df['High'].to_numpy()
    close_prices = df['Close'].to_numpy()
    
    # rolling 평균 거래량 계산: 첫 window-1 인덱스는 NaN 처리
    avg_vol = np.empty(n)
    for i in range(n):
        if i < window - 1:
            avg_vol[i] = np.nan
        else:
            avg_vol[i] = np.mean(volumes[i - window + 1:i + 1])
    
    # Numba 함수 호출하여 지원과 저항 계산
    support, resistance = compute_support_resistance_numba(volumes, lows, highs, avg_vol, window)
    
    # 현재 종가와의 상대적 백분율 차이 계산
    support_rel = (close_prices - support) / close_prices * 100
    resistance_rel = (resistance - close_prices) / close_prices * 100
    
    # DataFrame에 결과 컬럼 추가
    df[support_col] = support
    df[resistance_col] = resistance
    df[support_rel_col] = support_rel
    df[resistance_rel_col] = resistance_rel
    
    new_cols = [support_col, resistance_col, support_rel_col, resistance_rel_col]
    return df, new_cols




def base_feature_fn(df, extra_str=None):
    base_feature_list = []
    
    open_close_diff_col = format_col_name('open_close_diff', extra_str)
    open_high_diff_col = format_col_name('open_high_diff', extra_str)
    open_low_diff_col = format_col_name('open_low_diff', extra_str)
    close_high_diff_col = format_col_name('close_high_diff', extra_str)
    close_low_diff_col = format_col_name('close_low_diff', extra_str)
    high_low_diff_col = format_col_name('high_low_diff', extra_str)
    close_diff_col = format_col_name('close_diff', extra_str)
    
    df[open_close_diff_col] = (df['Open'] - df['Close']) / df['Open'] * 100
    df[open_high_diff_col] = (df['Open'] - df['High']) / df['Open'] * 100
    df[open_low_diff_col] = (df['Open'] - df['Low']) / df['Open'] * 100
    df[close_high_diff_col] = (df['Close'] - df['High']) / df['Close'] * 100
    df[close_low_diff_col] = (df['Close'] - df['Low']) / df['Close'] * 100
    df[high_low_diff_col] = (df['High'] - df['Low']) / df['High'] * 100
    df[close_diff_col] = (df['Close'] - df['Close'].shift(1)) / df['Close'] * 100
    
    base_feature_list.append(open_close_diff_col)
    base_feature_list.append(open_high_diff_col)
    base_feature_list.append(open_low_diff_col)
    base_feature_list.append(close_high_diff_col)
    base_feature_list.append(close_low_diff_col)
    base_feature_list.append(high_low_diff_col)
    base_feature_list.append(close_diff_col)
    base_feature_list.append('Volume')
    
    return df, base_feature_list




def cyclic_encode_fn(df, timestamp_col='Open time', cycle='minute_of_day'):
    if cycle == 'minute_of_day':
        # Extract the minute of the day
        df['minute_of_day'] = df[timestamp_col].dt.hour * 60 + df[timestamp_col].dt.minute
        
        # Apply sine and cosine transformations for a 24-hour cycle
        df['minute_of_day_sin'] = np.sin(2 * np.pi * df['minute_of_day'] / (24 * 60))
        df['minute_of_day_cos'] = np.cos(2 * np.pi * df['minute_of_day'] / (24 * 60))
        
        # List of new column names
        new_columns = ['minute_of_day_sin', 'minute_of_day_cos']
        
        # Drop the temporary 'minute_of_day' column
        df.drop(columns=['minute_of_day'], inplace=True)
        
    elif cycle == 'day_of_year':
        # Extract the day of the year
        df['day_of_year'] = df[timestamp_col].dt.dayofyear
        
        # Apply sine and cosine transformations for a 365-day cycle
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # List of new column names
        new_columns = ['day_of_year_sin', 'day_of_year_cos']
        
        # Drop the temporary 'day_of_year' column
        df.drop(columns=['day_of_year'], inplace=True)
        
    else:
        raise ValueError("Invalid cycle option. Choose 'minute_of_day' or 'day_of_year'.")
    
    return df, new_columns

def log_transform(data):
    return np.sign(data) * np.log1p(np.abs(data))
