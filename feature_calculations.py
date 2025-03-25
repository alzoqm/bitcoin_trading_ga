# feature_calculations.py

import numpy as np
import pandas as pd
from numba import njit

def log_transform(data):
    return np.sign(data) * np.log1p(np.abs(data))

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
        data[relative_col_name] = log_transform((data['Close'] - data[col_name]) / data['Close'] * 100)
    elif mode == 'EMA':  # Exponential Moving Average
        col_name = format_col_name(f'EMA_{window}', extra_str)
        relative_col_name = format_col_name(f'EMA_{window}_rel', extra_str)
        data[col_name] = data['Close'].ewm(span=window, adjust=False).mean()
        data[relative_col_name] = log_transform((data['Close'] - data[col_name]) / data['Close'] * 100)
    elif mode == 'WMA':  # Weighted Moving Average
        col_name = format_col_name(f'WMA_{window}', extra_str)
        relative_col_name = format_col_name(f'WMA_{window}_rel', extra_str)
        weights = list(range(1, window + 1))
        data[col_name] = data['Close'].rolling(window=window).apply(
            lambda x: np.dot(x, weights) / sum(weights), raw=True
        )
        data[relative_col_name] = log_transform((data['Close'] - data[col_name]) / data['Close'] * 100)
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
    df[upper_col_rel] = log_transform((df[upper_col] - df['Close']) / df['Close'] * 100)
    df[lower_col_rel] = log_transform((df['Close'] - df[lower_col]) / df['Close'] * 100)
    
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
    df[rsi_col] = (rsi - 50) / 25
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
    df[macd_col] = log_transform(df[macd_col] / df['Close'] * 100)
    df[signal_col] = log_transform(df[signal_col] / df['Close'] * 100)
    
    new_cols.extend([macd_col, signal_col])
    
    return df, new_cols

def calculate_stochastic_oscillator(df, k_window=14, d_window=3, extra_str=None):
    new_cols = []
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    k_col = format_col_name(f'%K_{k_window}', extra_str)
    d_col = format_col_name(f'%D_{d_window}', extra_str)
    
    df[k_col] = (100 * ((df['Close'] - low_min) / (high_max - low_min)) - 50) / 25
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
    df[adx_col] = log_transform(adx) - 2.276703340200053
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
    df[wr_col] =(williams_r - 50) / 25
    new_cols.append(wr_col)
    
    return df, new_cols

@njit
def update_histogram_add(hist, price, volume, bin_edges, alpha_power):
    """
    hist: 히스토그램(누적 거래량)을 저장할 1D array
    price: 해당 봉의 typical price
    volume: 해당 봉의 거래량
    bin_edges: 히스토그램 구간 경계
    alpha_power: 시간 가중을 고려할 때 곱할 값(예: alpha^(w-1-k))
    """
    # 가격에 맞는 bin 인덱스 찾기
    # np.searchsorted(bin_edges, price, side='right') - 1
    idx = np.searchsorted(bin_edges, price) - 1
    if idx < 0:
        idx = 0
    elif idx >= len(hist):
        idx = len(hist) - 1
    
    # 가중치로 volume 반영
    hist[idx] += volume * alpha_power

@njit
def update_histogram_sub(hist, price, volume, bin_edges, alpha_power):
    """
    오래된 봉 제거 시 hist에서 빼기
    """
    idx = np.searchsorted(bin_edges, price) - 1
    if idx < 0:
        idx = 0
    elif idx >= len(hist):
        idx = len(hist) - 1
    
    hist[idx] -= volume * alpha_power
    if hist[idx] < 0:
        hist[idx] = 0  # 혹시 음수가 되면 0으로 보정

@njit
def find_poc(hist, bin_edges):
    """
    히스토그램 값이 최대인 bin을 찾아 그 구간의 중앙값을 반환
    """
    max_idx = 0
    max_val = hist[0]
    for i in range(1, len(hist)):
        if hist[i] > max_val:
            max_val = hist[i]
            max_idx = i
    # 구간 중앙값
    bin_center = (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2.0
    return bin_center

def calculate_sliding_volume_profile(
    df,
    window=20,
    alpha=0.9,
    bin_size=10,
    global_min_price=None,
    global_max_price=None,
    extra_str=None
):
    """
    고정 bin 범위를 사용한 슬라이딩(rolling) 방식 Volume Profile POC 계산
    
    df : pd.DataFrame 
        - 'High', 'Low', 'Close', 'Volume' 컬럼이 있어야 함
    window : int
        - 몇 개의 봉을 롤링 윈도우로 사용할지
    alpha : float
        - 시간 가중(오래된 봉 감쇠). 1에 가까울수록 감쇠가 작고, 0.5처럼 작으면 오래된 봉은 급격히 작아짐
    bin_size : float
        - 가격 구간 간격
    global_min_price, global_max_price : float, float
        - None이면, df 전체의 최저/최고를 이용해 bin 범위 설정
        - 특정 범위를 주면, 그 범위를 bin으로 사용
    """
    
    # 1) 먼저 전체 데이터에 대한 전역 min/max 설정
    if global_min_price is None:
        global_min_price = df['Low'].min()
    if global_max_price is None:
        global_max_price = df['High'].max()
    
    if global_min_price == global_max_price:
        raise ValueError("최저가와 최고가가 동일하여 bin을 생성할 수 없습니다.")
    
    # bin_edges 생성
    # bin_edges 예: [100, 110, 120, ... , 200] 식으로
    bin_edges = np.arange(global_min_price, global_max_price + bin_size, bin_size)
    # 히스토그램 array (bin_edges의 길이 - 1)
    hist = np.zeros(len(bin_edges) - 1, dtype=np.float64)
    
    # Typical price, volume, alpha_w (시간 가중용 보정값) 배열 미리 계산
    # alpha_w[i] = alpha^(window - 1 - i상대위치) 같은 로직으로 구현 가능
    # 여기서는 간단히: 가장 최근 봉일수록 alpha_power = alpha^0 = 1,
    #                   오래된 봉일수록 alpha^(window-1) 형태
    typical_prices = (df['High'] + df['Low'] + df['Close']) / 3.0
    volumes = df['Volume'].values
    
    # “가장 최근 봉” = 가중치 alpha^0 = 1
    # “w-1번째 봉(가장 오래된)” = alpha^(w-1)
    alpha_powers = np.array([alpha ** p for p in range(window)], dtype=np.float64)
    
    # POC를 저장할 배열
    poc_vals = [np.nan] * len(df)
    
    # 2) 초기 구간(처음 window개의 봉)에 대한 히스토그램 채우기
    #    i=0이 가장 오래된, i=window-1이 가장 최신
    if len(df) < window:
        print("데이터 길이가 window보다 짧아 계산 불가.")
        return df, []
    
    for k in range(window):
        price = typical_prices[k]
        volume = volumes[k]
        # 오래된 봉일수록 alpha^(큰 값)
        alpha_power = alpha_powers[window - 1 - k]
        update_histogram_add(hist, price, volume, bin_edges, alpha_power)
    
    # 첫 POC 계산
    poc_vals[window-1] = find_poc(hist, bin_edges)
    
    # 3) 슬라이딩
    #    새 봉(i) 추가 & i-window 봉 제거
    for i in range(window, len(df)):
        # 추가
        new_price = typical_prices[i]
        new_volume = volumes[i]
        update_histogram_add(hist, new_price, new_volume, bin_edges, alpha_powers[0])  
        # (새로 들어온 봉은 "가장 최근"이므로 alpha^0 = 1)
        
        # 제거
        old_price = typical_prices[i - window]
        old_volume = volumes[i - window]
        # 제거되는 봉은 window개 중 “가장 오래된” 것이었으므로 alpha^(window-1)
        update_histogram_sub(hist, old_price, old_volume, bin_edges, alpha_powers[window - 1])
        
        # 이제 기존에 들어있던 것들의 alpha 가중치도 한 단계씩 “더 오래된 봉” 쪽으로 이동시켜야 함
        # => hist 전체를 alpha로 곱하는 방식이 간단!
        #    모든 bin을 alpha로 한번에 곱하면,
        #    "가장 최근"이었던 건 alpha^1로, "가장 오래된"이었던 건 alpha^(w-1) -> alpha^w가 됨
        hist *= alpha
        
        # POC 계산
        poc_vals[i] = find_poc(hist, bin_edges)
    
    # DataFrame에 POC 컬럼 추가
    poc_col_name = format_col_name(f'SlidingPoC_{window}', extra_str)
    df[poc_col_name] = poc_vals
    
    return df, poc_col_name

def calculate_rs(data_1d, data_1m, col_name):
    # 1m 데이터의 날짜에 대해 1d 데이터의 전날 값을 사용하여 feature 생성
    data_1m['Date'] = data_1m['Close time'].dt.date
    data_1d['Date'] = data_1d['Close time'].dt.date
    data_1d_shifted = data_1d.set_index('Date').shift(1).reset_index()

    # 현재 가격 대비 POC의 상대적 거리
    diff_col_name = 'SlidingPoC_diff'
    data_1m = data_1m.merge(data_1d_shifted[['Date', col_name]], on='Date', how='left', suffixes=('', '_1d'))
    data_1m[diff_col_name] = log_transform((data_1m['Close'] - data_1m[f'{col_name}']) / data_1m['Close'] * 100)

    return data_1m, diff_col_name




def base_feature_fn(df, extra_str=None):
    base_feature_list = []
    
    open_close_diff_col = format_col_name('open_close_diff', extra_str)
    open_high_diff_col = format_col_name('open_high_diff', extra_str)
    open_low_diff_col = format_col_name('open_low_diff', extra_str)
    close_high_diff_col = format_col_name('close_high_diff', extra_str)
    close_low_diff_col = format_col_name('close_low_diff', extra_str)
    high_low_diff_col = format_col_name('high_low_diff', extra_str)
    close_diff_col = format_col_name('close_diff', extra_str)
    
    df[open_close_diff_col] = (df['Open'] - df['Close']) / df['Open']
    df[open_high_diff_col] = (df['Open'] - df['High']) / df['Open']
    df[open_low_diff_col] = (df['Open'] - df['Low']) / df['Open']
    df[close_high_diff_col] = (df['Close'] - df['High']) / df['Close']
    df[close_low_diff_col] = (df['Close'] - df['Low']) / df['Close']
    df[high_low_diff_col] = (df['High'] - df['Low']) / df['High']
    df[close_diff_col] = (df['Close'] - df['Close'].shift(1)) / df['Close']

    # df[open_close_diff_col] = log_transform((df['Open'] - df['Close']) / df['Open'])
    # df[open_high_diff_col] = log_transform((df['Open'] - df['High']) / df['Open'])
    # df[open_low_diff_col] = log_transform((df['Open'] - df['Low']) / df['Open'])
    # df[close_high_diff_col] = log_transform((df['Close'] - df['High']) / df['Close'])
    # df[close_low_diff_col] = log_transform((df['Close'] - df['Low']) / df['Close'])
    # df[high_low_diff_col] = log_transform((df['High'] - df['Low']) / df['High'])
    # df[close_diff_col] = log_transform((df['Close'] - df['Close'].shift(1)) / df['Close'])

    df['Volume'] = log_transform(df['Volume']) - 3.287480801178518


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


