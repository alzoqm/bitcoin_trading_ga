
from tqdm import tqdm

# 기존 Bollinger Bands 전략
class BBStrategy:
    def __init__(self):
        self.lower_band_out = False
        self.upper_band_out = False

    def check_bound(self, Open, Close, Upper, Lower):
        self.upper_band_out = Open > Upper and Close > Upper
        self.lower_band_out = Open < Lower and Close < Lower

    def entry_on_return(self, current_open, current_close, upper_band, lower_band):
        if self.upper_band_out and (current_open <= upper_band or current_close <= upper_band):
            return 'short'
        elif self.lower_band_out and (current_open >= lower_band or current_close >= lower_band):
            return 'long'
        return 'hold'

# 기존 MACD 전략
class MACDStrategy:
    def __init__(self):
        self.short_above_long = False
        self.long_above_short = False

    def check_macd(self, short_ma, long_ma):
        self.short_above_long = short_ma > long_ma
        self.long_above_short = long_ma > short_ma

    def entry_on_crossover(self, short_ma, long_ma):
        if self.short_above_long and short_ma <= long_ma:
            return 'short'
        elif self.long_above_short and short_ma >= long_ma:
            return 'long'
        return 'hold'

# 추가 보조지표: EMA 전략
class EMAStrategy:
    def __init__(self):
        # 초기 상태: 이전 시점에 가격가 EMA 위에 있었는지 여부 (None이면 초기 상태)
        self.price_above_ema = None

    def check_price(self, price, ema):
        """
        현재 시점의 가격과 EMA 관계에 따라 상태 업데이트.
        """
        self.price_above_ema = price > ema

    def entry_on_cross(self, price, ema):
        """
        이전 상태와 현재 비교하여 교차가 발생하면 신호를 반환합니다.
        - 이전에 가격이 EMA 위였으나 현재 EMA 이하이면 'short'
        - 이전에 EMA 이하였으나 현재 EMA 이상이면 'long'
        - 그 외에는 'hold'
        """
        if self.price_above_ema is None:
            return 'hold'
        if self.price_above_ema and price <= ema:
            return 'short'
        elif (not self.price_above_ema) and price >= ema:
            return 'long'
        return 'hold'

# 추가 보조지표: RSI 전략
class RSIStrategy:
    def __init__(self, upper=70, lower=30):
        # 과매수/과매도 임계치
        self.upper = upper
        self.lower = lower
        self.overbought = False
        self.oversold = False

    def check_rsi(self, rsi):
        """
        현재 RSI 값에 따라 상태 업데이트.
        """
        self.overbought = rsi > self.upper
        self.oversold = rsi < self.lower

    def entry_on_cross(self, rsi):
        """
        이전 RSI 상태와 현재 RSI 값을 비교하여 신호를 반환합니다.
        - 과매도 상태였으나 RSI가 임계치 이상으로 회복하면 'long'
        - 과매수 상태였으나 RSI가 임계치 이하로 내려가면 'short'
        - 그 외에는 'hold'
        """
        if self.oversold and rsi >= self.lower:
            return 'long'
        elif self.overbought and rsi <= self.upper:
            return 'short'
        return 'hold'


def BB_MACD_fitness_fn(data, window_size=240, short_window_size=20, long_window_size=60):
    bb_str = BBStrategy()
    macd_str = MACDStrategy()
    entry_pos_list = []
    patience_list = []
    index_list = []
    patience = 0
    before_pos = 'hold'

    for index, x in tqdm(data.iterrows(), total=len(data)):
        curr_open = x['Open']
        curr_close = x['Close']
        upper = x[f'Upper_BB_{window_size}']
        lower = x[f'Lower_BB_{window_size}']
        short_ma = x[f'SMA_{short_window_size}']
        long_ma = x[f'SMA_{long_window_size}']

        # MACD 우선순위로 진입 포지션 결정
        entry_pos = macd_str.entry_on_crossover(short_ma, long_ma)
        if entry_pos == 'hold':
            entry_pos = bb_str.entry_on_return(curr_open, curr_close, upper, lower)

        entry_pos_list.append(entry_pos)
        bb_str.check_bound(curr_open, curr_close, upper, lower)
        macd_str.check_macd(short_ma, long_ma)

        if entry_pos == 'long':
            if before_pos == 'long':
                patience += 1
            elif before_pos == 'short':
                patience = 0
            before_pos = entry_pos
            index_list.append(index)
        elif entry_pos == 'short':
            if before_pos == 'short':
                patience += 1
            elif before_pos == 'long':
                patience = 0
            before_pos = entry_pos
            index_list.append(index)
        patience_list.append(patience)

    return entry_pos_list, patience_list, index_list

def simple_fitness_fn(data, window_size=240, short_window_size=20, long_window_size=60):
    entry_pos_list = []
    patience_list = []
    index_list = []
    patience = 0
    before_pos = 'hold'
    counter = 0  # 반복 횟수를 센다.

    for index, x in tqdm(data.iterrows(), total=len(data)):
        counter += 1

        # 10번째마다 매수 포인트 생성, 나머지는 'hold'
        if counter % 20 == 0:
            entry_pos = 'long'
        else:
            entry_pos = 'hold'
        entry_pos_list.append(entry_pos)

        # 연속 'long' 상태의 경우 patience 증가, 새 매수 신호가 아니면 0으로 초기화
        if entry_pos == 'long':
            if before_pos == 'long':
                patience += 1
            else:
                patience = 0
            before_pos = entry_pos
            index_list.append(index)
        else:
            # 'hold'인 경우, 특별한 처리 없이 그대로 둔다.
            pass

        patience_list.append(patience)

    return entry_pos_list, patience_list, index_list
    
def BB_fitness_fn(data, window_size=240):
    bb_str = BBStrategy()
    entry_pos_list = []
    patience_list = []
    index_list = []
    patience = 0
    before_pos = 'hold'
    for index, x in tqdm(data.iterrows(), total=len(data)):
        curr_open = x['Open']
        curr_close = x['Close']
        upper = x[f'Upper_BB_{window_size}']
        lower = x[f'Lower_BB_{window_size}']
        entry_pos = bb_str.entry_on_return(curr_open, curr_close, upper, lower)
        entry_pos_list.append(entry_pos)
        bb_str.check_bound(curr_open, curr_close, upper, lower)
        if entry_pos == 'long':
            if before_pos == 'long':
                patience += 1
            elif before_pos == 'short':
                patience = 0
            before_pos = entry_pos
            index_list.append(index)
        elif entry_pos == 'short':
            if before_pos == 'short':
                patience += 1
            elif before_pos == 'long':
                patience = 0
            before_pos = entry_pos
            index_list.append(index)
        patience_list.append(patience)
    return entry_pos_list, patience_list, index_list
# 통합 피트니스 함수: BB, MACD, EMA, RSI 전략을 순차적으로 사용하여 진입 포지션 결정
def BB_MACD_EMA_RSI_fitness_fn(data, window_size=240, short_window_size=20, long_window_size=60, ema_window=50):
    bb_str = BBStrategy()
    macd_str = MACDStrategy()
    ema_str = EMAStrategy()
    rsi_str = RSIStrategy()  # 기본 임계치는 과매수 70, 과매도 30
    
    entry_pos_list = []
    patience_list = []
    index_list = []
    patience = 0
    before_pos = 'hold'

    for index, x in tqdm(data.iterrows(), total=len(data)):
        curr_open = x['Open']
        curr_close = x['Close']
        upper = x[f'Upper_BB_{window_size}']
        lower = x[f'Lower_BB_{window_size}']
        short_ma = x[f'SMA_{short_window_size}']
        long_ma = x[f'SMA_{long_window_size}']
        # EMA 추가 보조지표 (예: EMA_50)
        ema_val = x[f'SMA_{ema_window}']
        # RSI 값 (컬럼명이 'RSI'라고 가정)
        rsi_val = x[f'RSI_{short_window_size}']

        # 1. MACD 전략 우선 적용
        entry_pos = macd_str.entry_on_crossover(short_ma, long_ma)
        # 2. MACD 신호가 'hold'이면 BB 전략 적용
        if entry_pos == 'hold':
            entry_pos = bb_str.entry_on_return(curr_open, curr_close, upper, lower)
        # 3. BB 신호가 'hold'이면 EMA 전략 적용
        # if entry_pos == 'hold':
        #     entry_pos = ema_str.entry_on_cross(curr_close, ema_val)
        # 4. EMA 신호가 'hold'이면 RSI 전략 적용
        if entry_pos == 'hold':
            entry_pos = rsi_str.entry_on_cross(rsi_val)

        entry_pos_list.append(entry_pos)

        # 각 전략의 상태 업데이트 (다음 시점 비교를 위해)
        bb_str.check_bound(curr_open, curr_close, upper, lower)
        macd_str.check_macd(short_ma, long_ma)
        # ema_str.check_price(curr_close, ema_val)
        rsi_str.check_rsi(rsi_val)

        # 진입 포지션이 바뀌거나 동일한 포지션 지속 시 patience 업데이트
        if entry_pos == 'long':
            if before_pos == 'long':
                patience += 1
            elif before_pos == 'short':
                patience = 0
            before_pos = entry_pos
            index_list.append(index)
        elif entry_pos == 'short':
            if before_pos == 'short':
                patience += 1
            elif before_pos == 'long':
                patience = 0
            before_pos = entry_pos
            index_list.append(index)
        patience_list.append(patience)

    return entry_pos_list, patience_list, index_list

class SupportResistanceStrategy:
    def __init__(self, lookback=20):
        # lookback: 지지/저항 산출에 사용할 과거 캔들 수
        self.lookback = lookback
        self.support_broken = False
        self.resistance_broken = False

    def calculate_levels(self, ohlcv_window):
        """
        ohlcv_window: 과거 N 기간의 데이터. 각 항목은 (open, high, low, close, volume) 튜플로 구성됨.
        거래량이 평균 이상인 캔들을 대상으로 지지와 저항 레벨을 도출.
        """
        if len(ohlcv_window) == 0:
            raise ValueError("ohlcv_window 데이터가 비어있습니다.")

        # 거래량 리스트 추출 및 평균 거래량 계산
        volumes = [candle[4] for candle in ohlcv_window]
        avg_volume = sum(volumes) / len(volumes)

        # 평균 이상의 거래량을 기록한 캔들의 저가와 고가 후보군 선정
        support_candidates = [candle[2] for candle in ohlcv_window if candle[4] >= avg_volume]
        resistance_candidates = [candle[1] for candle in ohlcv_window if candle[4] >= avg_volume]

        # 후보군이 없으면 전체 캔들 중 최소/최대값 사용
        support = min(support_candidates) if support_candidates else min(candle[2] for candle in ohlcv_window)
        resistance = max(resistance_candidates) if resistance_candidates else max(candle[1] for candle in ohlcv_window)

        return support, resistance

    def check_levels(self, current_open, current_close, support, resistance):
        """
        현재 캔들이 지지 또는 저항 레벨을 돌파했는지 확인.
        돌파했다면 내부 플래그를 설정.
        """
        # 가격이 저항 레벨 위에서 시작해 끝나면 저항 돌파로 간주
        self.resistance_broken = current_open > resistance and current_close > resistance
        # 가격이 지지 레벨 아래에서 시작해 끝나면 지지 돌파로 간주
        self.support_broken = current_open < support and current_close < support

    def entry_on_return(self, current_open, current_close, support, resistance):
        """
        돌파 후 가격이 다시 반등(또는 하락)하는 시점에서 매매 신호 반환.
        - 저항 돌파 후 가격이 저항 레벨 아래로 되돌아오면 short 신호.
        - 지지 돌파 후 가격이 지지 레벨 위로 되돌아오면 long 신호.
        그렇지 않으면 hold.
        """
        if self.resistance_broken and (current_open <= resistance or current_close <= resistance):
            return 'short'
        elif self.support_broken and (current_open >= support or current_close >= support):
            return 'long'
        return 'hold'