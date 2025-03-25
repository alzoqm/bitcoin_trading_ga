import os
import torch
import numpy as np
from scipy.stats import skew, kurtosis
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from copy import deepcopy
from strategies import BBStrategy  # 본인이 사용 중이라면
from icecream import ic           # 디버깅용이라면

# ---------------------------------------------------
# [도움 함수] 날짜 차이 계산
# ---------------------------------------------------
def days_difference(date1, date2):
    difference = date2 - date1
    return np.abs(difference / np.timedelta64(1, 'D')).astype(int)


# ---------------------------------------------------
# [도움 함수] Activation 함수 (예: identity)
# ---------------------------------------------------
torch.set_grad_enabled(False)  # 전역적으로 기울기 계산 비활성화
def activation_fn(x):
    return x


# ---------------------------------------------------
# [데이터셋 정의] 1D, 1m 등에 대해 예시
# ---------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, data, data_1d):
        self.data = data
        self.data_1d = data_1d

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data_1d[idx]


# ---------------------------------------------------
# [추론 함수] Prescriptor 모델의 예시 forward
# ---------------------------------------------------
def inference(scaled_tensor, scaled_tensor_1d, model, device='cuda:0'):
    dataset = CustomDataset(scaled_tensor, scaled_tensor_1d)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=8, pin_memory=True)
    logits = []
    for data, data_1d in dataloader:
        data = data.to(torch.float32).to(device)
        data_1d = data_1d.to(torch.float32).to(device)
        logit = model.base_forward(data, data_1d)
        logits.append(logit)
    return logits


# ---------------------------------------------------
# [손절 함수] -> 롱/숏 수익 분리
# ---------------------------------------------------
def loss_cut_fn(pos_list,
                price_list,
                leverage_ratio,
                enter_ratio,
                profit_long,
                profit_short,
                curr_low,
                curr_high,
                additional_count,
                alpha=1.,
                cut_percent=80.):
    """
    지정된 cut_percent 이상의 손실이 발생하면 강제 청산.
    - 숏 포지션의 손익은 short_profit
    - 롱 포지션의 손익은 long_profit
    """
    # 포지션: short -> 1, long -> 2, hold -> 0
    short_index = torch.where(pos_list == 1)[0]
    long_index = torch.where(pos_list == 2)[0]

    # 숏 포지션 평가손익
    short_profit = -((curr_high - price_list[short_index]) / price_list[short_index] * 100.) * leverage_ratio[short_index]
    # 롱 포지션 평가손익
    long_profit = ((curr_low - price_list[long_index]) / price_list[long_index] * 100.) * leverage_ratio[long_index]

    # 손절 여부 판단
    short_cut_index = torch.where(short_profit <= -cut_percent)[0]
    long_cut_index = torch.where(long_profit <= -cut_percent)[0]

    # [숏] 강제청산
    short_index = short_index[short_cut_index]
    profit_short[short_index] += - (enter_ratio[short_index] * cut_percent * alpha) - 0.1 * leverage_ratio[short_index] * enter_ratio[short_index]

    pos_list[short_index] = 0
    price_list[short_index] = -1.
    leverage_ratio[short_index] = -1
    enter_ratio[short_index] = -1.
    additional_count[short_index] = 0

    # [롱] 강제청산
    long_index = long_index[long_cut_index]
    profit_long[long_index] += - (enter_ratio[long_index] * cut_percent * alpha) - 0.1 * leverage_ratio[long_index] * enter_ratio[long_index]

    pos_list[long_index] = 0
    price_list[long_index] = -1.
    leverage_ratio[long_index] = -1
    enter_ratio[long_index] = -1.
    additional_count[long_index] = 0

    return pos_list, price_list, leverage_ratio, enter_ratio, additional_count, profit_long, profit_short


# ---------------------------------------------------
# [액션 계산] -> 포지션 진입/청산/추가진입/반대포지션
# ---------------------------------------------------
def calculate_action(prob,
                     pos_list,
                     price_list,
                     leverage_ratio,
                     enter_ratio,
                     profit_long,
                     profit_short,
                     curr_close,
                     additional_count,
                     limit=2,
                     cut_value=1.0,
                     min_enter_ratio=0.05):
    """
    action = 0: hold
    action = 1: short
    action = 2: long
    """
    # 1) discrete action / continuous (enter_ratio, leverage)
    action = torch.argmax(prob[:, :3], dim=1)
    raw_enter_ratio = prob[:, 3]
    raw_leverage = prob[:, 4]

    # 2) 액션별 파라미터(진입비율, 레버리지) 처리
    enter_enter_ratio = activation_fn(raw_enter_ratio)
    enter_enter_ratio = torch.clamp(enter_enter_ratio, min=min_enter_ratio, max=cut_value)

    enter_leverage = activation_fn(raw_leverage) * 124.0
    enter_leverage_int = enter_leverage.int() + 1
    enter_leverage_int = torch.clamp(enter_leverage_int, 5, 125)

    # 현재 포지션 상태
    currently_hold  = (pos_list == 0)
    currently_short = (pos_list == 1)
    currently_long  = (pos_list == 2)

    # ------------------------------------------------
    # (1) hold action인 경우 -> 기존 숏/롱 포지션 청산
    # ------------------------------------------------
    hold_index = (action == 0)

    # [숏 -> hold] 청산
    close_short_idx = torch.where(currently_short & hold_index)[0]
    if len(close_short_idx) > 0:
        realized_pnl = (price_list[close_short_idx] - curr_close) / price_list[close_short_idx] * 100.0
        realized_pnl = realized_pnl * leverage_ratio[close_short_idx] * enter_ratio[close_short_idx]
        fee = 0.1 * leverage_ratio[close_short_idx] * enter_ratio[close_short_idx]

        profit_short[close_short_idx] += (realized_pnl - fee)

        pos_list[close_short_idx] = 0
        price_list[close_short_idx] = -1.0
        leverage_ratio[close_short_idx] = -1
        enter_ratio[close_short_idx] = -1.0
        additional_count[close_short_idx] = 0

    # [롱 -> hold] 청산
    close_long_idx = torch.where(currently_long & hold_index)[0]
    if len(close_long_idx) > 0:
        realized_pnl = (curr_close - price_list[close_long_idx]) / price_list[close_long_idx] * 100.0
        realized_pnl = realized_pnl * leverage_ratio[close_long_idx] * enter_ratio[close_long_idx]
        fee = 0.1 * leverage_ratio[close_long_idx] * enter_ratio[close_long_idx]

        profit_long[close_long_idx] += (realized_pnl - fee)

        pos_list[close_long_idx] = 0
        price_list[close_long_idx] = -1.0
        leverage_ratio[close_long_idx] = -1
        enter_ratio[close_long_idx] = -1.0
        additional_count[close_long_idx] = 0

    # ------------------------------------------------
    # (2) 반대 포지션 전환: short -> long
    # ------------------------------------------------
    short_index = (action == 1)
    long_index  = (action == 2)

    flip_short_to_long_idx = torch.where(currently_short & long_index)[0]
    # flip은 (action==2)에서 prob[:,2]가 충분히 높은 경우 등 다양한 조건 가능
    flip_short_to_long_idx = flip_short_to_long_idx[prob[flip_short_to_long_idx, 2] >= 0.7]

    if len(flip_short_to_long_idx) > 0:
        realized_pnl = (price_list[flip_short_to_long_idx] - curr_close) / price_list[flip_short_to_long_idx] * 100.0
        realized_pnl = realized_pnl * leverage_ratio[flip_short_to_long_idx] * enter_ratio[flip_short_to_long_idx]
        fee = 0.1 * leverage_ratio[flip_short_to_long_idx] * enter_ratio[flip_short_to_long_idx]

        # 숏 종료 -> 숏 수익
        profit_short[flip_short_to_long_idx] += (realized_pnl - fee)

        # 새롭게 롱 포지션
        pos_list[flip_short_to_long_idx] = 2
        price_list[flip_short_to_long_idx] = curr_close
        leverage_ratio[flip_short_to_long_idx] = torch.clamp(enter_leverage_int[flip_short_to_long_idx], 5, 125)
        enter_ratio[flip_short_to_long_idx] = enter_enter_ratio[flip_short_to_long_idx]
        additional_count[flip_short_to_long_idx] = 0

    # ------------------------------------------------
    # (3) 반대 포지션 전환: long -> short
    # ------------------------------------------------
    flip_long_to_short_idx = torch.where(currently_long & short_index)[0]
    flip_long_to_short_idx = flip_long_to_short_idx[prob[flip_long_to_short_idx, 1] >= 0.7]

    if len(flip_long_to_short_idx) > 0:
        realized_pnl = (curr_close - price_list[flip_long_to_short_idx]) / price_list[flip_long_to_short_idx] * 100.0
        realized_pnl = realized_pnl * leverage_ratio[flip_long_to_short_idx] * enter_ratio[flip_long_to_short_idx]
        fee = 0.1 * leverage_ratio[flip_long_to_short_idx] * enter_ratio[flip_long_to_short_idx]

        # 롱 종료 -> 롱 수익
        profit_long[flip_long_to_short_idx] += (realized_pnl - fee)

        # 새롭게 숏 포지션
        pos_list[flip_long_to_short_idx] = 1
        price_list[flip_long_to_short_idx] = curr_close
        leverage_ratio[flip_long_to_short_idx] = torch.clamp(enter_leverage_int[flip_long_to_short_idx], 5, 125)
        enter_ratio[flip_long_to_short_idx] = enter_enter_ratio[flip_long_to_short_idx]
        additional_count[flip_long_to_short_idx] = 0

    # ------------------------------------------------
    # (4) 포지션 열기 또는 추가진입: short
    # ------------------------------------------------
    open_short_idx = torch.where(currently_hold & short_index)[0]
    if len(open_short_idx) > 0:
        pos_list[open_short_idx] = 1
        price_list[open_short_idx] = curr_close
        leverage_ratio[open_short_idx] = torch.clamp(enter_leverage_int[open_short_idx], 5, 125)
        enter_ratio[open_short_idx] = enter_enter_ratio[open_short_idx]
        additional_count[open_short_idx] = 0

    add_short_idx = torch.where(currently_short & short_index)[0]
    if len(add_short_idx) > 0:
        can_add_idx = add_short_idx[additional_count[add_short_idx] < limit]
        if len(can_add_idx) > 0:
            before_price = price_list[can_add_idx]
            before_ratio = enter_ratio[can_add_idx]
            add_ratio = enter_enter_ratio[can_add_idx]
            add_ratio = torch.minimum((cut_value - before_ratio), add_ratio)
            add_ratio = torch.clamp(add_ratio, min=0.0)

            after_price = (
                before_price * (before_ratio / (before_ratio + add_ratio)) +
                curr_close * (add_ratio / (before_ratio + add_ratio))
            )
            after_ratio = before_ratio + add_ratio
            after_ratio = torch.clamp(after_ratio, max=cut_value)

            price_list[can_add_idx] = after_price
            enter_ratio[can_add_idx] = after_ratio
            additional_count[can_add_idx] += 1

    # ------------------------------------------------
    # (5) 포지션 열기 또는 추가진입: long
    # ------------------------------------------------
    open_long_idx = torch.where(currently_hold & long_index)[0]
    if len(open_long_idx) > 0:
        pos_list[open_long_idx] = 2
        price_list[open_long_idx] = curr_close
        leverage_ratio[open_long_idx] = torch.clamp(enter_leverage_int[open_long_idx], 5, 125)
        enter_ratio[open_long_idx] = enter_enter_ratio[open_long_idx]
        additional_count[open_long_idx] = 0

    add_long_idx = torch.where(currently_long & long_index)[0]
    if len(add_long_idx) > 0:
        can_add_idx = add_long_idx[additional_count[add_long_idx] < limit]
        if len(can_add_idx) > 0:
            before_price = price_list[can_add_idx]
            before_ratio = enter_ratio[can_add_idx]
            add_ratio = enter_enter_ratio[can_add_idx]
            add_ratio = torch.minimum((cut_value - before_ratio), add_ratio)
            add_ratio = torch.clamp(add_ratio, min=0.0)

            after_price = (
                before_price * (before_ratio / (before_ratio + add_ratio)) +
                curr_close * (add_ratio / (before_ratio + add_ratio))
            )
            after_ratio = before_ratio + add_ratio
            after_ratio = torch.clamp(after_ratio, max=cut_value)

            price_list[can_add_idx] = after_price
            enter_ratio[can_add_idx] = after_ratio
            additional_count[can_add_idx] += 1

    return pos_list, price_list, leverage_ratio, enter_ratio, additional_count, profit_long, profit_short


# ---------------------------------------------------
# [floating PnL 계산 - 필요하다면 사용]
# ---------------------------------------------------
def calculate_now_profit(pos_list,
                         price_list,
                         leverage_ratio,
                         enter_ratio,
                         curr_price):
    """
    열린 포지션(미체결)에 대한 현재 평가손익
    """
    now_profit = torch.zeros_like(pos_list, dtype=torch.float32)
    short_index = torch.where(pos_list == 1)[0]
    long_index = torch.where(pos_list == 2)[0]

    # 숏 현재 평가손익
    short_profit = -((curr_price - price_list[short_index]) / price_list[short_index] * 100.) * leverage_ratio[short_index]
    short_profit = short_profit * enter_ratio[short_index]
    short_profit -= 0.1 * leverage_ratio[short_index] * enter_ratio[short_index]  # 수수료 반영

    # 롱 현재 평가손익
    long_profit = ((curr_price - price_list[long_index]) / price_list[long_index] * 100.) * leverage_ratio[long_index]
    long_profit = long_profit * enter_ratio[long_index]
    long_profit -= 0.1 * leverage_ratio[long_index] * enter_ratio[long_index]  # 수수료 반영

    now_profit[short_index] = short_profit
    now_profit[long_index] = long_profit

    return now_profit


# ---------------------------------------------------
# [after_forward] - Prescriptor 모델의 추가 후처리
# ---------------------------------------------------
def after_forward(model,
                  prob,
                  now_profit,
                  leverage_ratio,
                  enter_ratio,
                  pos_list,
                  device='cuda:0'):
    """
    예시: 현재 평가손익(now_profit) + (prob, 레버리지, 진입비율 등)을 합쳐서
    모델이 다시 후처리할 수 있게 함.
    """
    ch_size = len(now_profit)
    now_profit_tensor = now_profit.unsqueeze(dim=1)
    leverage_ratio_tensor = leverage_ratio.unsqueeze(dim=1).to(torch.float32)
    enter_ratio_tensor = enter_ratio.unsqueeze(dim=1)

    # pos_list를 카테고리화(0,1,2)에 따라 임베딩 식으로 사용할 수도 있음
    step = torch.arange(0, ch_size * 3, step=3, device=device)
    cate_x = pos_list + step  # 하나의 아이디어

    x = torch.cat([prob, now_profit_tensor, leverage_ratio_tensor, enter_ratio_tensor], dim=1)
    x = x.to(torch.float32).to(device)
    cate_x = cate_x.to(device).long()

    after_output = model.after_forward(x=x, x_cate=cate_x)
    return after_output.squeeze(dim=0)


# ---------------------------------------------------
# [최종 Fitness 계산을 위한 Metric 보조 함수 - 롱/숏 분리 후 병합]
# ---------------------------------------------------
def calculate_fitness_long_short(
    mean_returns_long, sharpe_long, sortino_long, profit_factors_long, win_rates_long, max_dd_long, compound_long,
    mean_returns_short, sharpe_short, sortino_short, profit_factors_short, win_rates_short, max_dd_short, compound_short
):
    """
    롱과 숏 각각의 Metric을 구한 뒤,
    최종 = (롱 메트릭 + 숏 메트릭) / 2 형태로 통합
    """
    # -1e9이면 유효하지 않은 값(거래 횟수 부족 등)
    invalid_long = (mean_returns_long == -1e9)
    invalid_short = (mean_returns_short == -1e9)

    # 둘 중 하나라도 -1e9이면 최종 -1e9 처리(즉, 무효)
    invalid_both = invalid_long | invalid_short

    # 각각 합산 후 /2
    mean_returns = (mean_returns_long + mean_returns_short) / 2
    sharpe = (sharpe_long + sharpe_short) / 2
    sortino = (sortino_long + sortino_short) / 2
    profit_factor = (profit_factors_long + profit_factors_short) / 2
    win_rate = (win_rates_long + win_rates_short) / 2
    max_drawdown = (max_dd_long + max_dd_short) / 2
    compound = (compound_long * compound_short) / 2

    # invalid인 경우 -1e9 대입
    mean_returns[invalid_both] = -1e9
    sharpe[invalid_both] = -1e9
    sortino[invalid_both] = -1e9
    profit_factor[invalid_both] = -1e9
    win_rate[invalid_both] = -1e9
    max_drawdown[invalid_both] = 1e9
    compound[invalid_both] = -1e9

    # shape: (N, 7)
    metrics = torch.stack([
        mean_returns,
        sharpe,
        sortino,
        profit_factor,
        win_rate,
        max_drawdown,
        compound
    ], dim=1)
    return metrics.cpu().numpy()


def calculate_fitness(metrics):
    chromosomes_size = len(metrics)
    
    def normalize_metric(metric, higher_is_better=True):
        valid_indices = metric != -1e9
        valid_metric = metric[valid_indices]
        if len(valid_metric) == 0:
            return np.zeros_like(metric)
        min_val = np.nanmin(valid_metric)
        max_val = np.nanmax(valid_metric)
        if min_val == max_val:
            normalized = np.ones_like(metric) if higher_is_better else np.zeros_like(metric)
        else:
            if higher_is_better:
                normalized = (metric - min_val) / (max_val - min_val + 1e-8)
            else:
                normalized = (max_val - metric) / (max_val - min_val + 1e-8)
        normalized[~valid_indices] = 0.0
        return normalized

    higher_is_better_list = [
        True,   # mean_returns
        True,   # sharpe_ratios
        True,   # sortino_ratios
        True,   # profit_factors
        True,   # win_rates
        False,  # max_drawdowns
        True    # cumulative_returns
    ]
    for index in range(len(higher_is_better_list)):
        metrics[:, index] = normalize_metric(metrics[:, index], higher_is_better=higher_is_better_list[index])

    weights = [
        0.1,   # mean_returns
        0.05,  # sharpe_ratios
        0.05,  # sortino_ratios
        0.05,  # profit_factors
        0.15,  # win_rates
        0.1,   # max_drawdowns
        0.6    # cumulative_returns
    ]
    fitness_values = np.zeros(chromosomes_size)
    for index in range(len(weights)):
        fitness_values += weights[index] * metrics[:, index]

    fitness_values[metrics[:, 0] == -1e9] = -1e9

    return fitness_values

# ---------------------------------------------------
# [fitness_fn] - 롱/숏 분리 + 시뮬레이션
# ---------------------------------------------------
def fitness_fn(prescriptor,
               data,
               probs,
               entry_index_list,
               entry_pos_list,
               skip_data_cnt,
               start_data_cnt,
               chromosomes_size,
               window_size,
               alpha=1.,
               cut_percent=90.,
               device='cpu',
               stop_cnt=1e9,
               profit_init=10,
               limit=4,
               minimum_date=40):
    """
    각 chromosome별 롱/숏 포지션의 거래 통계를 누적한 후, 
    평균 수익률, Sharpe, Sortino Ratio (하방 위험 기반) 등 여러 metric을 계산하여
    calculate_fitness_long_short()를 통해 최종 통계를 반환합니다.
    """
    # 1) 포지션 관리 변수 초기화
    pos_list = torch.zeros(chromosomes_size, dtype=torch.long, device=device)  # 0: hold
    price_list = torch.full((chromosomes_size,), -1.0, dtype=torch.float32, device=device)
    leverage_ratio = torch.full((chromosomes_size,), -1, dtype=torch.int, device=device)
    enter_ratio = torch.full((chromosomes_size,), -1.0, dtype=torch.float32, device=device)
    additional_count = torch.zeros(chromosomes_size, dtype=torch.long, device=device)
    holding_period = torch.zeros(chromosomes_size, dtype=torch.long, device=device)
    
    # 2) 롱/숏 각각의 "이번에 실현된 수익"을 저장할 변수
    profit_long = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)
    profit_short = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)
    
    # 3) 롱/숏 누적 통계를 위한 변수들
    # [롱]
    sum_returns_long = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)
    count_returns_long = torch.zeros(chromosomes_size, dtype=torch.int32, device=device)
    sum_sq_returns_long = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)
    compound_value_long = torch.ones(chromosomes_size, dtype=torch.float32, device=device)
    running_sum_long = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)
    running_max_long = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)
    max_drawdown_long = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)
    win_count_long = torch.zeros(chromosomes_size, dtype=torch.int32, device=device)
    total_loss_long = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)
    total_profit_long = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)
    # 하방 위험 누적 (음수 거래)
    sum_sq_downside_long = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)
    count_downside_long = torch.zeros(chromosomes_size, dtype=torch.int32, device=device)
    
    # [숏]
    sum_returns_short = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)
    count_returns_short = torch.zeros(chromosomes_size, dtype=torch.int32, device=device)
    sum_sq_returns_short = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)
    compound_value_short = torch.ones(chromosomes_size, dtype=torch.float32, device=device)
    running_sum_short = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)
    running_max_short = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)
    max_drawdown_short = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)
    win_count_short = torch.zeros(chromosomes_size, dtype=torch.int32, device=device)
    total_loss_short = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)
    total_profit_short = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)
    # 하방 위험 누적 (음수 거래)
    sum_sq_downside_short = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)
    count_downside_short = torch.zeros(chromosomes_size, dtype=torch.int32, device=device)
    
    risk_free_rate = 0.0

    entry_pos_mapping = {'hold': 0, 'short': 1, 'long': 2}
    entry_pos_list_int = [entry_pos_mapping[ep] for ep in entry_pos_list]

    before_index = 0

    # 시뮬레이션 Loop
    for data_cnt, (entry_index, entry_pos) in tqdm(enumerate(zip(entry_index_list, entry_pos_list_int)),
                                                   total=len(entry_pos_list_int)):
        if data_cnt >= stop_cnt:
            break
        if data_cnt < start_data_cnt:
            continue

        entry_pos = torch.tensor(entry_pos).long()
        x = data.iloc[entry_index]
        curr_open = torch.tensor(x['Open'], dtype=torch.float32, device=device)
        curr_close = torch.tensor(x['Close'], dtype=torch.float32, device=device)
        curr_high = torch.tensor(x['High'], dtype=torch.float32, device=device)
        curr_low = torch.tensor(x['Low'], dtype=torch.float32, device=device)

        # BB 밴드 사용 예시 (필요시 활용)
        upper = torch.tensor(x[f'Upper_BB_{window_size}'], dtype=torch.float32, device=device)
        lower = torch.tensor(x[f'Lower_BB_{window_size}'], dtype=torch.float32, device=device)

        history_x = data.iloc[before_index+1:entry_index+1]
        history_high = torch.tensor(history_x['High'].max(), dtype=torch.float32, device=device)
        history_low = torch.tensor(history_x['Low'].min(), dtype=torch.float32, device=device)

        # 1) 손절 체크
        pos_list, price_list, leverage_ratio, enter_ratio, additional_count, profit_long, profit_short = loss_cut_fn(
            pos_list,
            price_list,
            leverage_ratio,
            enter_ratio,
            profit_long,
            profit_short,
            history_low,
            history_high,
            additional_count,
            alpha,
            cut_percent
        )

        # 2) prescriptor 확률값 (skip_data_cnt를 반영한 인덱스 사용)
        prob = torch.tensor(probs[:, data_cnt - skip_data_cnt]).float().to(device)

        # 3) 현재 미체결 포지션 평가손익 계산
        now_profit = calculate_now_profit(pos_list, price_list, leverage_ratio, enter_ratio, curr_close)

        # 4) after_forward 통해 액션 prob 보정
        prob = after_forward(prescriptor, prob, now_profit, leverage_ratio, enter_ratio, pos_list, device=device)

        # 5) 액션 수행 (청산, 반대 포지션, 추가 진입 등)
        pos_list, price_list, leverage_ratio, enter_ratio, additional_count, profit_long, profit_short = calculate_action(
            prob,
            pos_list,
            price_list,
            leverage_ratio,
            enter_ratio,
            profit_long,
            profit_short,
            curr_close,
            additional_count,
            limit=limit,
            min_enter_ratio=0.05
        )

        # 6) 홀딩 기간 증가
        holding_period[pos_list != 0] += 1
        holding_period[pos_list == 0] = 0

        # 7) 시간 기반 강제 청산
        pos_list, price_list, leverage_ratio, enter_ratio, additional_count, profit_long, profit_short, holding_period = time_based_exit_fn(
            pos_list,
            price_list,
            leverage_ratio,
            enter_ratio,
            additional_count,
            profit_long,
            profit_short,
            curr_close,
            holding_period,
            max_holding_bars=100,
            device=device
        )

        # 8) 이번 스텝에서 실현된 롱/숏 수익을 누적 통계에 반영
        # [롱 수익 갱신]
        non_zero_long = profit_long != 0
        if non_zero_long.any():
            sum_returns_long[non_zero_long] += profit_long[non_zero_long]
            count_returns_long[non_zero_long] += 1
            sum_sq_returns_long[non_zero_long] += profit_long[non_zero_long] ** 2
            compound_value_long[non_zero_long] *= (1.0 + profit_long[non_zero_long] / 100.0)

            win_mask_long = (profit_long > 0)
            win_count_long[win_mask_long] += 1
            total_loss_long[profit_long < 0] += -(profit_long[profit_long < 0])
            total_profit_long[profit_long > 0] += profit_long[profit_long > 0]

            # 하방(음수) 거래 누적
            downside_mask_long = profit_long < 0
            if downside_mask_long.any():
                sum_sq_downside_long[downside_mask_long] += profit_long[downside_mask_long] ** 2
                count_downside_long[downside_mask_long] += 1

            profit_long[non_zero_long] = 0

        # [숏 수익 갱신]
        non_zero_short = profit_short != 0
        if non_zero_short.any():
            sum_returns_short[non_zero_short] += profit_short[non_zero_short]
            count_returns_short[non_zero_short] += 1
            sum_sq_returns_short[non_zero_short] += profit_short[non_zero_short] ** 2
            compound_value_short[non_zero_short] *= (1.0 + profit_short[non_zero_short] / 100.0)

            win_mask_short = (profit_short > 0)
            win_count_short[win_mask_short] += 1
            total_loss_short[profit_short < 0] += -(profit_short[profit_short < 0])
            total_profit_short[profit_short > 0] += profit_short[profit_short > 0]

            # 하방(음수) 거래 누적
            downside_mask_short = profit_short < 0
            if downside_mask_short.any():
                sum_sq_downside_short[downside_mask_short] += profit_short[downside_mask_short] ** 2
                count_downside_short[downside_mask_short] += 1

            profit_short[non_zero_short] = 0

        running_sum_long += profit_long
        running_max_long = torch.max(running_max_long, running_sum_long)
        current_dd_long = running_max_long - running_sum_long
        max_drawdown_long = torch.max(max_drawdown_long, current_dd_long)

        running_sum_short += profit_short
        running_max_short = torch.max(running_max_short, running_sum_short)
        current_dd_short = running_max_short - running_sum_short
        max_drawdown_short = torch.max(max_drawdown_short, current_dd_short)

        before_index = entry_index

    # -----------------------------
    # 최종 통계 계산
    # -----------------------------
    # [롱 통계]
    c_long = count_returns_long.float()
    mean_returns_long = torch.where(c_long > 0, sum_returns_long / c_long, torch.full_like(sum_returns_long, -1e9))
    variance_long = torch.where(c_long > 0, sum_sq_returns_long / c_long - mean_returns_long ** 2, torch.zeros_like(sum_returns_long))
    std_long = torch.sqrt(variance_long + 1e-9)
    sharpe_long = torch.where(std_long > 0, (mean_returns_long - risk_free_rate) / std_long, torch.full_like(mean_returns_long, -1e9))
    # 하방 표준편차 (음수 거래에 대한)
    downside_std_long = torch.sqrt(sum_sq_downside_long / (count_downside_long.float() + 1e-9) + 1e-9)
    default_sortino_value = 30.0  # 거래는 있었으나 음수 거래가 없는 경우에 사용할 기본값
    sortino_long = torch.where(
        c_long > 0,
        torch.where(
            count_downside_long > 0,
            (mean_returns_long - risk_free_rate) / downside_std_long,
            torch.full_like(mean_returns_long, default_sortino_value)
        ),
        torch.full_like(mean_returns_long, -1e9)
    )
    pf_long = torch.where(total_loss_long > 0, total_profit_long / (total_loss_long + 1e-9), torch.full_like(total_loss_long, -1e9))
    wr_long = torch.where(c_long > 0, win_count_long.float() / c_long, torch.full_like(c_long, -1e9))
    invalid_long = (c_long < minimum_date)
    mean_returns_long[invalid_long] = -1e9
    sharpe_long[invalid_long] = -1e9
    sortino_long[invalid_long] = -1e9
    pf_long[invalid_long] = -1e9
    wr_long[invalid_long] = -1e9
    max_drawdown_long[invalid_long] = 1e9
    compound_value_long[invalid_long] = -1e9

    # [숏 통계]
    c_short = count_returns_short.float()
    mean_returns_short = torch.where(c_short > 0, sum_returns_short / c_short, torch.full_like(sum_returns_short, -1e9))
    variance_short = torch.where(c_short > 0, sum_sq_returns_short / c_short - mean_returns_short ** 2, torch.zeros_like(sum_returns_short))
    std_short = torch.sqrt(variance_short + 1e-9)
    sharpe_short = torch.where(std_short > 0, (mean_returns_short - risk_free_rate) / std_short, torch.full_like(mean_returns_short, -1e9))
    downside_std_short = torch.sqrt(sum_sq_downside_short / (count_downside_short.float() + 1e-9) + 1e-9)
    sortino_short = torch.where(
        c_short > 0,
        torch.where(
            count_downside_short > 0,
            (mean_returns_short - risk_free_rate) / downside_std_short,
            torch.full_like(mean_returns_short, default_sortino_value)
        ),
        torch.full_like(mean_returns_short, -1e9)
    )
    pf_short = torch.where(total_loss_short > 0, total_profit_short / (total_loss_short + 1e-9), torch.full_like(total_loss_short, -1e9))
    wr_short = torch.where(c_short > 0, win_count_short.float() / c_short, torch.full_like(c_short, -1e9))
    invalid_short = (c_short < minimum_date)
    mean_returns_short[invalid_short] = -1e9
    sharpe_short[invalid_short] = -1e9
    sortino_short[invalid_short] = -1e9
    pf_short[invalid_short] = -1e9
    wr_short[invalid_short] = -1e9
    max_drawdown_short[invalid_short] = 1e9
    compound_value_short[invalid_short] = -1e9

    metrics = calculate_fitness_long_short(
        mean_returns_long, sharpe_long, sortino_long, pf_long, wr_long, max_drawdown_long, compound_value_long,
        mean_returns_short, sharpe_short, sortino_short, pf_short, wr_short, max_drawdown_short, compound_value_short
    )
    return metrics


# ---------------------------------------------------
# [시간 기반 청산 함수] -> 롱/숏 수익 분리
# ---------------------------------------------------
def time_based_exit_fn(pos_list,
                       price_list,
                       leverage_ratio,
                       enter_ratio,
                       additional_count,
                       profit_long,
                       profit_short,
                       curr_price,
                       holding_period,
                       max_holding_bars=100,
                       device='cpu'):
    """
    일정 기간 이상 보유한 포지션 강제 청산
    """
    close_indices = torch.where((holding_period > max_holding_bars) & (pos_list != 0))[0]
    if len(close_indices) > 0:
        # 숏 포지션 청산
        short_idx = close_indices[pos_list[close_indices] == 1]
        if len(short_idx) > 0:
            realized_pnl = (price_list[short_idx] - curr_price) / price_list[short_idx] * 100.0
            realized_pnl = realized_pnl * leverage_ratio[short_idx] * enter_ratio[short_idx]
            fee = 0.1 * leverage_ratio[short_idx] * enter_ratio[short_idx]

            profit_short[short_idx] += (realized_pnl - fee)

            pos_list[short_idx] = 0
            price_list[short_idx] = -1.0
            leverage_ratio[short_idx] = -1
            enter_ratio[short_idx] = -1.0
            additional_count[short_idx] = 0
            holding_period[short_idx] = 0

        # 롱 포지션 청산
        long_idx = close_indices[pos_list[close_indices] == 2]
        if len(long_idx) > 0:
            realized_pnl = (curr_price - price_list[long_idx]) / price_list[long_idx] * 100.0
            realized_pnl = realized_pnl * leverage_ratio[long_idx] * enter_ratio[long_idx]
            fee = 0.1 * leverage_ratio[long_idx] * enter_ratio[long_idx]

            profit_long[long_idx] += (realized_pnl - fee)

            pos_list[long_idx] = 0
            price_list[long_idx] = -1.0
            leverage_ratio[long_idx] = -1
            enter_ratio[long_idx] = -1.0
            additional_count[long_idx] = 0
            holding_period[long_idx] = 0

    return pos_list, price_list, leverage_ratio, enter_ratio, additional_count, profit_long, profit_short, holding_period


# ---------------------------------------------------
# [유전자 알고리즘 학습 루프(예시)] -> generation_valid
# ---------------------------------------------------
def generation_valid(data_1m,
                     dataset_1m,
                     dataset_1d,
                     prescriptor,
                     evolution,
                     skip_data_cnt,
                     valid_skip_data_cnt,
                     test_skip_data_cnt,
                     chromosomes_size,
                     window_size,
                     gen_loop,
                     best_size,
                     elite_size,
                     profit_init,
                     entry_index_list=None,
                     entry_pos_list=None,
                     best_profit=None,
                     best_chromosomes=None,
                     start_gen=0,
                     device='cuda:0',
                     warming_step=5):
    """
    예시: GA 학습 단계에서 generation별로 훈련/검증 로직
    """
    temp_dir = 'generation'
    os.makedirs(temp_dir, exist_ok=True)
    
    # ### MODIFIED: train profit을 저장할 변수 추가
    best_train_profit = None

    for gen_idx in range(start_gen, gen_loop):
        print(f'[Generation {gen_idx}]')

        # 1) Prescriptor로부터 각 시점별 action 확률 추론
        probs = inference(dataset_1m, dataset_1d, prescriptor, device)
        probs = torch.concat(probs, dim=1).squeeze(dim=2)

        # 2) 훈련 구간 Fitness
        train_metrics = fitness_fn(
            prescriptor=prescriptor,
            data=data_1m,
            probs=probs,
            entry_index_list=entry_index_list,
            entry_pos_list=entry_pos_list,
            skip_data_cnt=skip_data_cnt,
            start_data_cnt=skip_data_cnt,
            chromosomes_size=chromosomes_size,
            window_size=window_size,
            alpha=1,
            cut_percent=90,
            device=device,
            stop_cnt=valid_skip_data_cnt,
            profit_init=profit_init,
            limit=4
        )

        # 3) 웜업 기간 지나면 Valid 구간 확인 + best 업데이트
        if gen_idx >= warming_step:
            valid_metrics = fitness_fn(
                prescriptor=prescriptor,
                data=data_1m,
                probs=probs,
                entry_index_list=entry_index_list,
                entry_pos_list=entry_pos_list,
                skip_data_cnt=skip_data_cnt,
                start_data_cnt=valid_skip_data_cnt,
                chromosomes_size=chromosomes_size,
                window_size=window_size,
                alpha=1,
                cut_percent=90,
                device=device,
                stop_cnt=test_skip_data_cnt,
                profit_init=profit_init,
                limit=4
            )

            valid_metrics = torch.from_numpy(valid_metrics[:elite_size])
            valid_index = np.where((train_metrics[:elite_size][:, 6] > 3.0) & 
                                    (train_metrics[:elite_size][:, 4] > 0.6))[0]
            valid_metrics = valid_metrics[valid_index]
            
            # ### MODIFIED: train profit도 같은 valid_index로 필터링하여 저장
            train_metrics_tensor = torch.from_numpy(train_metrics[:elite_size])
            train_metrics_filtered = train_metrics_tensor[valid_index]

            if best_profit is None:
                best_profit = valid_metrics
                # ### MODIFIED: best_train_profit 초기화
                best_train_profit = train_metrics_filtered
                best_chromosomes, _, _, _ = evolution.flatten_chromosomes()
                best_chromosomes = torch.tensor(best_chromosomes[:elite_size])[valid_index].clone()
            else:
                chromosomes, _, _, _ = evolution.flatten_chromosomes()
                chromosomes = chromosomes[:elite_size][valid_index].clone()
                
                new_indices = [index for index, t in enumerate(valid_metrics) if t not in best_profit]
                
                new_fitness = deepcopy(valid_metrics[new_indices])
                new_chromosomes = chromosomes[new_indices]
                # ### MODIFIED: 새로운 train profit도 함께 업데이트
                new_train_profit = train_metrics_filtered[new_indices]
                
                best_profit = torch.concat([best_profit, new_fitness])
                best_chromosomes = torch.concat([best_chromosomes, torch.tensor(new_chromosomes)])
                best_train_profit = torch.concat([best_train_profit, new_train_profit])

            if len(best_chromosomes) > best_size:
                print('check_discard')
                valid_fitness = calculate_fitness(deepcopy(best_profit).numpy())
                elite_idx, elite_chromosomes = evolution.select_elite(torch.from_numpy(valid_fitness), best_chromosomes, best_size)

                best_profit = best_profit[elite_idx]
                best_chromosomes = elite_chromosomes
                # ### MODIFIED: best_train_profit도 elite에 맞게 필터링
                best_train_profit = best_train_profit[elite_idx]

        gen_data = {
            "generation": gen_idx,
            "prescriptor_state_dict": prescriptor.state_dict(),
            "best_profit": best_profit,
            "best_chromosomes": best_chromosomes,
            # ### MODIFIED: best_train_profit도 저장
            "best_train_profit": best_train_profit,
        }
        
        train_fitness = calculate_fitness(train_metrics)
        torch.save(gen_data, os.path.join(temp_dir, f'generation_{gen_idx}.pt')) 
        evolution.evolve(torch.from_numpy(train_fitness))
        prescriptor = prescriptor.to(device)
        
        del probs
    # ### MODIFIED: best_train_profit도 반환
    return best_chromosomes, best_profit, best_train_profit


# ---------------------------------------------------
# [테스트 구간 평가(예시)] -> generation_test
# ---------------------------------------------------
def generation_test(data_1m,
                    dataset_1m,
                    dataset_1d,
                    prescriptor,
                    skip_data_cnt,
                    start_data_cnt,
                    end_data_cnt,
                    chromosomes_size,
                    window_size,
                    profit_init,
                    entry_index_list=None,
                    entry_pos_list=None,
                    device='cuda:0'):
    """
    GA 완료 후 최종 Test 구간을 한 번 더 시뮬레이션
    """
    probs = inference(dataset_1m, dataset_1d, prescriptor, device)
    probs = torch.concat(probs, dim=1).squeeze(dim=2)

    final_metrics = fitness_fn(
        prescriptor=prescriptor,
        data=data_1m,
        probs=probs,
        entry_index_list=entry_index_list,
        entry_pos_list=entry_pos_list,
        skip_data_cnt=skip_data_cnt,
        start_data_cnt=start_data_cnt,
        chromosomes_size=chromosomes_size,
        window_size=window_size,
        alpha=1,
        cut_percent=90,
        device=device,
        stop_cnt=end_data_cnt,
        profit_init=profit_init,
        limit=4
    )

    return final_metrics