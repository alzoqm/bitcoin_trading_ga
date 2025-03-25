import os

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import skew, kurtosis
from tqdm import tqdm
from strategies import BBStrategy
from icecream import ic
from copy import deepcopy

def days_difference(date1, date2):
    # 날짜 차이 계산
    difference = date2 - date1
    # 일수 반환
    return np.abs(difference / np.timedelta64(1, 'D')).astype(int)

import numpy as np
from scipy.stats import skew, kurtosis

# 전역적으로 기울기 계산 비활성화
torch.set_grad_enabled(False)

# def activation_fn(x: torch.Tensor, alpha=30):
#     # return (x / (alpha + abs(x)) + 1) / 2
#     return torch.sigmoid(x)

# Example activation function (identity for simplicity in this test).
def activation_fn(x):
    return x

def calculate_performance_metrics(returns_list, minimum_date=40):
    """
    Calculate performance metrics for each chromosome based on their returns.

    Args:
        returns_list (np.ndarray): An array of shape (chromosomes_size, time_steps) containing the returns.
        minimum_date (int): Minimum number of non-zero returns required to calculate metrics.

    Returns:
        np.ndarray: An array containing performance metrics for each chromosome.
    """
    chromosomes_size = returns_list.shape[0]

    # Initialize arrays to store performance metrics
    mean_returns = np.full(chromosomes_size, -1e9)
    sharpe_ratios = np.full(chromosomes_size, -1e9)
    sortino_ratios = np.full(chromosomes_size, -1e9)
    profit_factors = np.full(chromosomes_size, -1e9)
    win_rates = np.full(chromosomes_size, -1e9)
    max_drawdowns = np.full(chromosomes_size, 1e9)
    cumulative_returns = np.full(chromosomes_size, -1e9)  # Initialize Cumulative Returns

    risk_free_rate = 0.0  # Adjust as needed

    # Compute the number of non-zero returns for each chromosome
    num_non_zero = np.count_nonzero(returns_list != 0, axis=1)
    valid_chromosomes = num_non_zero > minimum_date

    if sum(valid_chromosomes) != 0:
        # Replace zeros with NaN for non-zero returns calculations
        non_zero_returns_list = np.where(returns_list != 0, returns_list, np.nan)

        # Compute mean returns and standard deviations
        mean_returns[valid_chromosomes] = np.nanmean(non_zero_returns_list[valid_chromosomes], axis=1)
        std_returns_i = np.nanstd(non_zero_returns_list[valid_chromosomes], axis=1) + 1e-9

        # Sharpe Ratios
        valid_std = (std_returns_i != 0) & (~np.isnan(std_returns_i))
        sharpe_ratios_subset = (mean_returns[valid_chromosomes] - risk_free_rate) / std_returns_i
        sharpe_ratios[valid_chromosomes] = np.where(valid_std, sharpe_ratios_subset, -1e9)
        sharpe_ratios = np.where(np.isnan(sharpe_ratios), -1e9, sharpe_ratios)

        # Max Drawdown
        cumulative_returns_raw = np.cumsum(returns_list, axis=1)
        running_max = np.maximum.accumulate(cumulative_returns_raw, axis=1)
        drawdowns = running_max - cumulative_returns_raw
        max_drawdowns[valid_chromosomes] = np.nanmax(drawdowns[valid_chromosomes], axis=1)

        # Sortino Ratios
        negative_returns = np.where(non_zero_returns_list < 0, non_zero_returns_list, np.nan)
        downside_std = np.nanstd(negative_returns[valid_chromosomes], axis=1) + 1e-9
        valid_downside_std = (downside_std != 0) & (~np.isnan(downside_std))
        sortino_ratios_subset = (mean_returns[valid_chromosomes] - risk_free_rate) / downside_std
        sortino_ratios[valid_chromosomes] = np.where(valid_downside_std, sortino_ratios_subset, -1e9)
        sortino_ratios = np.where(np.isnan(sortino_ratios), -1e9, sortino_ratios)

        # Profit Factor
        total_profit = np.nansum(np.where(non_zero_returns_list > 0, non_zero_returns_list, 0), axis=1)
        total_loss = -np.nansum(np.where(non_zero_returns_list < 0, non_zero_returns_list, 0), axis=1)
        valid_total_loss = (total_loss != 0) & (~np.isnan(total_loss))
        profit_factors[valid_chromosomes] = -1e9
        profit_factors[valid_chromosomes & valid_total_loss] = total_profit[valid_chromosomes & valid_total_loss] / (total_loss[valid_chromosomes & valid_total_loss] + 1e-9)
        profit_factors = np.where(np.isnan(profit_factors), -1e9, profit_factors)

        # Win Rate
        num_wins = np.nansum(np.where(non_zero_returns_list > 0, 1, 0), axis=1)
        num_trades = num_non_zero
        valid_num_trades = (num_trades != 0) & (~np.isnan(num_trades))
        win_rates[valid_chromosomes] = -1e9
        win_rates[valid_chromosomes & valid_num_trades] = num_wins[valid_chromosomes & valid_num_trades] / num_trades[valid_chromosomes & valid_num_trades]
        win_rates = np.where(np.isnan(win_rates), -1e9, win_rates)

        # Calculate Cumulative Returns
        initial_value = 1.0
        for idx in np.where(valid_chromosomes)[0]:
            clean_returns = returns_list[idx][returns_list[idx] != 0]
            current_value = initial_value
            for ret in clean_returns:
                current_value += current_value * (ret / 100.0)
            cumulative_returns[idx] = current_value

    # high_drawdown_indices = max_drawdowns >= 60
    # penalty = max_drawdowns[high_drawdown_indices] - 60
    # mean_returns[high_drawdown_indices] -= penalty
    # sharpe_ratios[high_drawdown_indices] -= penalty
    # sortino_ratios[high_drawdown_indices] -= penalty
    # profit_factors[high_drawdown_indices] -= penalty
    # win_rates[high_drawdown_indices] -= penalty
    # max_drawdowns[high_drawdown_indices] += penalty
    # cumulative_returns[high_drawdown_indices] -= penalty

    # Expand dimensions and concatenate
    metrics = np.concatenate([
        np.expand_dims(mean_returns, axis=1),
        np.expand_dims(sharpe_ratios, axis=1),
        np.expand_dims(sortino_ratios, axis=1),
        np.expand_dims(profit_factors, axis=1),
        np.expand_dims(win_rates, axis=1),
        np.expand_dims(max_drawdowns, axis=1),
        np.expand_dims(cumulative_returns, axis=1)  # Add Cumulative Returns
    ], axis=1)

    # print(metrics[np.where(high_drawdown_indices)[0]])
    
    return metrics

class CustomDataset(Dataset):
    def __init__(self, data, data_1d):
        self.data = data
        self.data_1d = data_1d

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data_1d[idx]

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

def loss_cut_fn(pos_list, price_list, leverage_ratio, enter_ratio, profit, curr_low, curr_high, additional_count, alpha=1., cut_percent=80.):

    # Positions: 'short' -> 1, 'long' -> 2, 'hold' -> 0
    short_index = torch.where(pos_list == 1)[0]
    long_index = torch.where(pos_list == 2)[0]

    # Calculate profit or loss
    short_profit = -((curr_high - price_list[short_index]) / price_list[short_index] * 100.) * leverage_ratio[short_index]
    long_profit = ((curr_low - price_list[long_index]) / price_list[long_index] * 100.) * leverage_ratio[long_index]
    
    # Determine positions to cut
    short_cut_index = torch.where(short_profit <= -cut_percent)[0]
    long_cut_index = torch.where(long_profit <= -cut_percent)[0]

    # Update state for short positions to be cut
    short_index = short_index[short_cut_index]
    profit[short_index] = profit[short_index] - (enter_ratio[short_index] * cut_percent * alpha) - 0.1 * leverage_ratio[short_index] * enter_ratio[short_index]
    pos_list[short_index] = 0
    price_list[short_index] = -1.
    leverage_ratio[short_index] = -1
    enter_ratio[short_index] = -1.
    additional_count[short_index] = 0

    # Update state for long positions to be cut
    long_index = long_index[long_cut_index]
    profit[long_index] = profit[long_index] - (enter_ratio[long_index] * cut_percent * alpha) - 0.1 * leverage_ratio[long_index] * enter_ratio[long_index]
    pos_list[long_index] = 0
    price_list[long_index] = -1.
    leverage_ratio[long_index] = -1
    enter_ratio[long_index] = -1.
    additional_count[long_index] = 0

    return pos_list, price_list, leverage_ratio, enter_ratio, additional_count, profit



def calculate_action(
    prob,                # shape: (N, 5)
    pos_list,            # current positions (0=hold, 1=short, 2=long)
    price_list,
    leverage_ratio,
    enter_ratio,
    profit,
    curr_close,
    additional_count,
    limit=2,             # maximum number of "add" actions allowed
    cut_value=1.0,
    min_enter_ratio=0.05
):
    """
    prob: (N, 5) 텐서
      - prob[:, :3] => (hold, short, long)에 대한 확률 분포
      - prob[:, 3]  => 진입 비중(enter_ratio)을 위한 연속값
      - prob[:, 4]  => 레버리지(1~125 사이)를 위한 연속값
    
    pos_list: 현재 보유 포지션 (0=hold, 1=short, 2=long)
    price_list: 각 포지션의 평균 진입가격
    leverage_ratio: 각 포지션의 레버리지
    enter_ratio: 각 포지션의 진입 비중
    profit: 누적 손익
    curr_close: 현재 가격(배치 단위로 동일 가정)
    additional_count: 동일 방향 'add'를 몇 번 했는지 기록
    limit: 'add' 횟수 제한
    cut_value: 최대 진입 비중
    min_enter_ratio: 최소 진입 비중
    """

    # ------------------------------------------------
    # 1) Parse discrete (action) and continuous (enter_ratio, leverage) from prob
    # ------------------------------------------------
    action = torch.argmax(prob[:, :3], dim=1)  # discrete action
    raw_enter_ratio = prob[:, 3]
    raw_leverage = prob[:, 4]
    
    # Activation & clamp
    enter_enter_ratio = activation_fn(raw_enter_ratio)
    enter_enter_ratio = torch.clamp(enter_enter_ratio, min=min_enter_ratio, max=cut_value)
    
    # 레버리지 범위 [1, 125]
    enter_leverage = activation_fn(raw_leverage) * 124.0   # [0, 124]
    enter_leverage_int = enter_leverage.int() + 1          # [1, 125]
    enter_leverage_int = torch.clamp(enter_leverage_int, 20, 125)

    # ------------------------------------------------
    # 2) Indices for each action
    # ------------------------------------------------
    hold_index  = (action == 0)
    short_index = (action == 1)
    long_index  = (action == 2)

    currently_hold  = (pos_list == 0)
    currently_short = (pos_list == 1)
    currently_long  = (pos_list == 2)

    # ------------------------------------------------
    # 3) Close positions if action=hold
    # ------------------------------------------------
    # 3-A) Close shorts -> move to hold
    close_short_idx = torch.where(currently_short & hold_index)[0]
    if len(close_short_idx) > 0:
        realized_pnl = (price_list[close_short_idx] - curr_close) / price_list[close_short_idx] * 100.0
        realized_pnl = realized_pnl * leverage_ratio[close_short_idx] * enter_ratio[close_short_idx]
        fee = 0.1 * leverage_ratio[close_short_idx] * enter_ratio[close_short_idx]
        profit[close_short_idx] += (realized_pnl - fee)

        pos_list[close_short_idx] = 0
        price_list[close_short_idx] = -1.0
        leverage_ratio[close_short_idx] = -1
        enter_ratio[close_short_idx] = -1.0
        additional_count[close_short_idx] = 0

    # 3-B) Close longs -> move to hold
    close_long_idx = torch.where(currently_long & hold_index)[0]
    if len(close_long_idx) > 0:
        realized_pnl = (curr_close - price_list[close_long_idx]) / price_list[close_long_idx] * 100.0
        realized_pnl = realized_pnl * leverage_ratio[close_long_idx] * enter_ratio[close_long_idx]
        fee = 0.1 * leverage_ratio[close_long_idx] * enter_ratio[close_long_idx]
        profit[close_long_idx] += (realized_pnl - fee)

        pos_list[close_long_idx] = 0
        price_list[close_long_idx] = -1.0
        leverage_ratio[close_long_idx] = -1
        enter_ratio[close_long_idx] = -1.0
        additional_count[close_long_idx] = 0

    # ------------------------------------------------
    # 3-1) Flip short->long (반대 포지션 전환: short -> long)
    #     조건: currently_short 상태이고 action이 long이면서, prob[:,2] >= 0.7
    # ------------------------------------------------
    flip_short_to_long_idx = torch.where(currently_short & long_index)[0]
    # 해당 인덱스 중, "long 확률이 0.7 이상"인 케이스만 필터
    flip_short_to_long_idx = flip_short_to_long_idx[prob[flip_short_to_long_idx, 2] >= 0.7]

    if len(flip_short_to_long_idx) > 0:
        # 먼저 short 포지션 청산
        realized_pnl = (price_list[flip_short_to_long_idx] - curr_close) / price_list[flip_short_to_long_idx] * 100.0
        realized_pnl = realized_pnl * leverage_ratio[flip_short_to_long_idx] * enter_ratio[flip_short_to_long_idx]
        fee = 0.1 * leverage_ratio[flip_short_to_long_idx] * enter_ratio[flip_short_to_long_idx]
        profit[flip_short_to_long_idx] += (realized_pnl - fee)

        # 바로 long 포지션 진입
        pos_list[flip_short_to_long_idx] = 2
        price_list[flip_short_to_long_idx] = curr_close
        leverage_ratio[flip_short_to_long_idx] = enter_leverage_int[flip_short_to_long_idx]
        leverage_ratio[flip_short_to_long_idx] = torch.clamp(leverage_ratio[flip_short_to_long_idx], 5, 125)
        enter_ratio[flip_short_to_long_idx] = enter_enter_ratio[flip_short_to_long_idx]
        additional_count[flip_short_to_long_idx] = 0

    # ------------------------------------------------
    # 3-2) Flip long->short (반대 포지션 전환: long -> short)
    #     조건: currently_long 상태이고 action이 short이면서, prob[:,1] >= 0.7
    # ------------------------------------------------
    flip_long_to_short_idx = torch.where(currently_long & short_index)[0]
    # 해당 인덱스 중, "short 확률이 0.7 이상"인 케이스만 필터
    flip_long_to_short_idx = flip_long_to_short_idx[prob[flip_long_to_short_idx, 1] >= 0.7]

    if len(flip_long_to_short_idx) > 0:
        # 먼저 long 포지션 청산
        realized_pnl = (curr_close - price_list[flip_long_to_short_idx]) / price_list[flip_long_to_short_idx] * 100.0
        realized_pnl = realized_pnl * leverage_ratio[flip_long_to_short_idx] * enter_ratio[flip_long_to_short_idx]
        fee = 0.1 * leverage_ratio[flip_long_to_short_idx] * enter_ratio[flip_long_to_short_idx]
        profit[flip_long_to_short_idx] += (realized_pnl - fee)

        # 바로 short 포지션 진입
        pos_list[flip_long_to_short_idx] = 1
        price_list[flip_long_to_short_idx] = curr_close
        leverage_ratio[flip_long_to_short_idx] = enter_leverage_int[flip_long_to_short_idx]
        leverage_ratio[flip_long_to_short_idx] = torch.clamp(leverage_ratio[flip_long_to_short_idx], 5, 125)
        enter_ratio[flip_long_to_short_idx] = enter_enter_ratio[flip_long_to_short_idx]
        additional_count[flip_long_to_short_idx] = 0

    # ------------------------------------------------
    # 4) Open/Add short
    # ------------------------------------------------
    # 4-A) 현재 포지션이 hold이고, action=short 일 때 open
    open_short_idx = torch.where(currently_hold & short_index)[0]
    if len(open_short_idx) > 0:
        pos_list[open_short_idx] = 1
        price_list[open_short_idx] = curr_close
        leverage_ratio[open_short_idx] = enter_leverage_int[open_short_idx]
        leverage_ratio[open_short_idx] = torch.clamp(leverage_ratio[open_short_idx], 5, 125)
        enter_ratio[open_short_idx] = enter_enter_ratio[open_short_idx]
        additional_count[open_short_idx] = 0

    # 4-B) 현재 포지션이 short이고, action=short 일 때 add (추가 진입)
    add_short_idx = torch.where(currently_short & short_index)[0]
    if len(add_short_idx) > 0:
        can_add_idx = add_short_idx[additional_count[add_short_idx] < limit]
        if len(can_add_idx) > 0:
            before_price = price_list[can_add_idx]
            before_ratio = enter_ratio[can_add_idx]
            add_ratio = enter_enter_ratio[can_add_idx]
            add_ratio = torch.minimum((cut_value - before_ratio), add_ratio)
            # add_ratio가 음수가 되지 않도록 clamp
            add_ratio = torch.clamp(add_ratio, min=0.0)

            after_price = (
                before_price * (before_ratio / (before_ratio + add_ratio)) +
                curr_close * (add_ratio / (before_ratio + add_ratio))
            )
            after_ratio = before_ratio + add_ratio
            after_ratio = torch.clamp(after_ratio, max=cut_value)

            price_list[can_add_idx] = after_price
            enter_ratio[can_add_idx] = after_ratio
            # (주의) 레버리지 합산 로직이 필요하다면 아래를 사용
            # new_lev = leverage_ratio[can_add_idx] + enter_leverage_int[can_add_idx]
            # leverage_ratio[can_add_idx] = torch.clamp(new_lev, 1, 125)

            additional_count[can_add_idx] += 1

    # ------------------------------------------------
    # 5) Open/Add long
    # ------------------------------------------------
    # 5-A) 현재 포지션이 hold이고, action=long 일 때 open
    open_long_idx = torch.where(currently_hold & long_index)[0]
    if len(open_long_idx) > 0:
        pos_list[open_long_idx] = 2
        price_list[open_long_idx] = curr_close
        leverage_ratio[open_long_idx] = enter_leverage_int[open_long_idx]
        leverage_ratio[open_long_idx] = torch.clamp(leverage_ratio[open_long_idx], 5, 125)
        enter_ratio[open_long_idx] = enter_enter_ratio[open_long_idx]
        additional_count[open_long_idx] = 0

    # 5-B) 현재 포지션이 long이고, action=long 일 때 add (추가 진입)
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
            # (주의) 레버리지 합산 로직이 필요하다면 아래를 사용
            # new_lev = leverage_ratio[can_add_idx] + enter_leverage_int[can_add_idx]
            # leverage_ratio[can_add_idx] = torch.clamp(new_lev, 1, 125)

            additional_count[can_add_idx] += 1

    return pos_list, price_list, leverage_ratio, enter_ratio, additional_count, profit

def time_based_exit_fn(
    pos_list,
    price_list,
    leverage_ratio,
    enter_ratio,
    additional_count,
    profit,
    curr_price,
    holding_period,
    max_holding_bars,
    device='cpu'
):
    """
    일정 기간(여기서는 max_holding_bars) 이상 보유한 포지션을 강제로 청산한다.
    """
    # 청산 대상 인덱스: 현재 보유 기간이 max_holding_bars를 초과하고, 포지션이 열려 있는 경우
    close_indices = torch.where((holding_period > max_holding_bars) & (pos_list != 0))[0]
    if len(close_indices) > 0:
        # ---------------------------
        # short 포지션 강제청산
        # ---------------------------
        short_idx = close_indices[pos_list[close_indices] == 1]
        if len(short_idx) > 0:
            # 청산 손익 계산
            realized_pnl = (price_list[short_idx] - curr_price) / price_list[short_idx] * 100.0
            realized_pnl = realized_pnl * leverage_ratio[short_idx] * enter_ratio[short_idx]
            fee = 0.1 * leverage_ratio[short_idx] * enter_ratio[short_idx]
            profit[short_idx] += (realized_pnl - fee)
            # 포지션 종료 처리
            pos_list[short_idx] = 0
            price_list[short_idx] = -1.0
            leverage_ratio[short_idx] = -1
            enter_ratio[short_idx] = -1.0
            additional_count[short_idx] = 0
            holding_period[short_idx] = 0

        # ---------------------------
        # long 포지션 강제청산
        # ---------------------------
        long_idx = close_indices[pos_list[close_indices] == 2]
        if len(long_idx) > 0:
            realized_pnl = (curr_price - price_list[long_idx]) / price_list[long_idx] * 100.0
            realized_pnl = realized_pnl * leverage_ratio[long_idx] * enter_ratio[long_idx]
            fee = 0.1 * leverage_ratio[long_idx] * enter_ratio[long_idx]
            profit[long_idx] += (realized_pnl - fee)
            pos_list[long_idx] = 0
            price_list[long_idx] = -1.0
            leverage_ratio[long_idx] = -1
            enter_ratio[long_idx] = -1.0
            additional_count[long_idx] = 0
            holding_period[long_idx] = 0

    return pos_list, price_list, leverage_ratio, enter_ratio, additional_count, profit, holding_period

def calculate_now_profit(pos_list, price_list, leverage_ratio, enter_ratio, curr_price):
    now_profit = torch.zeros_like(pos_list, dtype=torch.float32)
    short_index = torch.where(pos_list == 1)[0]  # 'short' -> 1
    long_index = torch.where(pos_list == 2)[0]  # 'long' -> 2

    short_profit = (-((curr_price - price_list[short_index]) / price_list[short_index] * 100.) * leverage_ratio[short_index]) - 0.1 * leverage_ratio[short_index] * enter_ratio[short_index]
    long_profit = (((curr_price - price_list[long_index]) / price_list[long_index] * 100.) * leverage_ratio[long_index]) - 0.1 * leverage_ratio[long_index] * enter_ratio[long_index]
    short_profit = short_profit * enter_ratio[short_index]
    long_profit = long_profit * enter_ratio[long_index]
    now_profit[short_index] = short_profit
    now_profit[long_index] = long_profit

    return now_profit



def after_forward(model, prob, now_profit, leverage_ratio, enter_ratio, pos_list, device):
    ch_size = len(now_profit)
    now_profit_tensor = now_profit.unsqueeze(dim=1)
    leverage_ratio_tensor = leverage_ratio.unsqueeze(dim=1).to(torch.float32)
    enter_ratio_tensor = enter_ratio.unsqueeze(dim=1)
    mapping = {0: 0, 1: 1, 2: 2}  # Adjusted mapping
    mapped_array = pos_list
    step = torch.arange(0, ch_size * 3, step=3, device=device)

    x = torch.cat([prob, now_profit_tensor, leverage_ratio_tensor, enter_ratio_tensor], dim=1)
    cate_x = mapped_array + step

    x = x.to(torch.float32).to(device)
    cate_x = cate_x.to(device).long()

    after_output = model.after_forward(x=x.squeeze(dim=0), x_cate=cate_x)
    return after_output.squeeze(dim=0)



def calculate_fitness(metrics):
    chromosomes_size = len(metrics)
    
    # Normalize metrics
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
        normalized[~valid_indices] = 0.0  # Assign zero to invalid entries
        return normalized

    higher_is_better_list = [
        True,   # 'mean_returns'
        True,   # 'sharpe_ratios'
        True,   # 'sortino_ratios'
        True,   # 'profit_factors'
        True,   # 'win_rates'
        False,  # 'max_drawdowns'
        True    # cumulative_returns
    ]
    for index in range(len(higher_is_better_list)):
        metrics[:, index] = normalize_metric(metrics[:, index], higher_is_better=higher_is_better_list[index])

    # weights 배열을 metrics 순서에 맞춰서 재정렬
    # # 수익 위주의 학습
    # weights = [
    #     0.2,  # mean_returns: 0
    #     0.05,  # sharpe_ratios: 1
    #     0.10,  # sortino_ratios: 2
    #     0.10,  # profit_factors: 3
    #     0.15,  # win_rates: 4
    #     0.2,  # max_drawdowns: 5
    #     0.2  # cumulative_returns
    # ]
    # 승률 위주의 학습
    weights = [
        0.1,  # mean_returns: 0
        0.05,  # sharpe_ratios: 1
        0.05,  # sortino_ratios: 2
        0.05,  # profit_factors: 3
        0.15,  # win_rates: 4
        0.1,  # max_drawdowns: 5
        0.6  # cumulative_returns
    ]
    # Calculate the final fitness values
    fitness_values = np.zeros(chromosomes_size)
    for index in range(len(weights)):
        fitness_values += weights[index] * metrics[:, index]

    # Penalize chromosomes with invalid fitness
    fitness_values[metrics[:, 0] == -1e9] = -1e9

    return fitness_values

def fitness_fn(prescriptor, data, probs, entry_index_list, entry_pos_list, skip_data_cnt, start_data_cnt, chromosomes_size, window_size,
               alpha=1., cut_percent=90., device='cpu', stop_cnt=1e9, profit_init=10, limit=4, minimum_date=40):
    # Initialize variables
    if stop_cnt != 1e9:
        simulation_date = days_difference(data.iloc[entry_index_list[start_data_cnt]]['Open time'], data.iloc[entry_index_list[stop_cnt]]['Open time'])
    else:
        simulation_date = days_difference(data.iloc[entry_index_list[start_data_cnt]]['Open time'], data.iloc[-1]['Open time'])
    ic(simulation_date)
    pos_list = torch.zeros(chromosomes_size, dtype=torch.long, device=device)  # 0: 'hold'
    price_list = torch.full((chromosomes_size,), -1.0, dtype=torch.float32, device=device)
    leverage_ratio = torch.full((chromosomes_size,), -1, dtype=torch.int, device=device)
    enter_ratio = torch.full((chromosomes_size,), -1.0, dtype=torch.float32, device=device)
    profit = torch.zeros((chromosomes_size,), dtype=torch.float32, device=device)
    additional_count = torch.zeros(chromosomes_size, dtype=torch.long, device=device)
    holding_period = torch.zeros(chromosomes_size, dtype=torch.long, device=device)
    returns_list = []
    before_index = 0

    # Map entry positions
    entry_pos_mapping = {'hold': 0, 'short': 1, 'long': 2}
    entry_pos_list_int = [entry_pos_mapping[ep] for ep in entry_pos_list]

    for data_cnt, (entry_index, entry_pos) in tqdm(enumerate(zip(entry_index_list, entry_pos_list_int)), total=len(entry_pos_list_int)):
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
        upper = torch.tensor(x[f'Upper_BB_{window_size}'], dtype=torch.float32, device=device)
        lower = torch.tensor(x[f'Lower_BB_{window_size}'], dtype=torch.float32, device=device)
        
        history_x = data.iloc[before_index+1:entry_index+1]
        history_high = torch.tensor(history_x['High'].max(), dtype=torch.float32, device=device)
        history_low = torch.tensor(history_x['Low'].min(), dtype=torch.float32, device=device)
        
        pos_list, price_list, leverage_ratio, enter_ratio, additional_count, profit = loss_cut_fn(
            pos_list, price_list, leverage_ratio,
            enter_ratio, profit, history_low, history_high,
            additional_count, alpha, cut_percent
        )
        
        # if data_cnt < start_data_cnt:
        #     continue
        prob = torch.tensor(probs[:, data_cnt-skip_data_cnt]).to(device)
        
        now_profit = calculate_now_profit(pos_list, price_list, leverage_ratio, enter_ratio, curr_close)
        prob = after_forward(prescriptor, prob, now_profit, leverage_ratio, enter_ratio, pos_list, device=device)
        
        # 3) Decide how to update positions based on the new unified function
        pos_list, price_list, leverage_ratio, enter_ratio, additional_count, profit = calculate_action(
            prob, pos_list, price_list, leverage_ratio, enter_ratio,
            profit, curr_close, additional_count,
            limit=limit, min_enter_ratio=0.05
        )
        
        # -------------------------
        # (1) 보유 기간 추적 로직
        # -------------------------
        # 포지션이 열려 있으면 +1, 포지션이 0이면 보유 기간 0으로 초기화
        holding_period[pos_list != 0] += 1
        holding_period[pos_list == 0] = 0

        # # -------------------------
        # # (2) 최대 보유 기간 초과 시 강제 청산
        # -------------------------
        pos_list, price_list, leverage_ratio, enter_ratio, additional_count, profit, holding_period = time_based_exit_fn(
            pos_list,
            price_list,
            leverage_ratio,
            enter_ratio,
            additional_count,
            profit,
            curr_close,
            holding_period,
            max_holding_bars=100,  # 인자로 전달
            device=device
        )


        before_index = entry_index
        returns_list.append(profit.clone().cpu().detach().numpy())
        profit = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)

    returns_list = np.array(returns_list).T  # Shape: (chromosomes_size, time_steps)

    # Calculate performance metrics
    metrics = calculate_performance_metrics(returns_list, minimum_date=3)
    # metrics = calculate_performance_metrics(returns_list, minimum_date=3)

    return metrics

def get_chromosome_key(chromosome):
    # Quantize the chromosome values to 6 decimal places to handle floating-point precision
    quantized_chrom = np.round(chromosome.cpu().numpy(), decimals=6)
    # Convert to tuple to make it hashable
    return tuple(quantized_chrom.flatten())

def generation_valid(data_1m, dataset_1m, dataset_1d, prescriptor, evolution,
                     skip_data_cnt, valid_skip_data_cnt, test_skip_data_cnt, chromosomes_size,
                     window_size, gen_loop, best_size, elite_size, profit_init, 
                     entry_index_list=None, entry_pos_list=None,
                     best_profit=None, best_chromosomes=None, start_gen=0, device='cuda:0',
                     warming_step=5):
    
    best_profit = best_profit
    best_chromosomes = best_chromosomes
    # Create a temporary folder to save the generation data
    temp_dir = 'generation'
    os.makedirs(temp_dir, exist_ok=True)
    
    for gen_idx in range(start_gen, gen_loop):
        print(f'generation  {gen_idx}: ')

        probs = inference(dataset_1m, dataset_1d, prescriptor, device)
        probs = torch.concat(probs, dim=1)
        probs = probs.squeeze(dim=2)

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
        # profit = np.concatenate([profit]).T
        if warming_step <= gen_idx:
            if gen_idx != 0:
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
                                    #    (train_metrics[:elite_size][:, 5] < 60) & 
                                       (train_metrics[:elite_size][:, 4] > 0.6))[0]
                valid_metrics = valid_metrics[valid_index]
                
                # Initialize best fitness and chromosomes if not already done
                if best_profit is None:
                    best_profit = valid_metrics
                    best_chromosomes, _, _, _ = evolution.flatten_chromosomes()
                    best_chromosomes = torch.tensor(best_chromosomes[:elite_size])[valid_index].clone()
                else:
                    # Flatten current chromosomes
                    chromosomes, _, _, _ = evolution.flatten_chromosomes()
                    chromosomes = chromosomes[:elite_size][valid_index].clone()
                    
                    # Find indices of new fitness values not in best fitness
                    new_indices = [index for index, t in enumerate(valid_metrics) if t not in best_profit]
                    
                    # Update fitness and chromosomes with new values
                    new_fitness = deepcopy(valid_metrics[new_indices])
                    new_chromosomes = chromosomes[new_indices]
                    
                    best_profit = torch.concat([best_profit, new_fitness])
                    best_chromosomes = torch.concat([best_chromosomes, torch.tensor(new_chromosomes)])

                if len(best_chromosomes) > best_size:
                    print('check_discard')
                    valid_fitness = calculate_fitness(deepcopy(best_profit).numpy())
                    # Select elite chromosomes based on best fitness
                    elite_idx, elite_chromosomes = evolution.select_elite(torch.from_numpy(valid_fitness), best_chromosomes, best_size)

                    # Update best fitness and chromosomes with elite values
                    best_profit = best_profit[elite_idx]
                    best_chromosomes = elite_chromosomes

            

        # Save current generation values to a file
        gen_data = {
            "generation": gen_idx,
            "prescriptor_state_dict": prescriptor.state_dict(),
            "best_profit": best_profit,
            "best_chromosomes": best_chromosomes,

        }
        
        
        train_fitness = calculate_fitness(train_metrics)
        # torch.save(gen_data, os.path.join(temp_dir, f'generation_{gen_idx}.pt')) 
        torch.save(gen_data, os.path.join(temp_dir, f'generation_{gen_idx}.pt')) 
        evolution.evolve(torch.from_numpy(train_fitness))
        prescriptor = prescriptor.to(device)
        
        del probs
    return best_chromosomes, best_profit

def generation_test(data_1m, dataset_1m, dataset_1d, prescriptor, skip_data_cnt,
                     start_data_cnt, end_data_cnt, chromosomes_size,
                     window_size, profit_init, 
                     entry_index_list=None, entry_pos_list=None, device='cuda:0'):
    
    probs = inference(dataset_1m, dataset_1d, prescriptor, device)
    probs = torch.concat(probs, dim=1)
    probs = probs.squeeze(dim=2)
    
    profit = fitness_fn(
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
        
       
    return profit