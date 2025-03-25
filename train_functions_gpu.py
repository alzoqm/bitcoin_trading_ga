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
    valid_chromosomes = num_non_zero > (minimum_date // 3)

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

    high_drawdown_indices = max_drawdowns >= 60
    mean_returns[high_drawdown_indices] /= 2
    sharpe_ratios[high_drawdown_indices] /= 2
    sortino_ratios[high_drawdown_indices] /= 2
    profit_factors[high_drawdown_indices] /= 2
    win_rates[high_drawdown_indices] /= 2
    cumulative_returns[high_drawdown_indices] /= 2

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
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    model.eval()
    model.to(device)
    logits = []
    with torch.no_grad():
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


def calculate_same(same_prob, pos_list, price_list, leverage_ratio, enter_ratio, profit, entry_pos, curr_close, additional_count, limit=2, cut_value=1.):
    index = torch.tensor([0, 1, 3])
    logit = torch.argmax(same_prob[:, index], dim=1)
    hold_index = torch.where(logit == 0)[0]
    enter_index = torch.where((logit == 1) & (additional_count < limit))[0]
    loss_index = torch.where(logit == 2)[0]

    # loss
    pos_list[loss_index] = 0  # 'hold' -> 0
    loss_profit = (price_list[loss_index] - curr_close) / price_list[loss_index] * 100
    loss_profit = loss_profit * leverage_ratio[loss_index] * enter_ratio[loss_index]

    # enter
    before_price_list = price_list[enter_index]
    before_enter_list = enter_ratio[enter_index]
    cut_enter = cut_value - before_enter_list
    
    enter_enter_ratio = torch.sigmoid(same_prob[enter_index][:, 5])
    enter_enter_ratio = torch.minimum(cut_enter, enter_enter_ratio)
    after_price_list = before_price_list * (before_enter_list / (before_enter_list + enter_enter_ratio)) \
                       + curr_close * (enter_enter_ratio / (before_enter_list + enter_enter_ratio))
    after_enter_ratio = before_enter_list + enter_enter_ratio

    if entry_pos == 2:  # 'long' -> 2
        profit[loss_index] = profit[loss_index] - loss_profit - 0.1 * leverage_ratio[loss_index] * enter_ratio[loss_index]
    elif entry_pos == 1:  # 'short' -> 1
        profit[loss_index] = profit[loss_index] + loss_profit - 0.1 * leverage_ratio[loss_index] * enter_ratio[loss_index]

    price_list[loss_index] = -1.
    leverage_ratio[loss_index] = -1
    enter_ratio[loss_index] = -1.

    # Increment additional_count for allowed entries
    additional_count[enter_index] += 1
    price_list[enter_index] = after_price_list
    enter_ratio[enter_index] = after_enter_ratio

    return pos_list, price_list, leverage_ratio, enter_ratio, additional_count, profit

def calculate_diff(diff_prob, pos_list, price_list, leverage_ratio, enter_ratio, profit, entry_pos, curr_close, additional_count):
    index = torch.tensor([0, 1, 2])
    logit = torch.argmax(diff_prob[:, index], dim=1)
    hold_index = torch.where(logit == 0)[0]
    switch_index = torch.where(logit == 1)[0]
    take_index = torch.where(logit == 2)[0]

    # switch
    switch_profit = (price_list[switch_index] - curr_close) / price_list[switch_index] * 100
    switch_profit = switch_profit * leverage_ratio[switch_index] * enter_ratio[switch_index]
    switch_leverage = torch.sigmoid(diff_prob[switch_index][:, 4]) * 100.
    switch_enter_ratio = torch.sigmoid(diff_prob[switch_index][:, 5])

    # take
    pos_list[take_index] = 0  # 'hold' -> 0
    take_profit = (price_list[take_index] - curr_close) / price_list[take_index] * 100
    take_profit = take_profit * leverage_ratio[take_index] * enter_ratio[take_index]

    if entry_pos == 2:  # 'long' -> 2
        # switch
        profit[switch_index] = profit[switch_index] + switch_profit - leverage_ratio[switch_index] * 0.1 * enter_ratio[switch_index]
        pos_list[switch_index] = 2  # 'long' -> 2

        # take
        profit[take_index] = profit[take_index] + take_profit - leverage_ratio[take_index] * 0.1 * enter_ratio[take_index]
    elif entry_pos == 1:  # 'short' -> 1
        # switch
        profit[switch_index] = profit[switch_index] - switch_profit - leverage_ratio[switch_index] * 0.1 * enter_ratio[switch_index]
        pos_list[switch_index] = 1  # 'short' -> 1

        # take
        profit[take_index] = profit[take_index] - take_profit - leverage_ratio[take_index] * 0.1 * enter_ratio[take_index]

    price_list[switch_index] = curr_close
    leverage_ratio[switch_index] = switch_leverage.int()+1
    enter_ratio[switch_index] = switch_enter_ratio

    price_list[take_index] = -1.
    leverage_ratio[take_index] = -1
    enter_ratio[take_index] = -1.

    # Reset additional_count for switched and taken positions
    additional_count[switch_index] = 0
    additional_count[take_index] = 0

    return pos_list, price_list, leverage_ratio, enter_ratio, additional_count, profit


def calculate_hold(hold_prob, pos_list, price_list, leverage_ratio, enter_ratio, profit, entry_pos, curr_close, additional_count):
    index = torch.tensor([0, 1])
    logit = torch.argmax(hold_prob[:, index], dim=1)
    hold_index = torch.where(logit == 0)[0]
    enter_index = torch.where(logit == 1)[0]

    # enter
    enter_leverage = torch.sigmoid(hold_prob[enter_index][:, 4]) * 100.
    enter_enter_ratio = torch.sigmoid(hold_prob[enter_index][:, 5])
    price_list[enter_index] = curr_close
    leverage_ratio[enter_index] = enter_leverage.int() + 1
    enter_ratio[enter_index] = enter_enter_ratio

    if entry_pos == 2:  # 'long' -> 2
        pos_list[enter_index] = 2  # 'long' -> 2
    elif entry_pos == 1:  # 'short' -> 1
        pos_list[enter_index] = 1  # 'short' -> 1

    # Initialize additional_count for new positions
    additional_count[enter_index] = 0

    return pos_list, price_list, leverage_ratio, enter_ratio, additional_count, profit

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

    model.eval()
    with torch.no_grad():
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
    weights = [
        0.2,  # mean_returns: 0
        0.05,  # sharpe_ratios: 1
        0.10,  # sortino_ratios: 2
        0.10,  # profit_factors: 3
        0.15,  # win_rates: 4
        0.2,  # max_drawdowns: 5
        0.2  # cumulative_returns
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
        hold_pos = torch.where(pos_list == 0)[0]
        same_pos = torch.where(pos_list == entry_pos)[0]
        diff_pos = torch.where((pos_list != entry_pos) & (pos_list != 0))[0]
        
        now_profit = calculate_now_profit(pos_list, price_list, leverage_ratio, enter_ratio, curr_close)
        prob = after_forward(prescriptor, prob, now_profit, leverage_ratio, enter_ratio, pos_list, device=device)
        same_prob = prob[same_pos]
        diff_prob = prob[diff_pos]
        hold_prob = prob[hold_pos]
        
        pos_list[same_pos], price_list[same_pos], leverage_ratio[same_pos], enter_ratio[same_pos], additional_count[same_pos], profit[same_pos] = calculate_same(
            same_prob, pos_list[same_pos], price_list[same_pos], leverage_ratio[same_pos], enter_ratio[same_pos], profit[same_pos],
            entry_pos, curr_close, additional_count[same_pos], limit
        )
                
        pos_list[diff_pos], price_list[diff_pos], leverage_ratio[diff_pos], enter_ratio[diff_pos], additional_count[diff_pos], profit[diff_pos] = calculate_diff(
            diff_prob, pos_list[diff_pos], price_list[diff_pos], leverage_ratio[diff_pos], enter_ratio[diff_pos], profit[diff_pos],
            entry_pos, curr_close, additional_count[diff_pos]
        )
                
        pos_list[hold_pos], price_list[hold_pos], leverage_ratio[hold_pos], enter_ratio[hold_pos], additional_count[hold_pos], profit[hold_pos] = calculate_hold(
            hold_prob, pos_list[hold_pos], price_list[hold_pos], leverage_ratio[hold_pos], enter_ratio[hold_pos], profit[hold_pos],
            entry_pos, curr_close, additional_count[hold_pos]
        )
        
        before_index = entry_index
        returns_list.append(profit.clone().cpu().detach().numpy())
        profit = torch.zeros(chromosomes_size, dtype=torch.float32, device=device)

    returns_list = np.array(returns_list).T  # Shape: (chromosomes_size, time_steps)

    # Calculate performance metrics
    metrics = calculate_performance_metrics(returns_list, minimum_date=simulation_date)

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

        logits = inference(dataset_1m, dataset_1d, prescriptor, device)
        probs = []
        for logit in logits:
            logit = torch.stack(logit, dim=0)
            probs.append(logit)
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
                
                # Initialize best fitness and chromosomes if not already done
                if best_profit is None:
                    best_profit = valid_metrics
                    best_chromosomes, _, _, _ = evolution.flatten_chromosomes()
                    best_chromosomes = torch.tensor(best_chromosomes[:elite_size]).clone()
                else:
                    # Flatten current chromosomes
                    chromosomes, _, _, _ = evolution.flatten_chromosomes()
                    chromosomes = chromosomes[:elite_size].clone()
                    
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
        torch.save(gen_data, os.path.join(temp_dir, f'generation_{gen_idx}.pt')) 
        
        train_fitness = calculate_fitness(train_metrics)
        evolution.evolve(torch.from_numpy(train_fitness))
        prescriptor = prescriptor.to(device)
        
        del logits
        del probs
    return best_chromosomes, best_profit

def generation_test(data_1m, dataset_1m, dataset_1d, prescriptor, skip_data_cnt,
                     start_data_cnt, end_data_cnt, chromosomes_size,
                     window_size, profit_init, 
                     entry_index_list=None, entry_pos_list=None, device='cuda:0'):
    
    logits = inference(dataset_1m, dataset_1d, prescriptor, device)
    probs = []
    for logit in logits:
        logit = torch.stack(logit, dim=0)
        probs.append(logit)
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