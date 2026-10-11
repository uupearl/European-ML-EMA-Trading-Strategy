
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'Data' / 'European_ETF_Daily_2021_2025.csv'
OUT = ROOT.parent / 'Results'
OUT.mkdir(exist_ok=True)
EMA_SHORT, EMA_LONG, TRADING_DAYS = 10, 50, 252
N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_LEAF = 300, 6, 12
RANDOM_STATE, TOP_N_FEATURES = 42, 5
PDF_STATIC_THRESHOLD = 0.58
THRESHOLD_GRID = np.arange(0.50, 0.701, 0.01)
DYNAMIC_QUANTILE, DYNAMIC_LOOKBACK, DYNAMIC_MIN_PERIODS = .90, 60, 20
TRANSACTION_COST_BPS = 5.0
PORTFOLIO_MODE = 'gross_normalized'  # or 'strict_dollar_neutral'

CORE_PDF_FEATURES = ['ret_1d','ret_5d','momentum_10d','rsi_14','zscore_20',
                     'ema_10','ema_50','vol_10d','vol_20d']
EXTRA_FEATURES = ['momentum_20d','ema_gap_10_50','price_ma20_gap','vol_ratio_10_20',
                  'macd_pct','macd_signal_pct','macd_hist_pct','bb_position_20',
                  'bb_width_20','atr_14_pct','volume_z_20','volume_ratio_5_20',
                  'skew_20','drawdown_20','overnight_gap','intraday_return']
FEATURES = CORE_PDF_FEATURES + EXTRA_FEATURES

def rsi(series: pd.Series, window: int=14) -> pd.Series:
    """Wilder-style RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)

def add_features(g: pd.DataFrame) -> pd.DataFrame:
    """Construct backward-looking predictors for one ETF."""
    g = g.sort_values('Date').copy()
    p = g['Adj Close'].astype(float)
    close = g['Close'].astype(float)
    open_ = g['Open'].astype(float)
    high = g['High'].astype(float)
    low = g['Low'].astype(float)
    volume = g['Volume'].astype(float)
    g['ret_1d'] = p.pct_change(1)
    g['ret_5d'] = p.pct_change(5)
    g['momentum_10d'] = p.pct_change(10)
    g['momentum_20d'] = p.pct_change(20)
    g['rsi_14'] = rsi(p, 14)
    ma20 = p.rolling(20).mean()
    price_sd20 = p.rolling(20).std()
    g['zscore_20'] = (p - ma20) / price_sd20.replace(0, np.nan)
    g['ema_10'] = p.ewm(span=EMA_SHORT, adjust=False).mean()
    g['ema_50'] = p.ewm(span=EMA_LONG, adjust=False).mean()
    g['ema_gap_10_50'] = g['ema_10'] / g['ema_50'] - 1
    g['price_ma20_gap'] = p / ma20 - 1
    g['ema_direction'] = np.where(g['ema_10'] >= g['ema_50'], 1.0, -1.0)
    g['vol_10d'] = g['ret_1d'].rolling(10).std() * np.sqrt(TRADING_DAYS)
    g['vol_20d'] = g['ret_1d'].rolling(20).std() * np.sqrt(TRADING_DAYS)
    g['vol_ratio_10_20'] = g['vol_10d'] / g['vol_20d'].replace(0, np.nan)
    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    g['macd_pct'] = macd / p
    g['macd_signal_pct'] = macd_signal / p
    g['macd_hist_pct'] = (macd - macd_signal) / p
    bb_std = p.rolling(20).std()
    bb_upper = ma20 + 2 * bb_std
    bb_lower = ma20 - 2 * bb_std
    g['bb_position_20'] = (p - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    g['bb_width_20'] = (bb_upper - bb_lower) / ma20.replace(0, np.nan)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    g['atr_14_pct'] = tr.rolling(14).mean() / close.replace(0, np.nan)
    vol_ma20 = volume.rolling(20).mean()
    vol_sd20 = volume.rolling(20).std()
    g['volume_z_20'] = (volume - vol_ma20) / vol_sd20.replace(0, np.nan)
    g['volume_ratio_5_20'] = volume.rolling(5).mean() / vol_ma20.replace(0, np.nan)
    g['skew_20'] = g['ret_1d'].rolling(20).skew()
    g['drawdown_20'] = p / p.rolling(20).max() - 1
    g['overnight_gap'] = open_ / close.shift(1) - 1
    g['intraday_return'] = close / open_ - 1
    g['fwd_ret_1d'] = p.shift(-1) / p - 1
    g['target_up'] = np.where(g['fwd_ret_1d'].notna(), (g['fwd_ret_1d'] > 0).astype(int), np.nan)
    g['target_realization_date'] = g['Date'].shift(-1)
    return g

def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    pieces = [add_features(g) for _, g in raw.groupby('Ticker', sort=True)]
    return pd.concat(pieces, ignore_index=True).sort_values(['Date', 'Ticker'])

def fit_rf(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF, max_features='sqrt', class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X, y.astype(int))
    return model

def safe_auc(y_true, p_up) -> float:
    y = np.asarray(y_true)
    p = np.asarray(p_up)
    mask = pd.notna(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)

def add_direction_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make ML confidence consistent with EMA direction:
      Long EMA  -> P(up)
      Short EMA -> P(down) = 1 - P(up)
    """
    z = df.copy()
    z['direction_confidence'] = np.where(z['ema_direction'] > 0, z['p_up'], 1.0 - z['p_up'])
    return z

def add_static_signal(df: pd.DataFrame, threshold: float, signal_name: str) -> pd.DataFrame:
    z = add_direction_confidence(df)
    denom = max(1e-08, 1 - threshold)
    strength = ((z['direction_confidence'] - threshold) / denom).clip(lower=0, upper=1)
    z[signal_name] = np.where(z['direction_confidence'] >= threshold, z['ema_direction'] * z['direction_confidence'] * strength, 0.0)
    return z

def add_dynamic_signal(df: pd.DataFrame, fallback_threshold: float) -> pd.DataFrame:
    """
    Dynamic filter:
    today's cutoff is the ETF's rolling 90th percentile of PAST confidence scores.
    shift(1) prevents today's observation from determining today's cutoff.
    """
    z = add_direction_confidence(df)
    z = z.sort_values(['Ticker', 'Date']).copy()
    z['dynamic_threshold'] = z.groupby('Ticker')['direction_confidence'].transform(lambda s: s.shift(1).rolling(DYNAMIC_LOOKBACK, min_periods=DYNAMIC_MIN_PERIODS).quantile(DYNAMIC_QUANTILE)).fillna(fallback_threshold)
    denom = (1 - z['dynamic_threshold']).clip(lower=1e-08)
    strength = ((z['direction_confidence'] - z['dynamic_threshold']) / denom).clip(lower=0, upper=1)
    z['ml_dynamic_signal'] = np.where(z['direction_confidence'] >= z['dynamic_threshold'], z['ema_direction'] * z['direction_confidence'] * strength, 0.0)
    return z

def signal_to_weights(signal_wide: pd.DataFrame) -> pd.DataFrame:
    s = signal_wide.fillna(0.0).copy()
    if PORTFOLIO_MODE == 'gross_normalized':
        gross = s.abs().sum(axis=1).replace(0, np.nan)
        return s.div(gross, axis=0).fillna(0.0)
    if PORTFOLIO_MODE != 'strict_dollar_neutral':
        raise ValueError(f'Unknown PORTFOLIO_MODE: {PORTFOLIO_MODE}')
    positive = s.clip(lower=0)
    negative_abs = -s.clip(upper=0)
    pos_sum = positive.sum(axis=1)
    neg_sum = negative_abs.sum(axis=1)
    long_w = positive.div(pos_sum.replace(0, np.nan), axis=0) * 0.5
    short_w = -negative_abs.div(neg_sum.replace(0, np.nan), axis=0) * 0.5
    w = long_w.fillna(0.0) + short_w.fillna(0.0)
    valid = (pos_sum > 0) & (neg_sum > 0)
    w.loc[~valid, :] = 0.0
    return w

def portfolio_from_signal(panel: pd.DataFrame, signal_col: str):
    signal_wide = panel.pivot(index='Date', columns='Ticker', values=signal_col).sort_index()
    ret_wide = panel.pivot(index='Date', columns='Ticker', values='fwd_ret_1d').sort_index()
    dates = signal_wide.index.intersection(ret_wide.index)
    signal_wide = signal_wide.loc[dates]
    ret_wide = ret_wide.loc[dates]
    weights = signal_to_weights(signal_wide)
    gross_return = (weights * ret_wide.fillna(0)).sum(axis=1)
    turnover = 0.5 * weights.diff().abs().sum(axis=1)
    if len(turnover) > 0:
        turnover.iloc[0] = 0.5 * weights.iloc[0].abs().sum()
    cost = turnover * (TRANSACTION_COST_BPS / 10000)
    net_return = gross_return - cost
    return (net_return, weights, turnover)

def equal_weight_benchmark(panel: pd.DataFrame) -> pd.Series:
    ret_wide = panel.pivot(index='Date', columns='Ticker', values='fwd_ret_1d').sort_index()
    return ret_wide.mean(axis=1, skipna=True).fillna(0)

def maximum_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns.fillna(0)).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1
    return drawdown.min()

def performance_stats(returns: pd.Series, turnover: pd.Series | None=None) -> dict:
    r = returns.dropna()
    total_return = (1 + r).prod() - 1
    cagr = (1 + total_return) ** (TRADING_DAYS / len(r)) - 1
    ann_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS) if r.std(ddof=1) > 0 else np.nan
    downside = r[r < 0]
    downside_sd = downside.std(ddof=1)
    sortino = r.mean() / downside_sd * np.sqrt(TRADING_DAYS) if pd.notna(downside_sd) and downside_sd > 0 else np.nan
    mdd = maximum_drawdown(r)
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    var_95 = r.quantile(0.05)
    cvar_95 = r[r <= var_95].mean()
    result = {'Total Return': total_return, 'CAGR': cagr, 'Annualized Volatility': ann_vol, 'Sharpe Ratio': sharpe, 'Sortino Ratio': sortino, 'Maximum Drawdown': mdd, 'Calmar Ratio': calmar, 'Daily Hit Rate': (r > 0).mean(), 'VaR 95% (daily)': var_95, 'CVaR 95% (daily)': cvar_95, 'N Days': len(r)}
    if turnover is not None:
        t = turnover.reindex(r.index).fillna(0)
        result['Average Daily Turnover'] = t.mean()
        result['Annualized Turnover'] = t.mean() * TRADING_DAYS
    return result

def main():
    raw = pd.read_csv(DATA, parse_dates=['Date']).sort_values(['Ticker','Date'])
    required = ['Date','Ticker','Open','High','Low','Close','Adj Close','Volume']
    if raw[required].isna().any().any() or raw.duplicated(['Date','Ticker']).any():
        raise ValueError('Invalid input: missing OHLCV values or duplicated ETF dates')
    df = build_panel(raw)
    df[FEATURES] = df[FEATURES].replace([np.inf,-np.inf], np.nan)
    df = df.dropna(subset=FEATURES + ['target_up','fwd_ret_1d']).copy()
    df['target_up'] = df['target_up'].astype(int)

    # Purge forward labels that cross the validation/test cutoff.
    train = df[(df.Date <= '2023-12-31') & (df.target_realization_date <= '2023-12-31')]
    valid = df[(df.Date.between('2024-01-01','2024-12-31')) &
               (df.target_realization_date <= '2024-12-31')].copy()
    train_valid = df[(df.Date <= '2024-12-31') &
                     (df.target_realization_date <= '2024-12-31')]
    test = df[df.Date.between('2025-01-01','2025-12-31')].copy()

    importance_model = fit_rf(train[FEATURES], train.target_up)
    importance = pd.DataFrame({'Feature':FEATURES,
                               'Importance':importance_model.feature_importances_})
    importance = importance.sort_values('Importance', ascending=False)
    selected = importance.head(TOP_N_FEATURES).Feature.tolist()
    importance.to_csv(OUT/'feature_importance.csv',index=False)

    valid['p_up'] = fit_rf(train[selected],train.target_up).predict_proba(valid[selected])[:,1]
    auc_valid = valid.groupby('Ticker').apply(
        lambda g: safe_auc(g.target_up,g.p_up),include_groups=False)
    candidates = []
    for tau in THRESHOLD_GRID:
        v = add_static_signal(valid,float(tau),'tmp_signal')
        rets,_,_ = portfolio_from_signal(v,'tmp_signal')
        sharpe = rets.mean()/rets.std(ddof=1)*np.sqrt(252) if rets.std(ddof=1)>0 else np.nan
        candidates.append([float(tau),sharpe,(1+rets).prod()-1])
    tune = pd.DataFrame(candidates,columns=['Threshold','Validation Sharpe','Validation Total Return'])
    tau = float(tune.loc[tune['Validation Sharpe'].idxmax(),'Threshold']) if tune['Validation Sharpe'].notna().any() else .58
    tune.to_csv(OUT/'threshold_tuning_2024.csv',index=False)

    test['p_up'] = fit_rf(train_valid[selected],train_valid.target_up).predict_proba(test[selected])[:,1]
    auc_test = test.groupby('Ticker').apply(
        lambda g: safe_auc(g.target_up,g.p_up),include_groups=False)
    pd.DataFrame({'Ticker':auc_test.index,'Validation_AUC_2024':auc_valid.reindex(auc_test.index).values,
                  'Test_AUC_2025':auc_test.values}).to_csv(OUT/'auc_by_ticker.csv',index=False)

    test['ema_signal'] = test['ema_direction']
    test = add_static_signal(test,.58,'ml_fixed_058_signal')
    test = add_static_signal(test,tau,'ml_tuned_signal')
    test = add_dynamic_signal(test,tau)
    specs = {'Traditional_EMA':('ema_signal','weights_traditional_ema.csv'),
             'ML_EMA_Fixed_058':('ml_fixed_058_signal','weights_ml_fixed_058.csv'),
             f'ML_EMA_Tuned_{tau:.2f}':('ml_tuned_signal','weights_ml_tuned.csv'),
             'ML_EMA_Dynamic_Q90':('ml_dynamic_signal','weights_ml_dynamic.csv')}
    returns = {'EqualWeight_DailyRebalanced':equal_weight_benchmark(test)}
    turns = {}
    for name,(signal,file) in specs.items():
        returns[name],weight,turns[name] = portfolio_from_signal(test,signal)
        weight.to_csv(OUT/file)
    returns = pd.DataFrame(returns).fillna(0)
    returns.to_csv(OUT/'portfolio_daily_returns_2025.csv')
    perf = pd.DataFrame({name:performance_stats(returns[name],turns.get(name))
                         for name in returns}).T
    perf.to_csv(OUT/'performance_summary_2025.csv')

    keep = ['Date','Ticker','Market','ETF_Name','Adj Close','fwd_ret_1d','target_up',
            'ema_10','ema_50','ema_direction','p_up','direction_confidence','ema_signal',
            'ml_fixed_058_signal','ml_tuned_signal','dynamic_threshold','ml_dynamic_signal']
    test[keep].to_csv(OUT/'test_2025_predictions_and_signals.csv',index=False)
    (OUT/'run_summary.txt').write_text(
        f'ETFs: {raw.Ticker.nunique()}\nRows: {len(raw)}\nTop features: {selected}\n'
        f'Tuned threshold: {tau:.2f}\nPortfolio: {PORTFOLIO_MODE}\n\n'
        +perf[['Total Return','Sharpe Ratio','Maximum Drawdown']].round(5).to_string())
    print('Top features:', selected)
    print('Validation cutoff:',f'{tau:.2f}')
    print(perf[['Total Return','Sharpe Ratio','Maximum Drawdown']].round(5))
    print('Saved:',OUT)

if __name__ == '__main__':
    main()
