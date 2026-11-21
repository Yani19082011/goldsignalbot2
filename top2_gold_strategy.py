"""
TOP2 OR вътрешно-дневна стратегия за злато (5-мин свещи, само NY сесия
09:30-16:00 America/New_York) - добавена 21.11 по изрична молба "избери
най-добрата стратегия за злато и ми кажи как да я кача".

Произход: намерена чрез отделен, разширен intraday strategy-search проект
(catalog_v2 - intraday_strategy_search.py + gld_intraday_search.py +
gld_top2_combo_test.py), тествана на 5-мин свещи на GLD (SPDR Gold Shares
ETF - прокси за злато, НЕ самата champion_strategy.py стратегия по-горе,
която е на GC=F/1h) в редовната NYSE сесия. Избрани двойки (от всичко
тествано, двете най-чисти):

  A) macd_cross(fast=8, slow=26, signal_period=12)
     AND adx_di_cross(period=20, adx_min=20.0)
  B) cci_reversal(period=14, oversold=-150.0)
     AND orb_mid_bounce(orb_bars=6, confirm_bars=2)

Крайният сигнал = A ИЛИ B (който от двата гръмне пръв в деня), само
ПЪРВОТО огъване на всеки от тях на отделен ден (без повторни влизания
след първото).

ЧЕСТНО, преди да се довериш сляпо (прочети и app.py/README за още):
  - Всичките 4 подсигнала са LONG-ONLY (отскок/пробив нагоре) - тази
    комбинация никога не праща SHORT сигнал. Не е грешка, а как се
    представи резултатът от тестовете - ако в бъдеще искаш и SHORT
    страна, трябва отделно search.
  - Тествано на GLD (ETF, търгува се само 09:30-16:00 ET), не на реалния
    GC=F/XAU-USD фийд, с който се захранва тук - затова се филтрира
    входящите бари до СЪЩИТЕ часове (09:30-16:00 America/New_York) преди
    да се пуснат сигналите, но GC=F/XAU-USD в тези часове не е
    гарантирано да се движи ИДЕНТИЧНО на GLD в същите часове (макар да са
    силно корелирани - GLD е обезпечен от физическо злато). Това е
    пренасяне на находка от прокси към реалния инструмент, не нов
    backtest на самия GC=F/XAU-USD в тези часове - следи резултатите
    живо преди да разчиташ сериозно на тях.
  - TP/SL НЕ са хардкоднати процента - извеждат се ДИНАМИЧНО на всяка
    проверка от историческото mfe/mae разпределение на СЪЩИЯ сигнал
    (50-ти перцентил на mfe -> TP, 20-ти перцентил на |mae| -> SL),
    точно както в backtest методологията (виж derive_tp_sl тук и
    gld_intraday_finalize.py в изследователския проект). Ако историята,
    върната от data provider-а, не съдържа достатъчно сигнали
    (config.TOP2_MIN_SIGNALS_FOR_TPSL), стратегията връща честно FLAT с
    enough_data=False, вместо да познава TP/SL "на доверие".
  - Twelve Data безплатния план има таван на исторически точки на
    заявка (config.TOP2_TWELVEDATA_OUTPUTSIZE) - на 5-мин бари и само
    6.5ч/ден полезни (NY сесия), това може да не стигне за надеждна
    статистика веднага след деплой. Виж логовете за "налични исторически
    сигнали: N" на всяка проверка.
"""
import logging
from dataclasses import dataclass, field
from datetime import time as dt_time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config

log = logging.getLogger("gold_signal_bot.top2")

# --- Точните параметри от validirания backtest (НЕ default-ите на функциите) -
MACD_FAST, MACD_SLOW, MACD_SIGNAL_PERIOD = 8, 26, 12
ADX_PERIOD, ADX_MIN = 20, 20.0
CCI_PERIOD, CCI_OVERSOLD = 14, -150.0
ORB_BARS, ORB_CONFIRM_BARS = 6, 2

MIN_WARMUP_BARS = 10  # колкото в изследователския каталог - предпазва от сигнали в първите бари на деня


def _session_mask(index_ny: pd.DatetimeIndex) -> pd.Series:
    start = dt_time(config.TOP2_SESSION_START_HOUR, config.TOP2_SESSION_START_MINUTE)
    end = dt_time(config.TOP2_SESSION_END_HOUR, config.TOP2_SESSION_END_MINUTE)
    times = index_ny.time
    return pd.Series((times >= start) & (times <= end), index=index_ny)


def to_ny_session_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Приема суров OHLCV df (капитализирани колони High/Low/Close/Open,
    както го връща data_fetch.get_candles) и връща само баровете, попадащи
    в config.TOP2_SESSION_TZ/START/END (по подразбиране 09:30-16:00
    America/New_York) - за да наподобява GLD-ETF сесията, на която беше
    тествана стратегията, независимо от това колко часа реално търгува
    самия GC=F/XAU-USD фийд."""
    if df.empty:
        return df
    tz = ZoneInfo(config.TOP2_SESSION_TZ)
    idx = df.index
    idx_ny = idx.tz_localize("UTC").tz_convert(tz) if idx.tz is None else idx.tz_convert(tz)
    mask = _session_mask(idx_ny)
    out = df.loc[mask.values].copy()
    out.index = idx_ny[mask.values]
    return out


def split_by_ny_day(df_session: pd.DataFrame) -> list:
    """df_session трябва да е ВЕЧЕ филтриран през to_ny_session_bars (NY tz
    индекс, само сесийни часове). Връща списък (date, day_df), сортиран по
    дата, всеки ден с малки колони (open/high/low/close/volume), точно
    както очакват портнатите sig_* функции по-долу."""
    if df_session.empty:
        return []
    out = []
    df = df_session.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    for date, day_df in df.groupby(df.index.date):
        out.append((date, day_df))
    out.sort(key=lambda t: t[0])
    return out


# --- Портнати ТОЧНО от intraday_strategy_search.py (catalog_v2), без промяна в логиката ---

def sig_macd_cross(day_df: pd.DataFrame) -> pd.Series:
    n = len(day_df)
    out = pd.Series(False, index=day_df.index)
    if n <= MACD_SLOW + MACD_SIGNAL_PERIOD + 2:
        return out
    close = day_df["close"]
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig_line = macd.ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
    cross_up = (macd >= sig_line) & (macd.shift(1) < sig_line.shift(1))
    cross_up.iloc[:MACD_SLOW + MACD_SIGNAL_PERIOD] = False
    out[cross_up.fillna(False)] = True
    return out


def sig_adx_di_cross(day_df: pd.DataFrame) -> pd.Series:
    n = len(day_df)
    out = pd.Series(False, index=day_df.index)
    if n <= ADX_PERIOD * 2 + 2:
        return out
    high, low, close = day_df["high"], day_df["low"], day_df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=day_df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=day_df.index)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / ADX_PERIOD, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / ADX_PERIOD, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / ADX_PERIOD, adjust=False).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.ewm(alpha=1 / ADX_PERIOD, adjust=False).mean()
    cross_up = (plus_di >= minus_di) & (plus_di.shift(1) < minus_di.shift(1)) & (adx >= ADX_MIN)
    cross_up.iloc[:ADX_PERIOD * 2] = False
    out[cross_up.fillna(False)] = True
    return out


def sig_cci_reversal(day_df: pd.DataFrame) -> pd.Series:
    n = len(day_df)
    out = pd.Series(False, index=day_df.index)
    if n <= CCI_PERIOD + 2:
        return out
    tp = (day_df["high"] + day_df["low"] + day_df["close"]) / 3.0
    sma = tp.rolling(CCI_PERIOD).mean()
    mad = tp.rolling(CCI_PERIOD).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    was_oversold = cci.shift(1) < CCI_OVERSOLD
    cross_up = (cci >= CCI_OVERSOLD) & was_oversold.fillna(False)
    cross_up.iloc[:CCI_PERIOD + 1] = False
    out[cross_up.fillna(False)] = True
    return out


def sig_orb_mid_bounce(day_df: pd.DataFrame) -> pd.Series:
    n = len(day_df)
    out = pd.Series(False, index=day_df.index)
    if n <= ORB_BARS + ORB_CONFIRM_BARS + MIN_WARMUP_BARS:
        return out
    orb_high = day_df["high"].iloc[:ORB_BARS].max()
    orb_low = day_df["low"].iloc[:ORB_BARS].min()
    mid = (orb_high + orb_low) / 2.0
    close = day_df["close"]
    below_mid = close < mid
    was_below = below_mid.shift(1).rolling(ORB_CONFIRM_BARS).sum() >= ORB_CONFIRM_BARS
    above_mid = close >= mid
    bounce = above_mid & ~above_mid.shift(1, fill_value=False) & was_below.fillna(False)
    bounce.iloc[:ORB_BARS + ORB_CONFIRM_BARS] = False
    out[bounce.fillna(False)] = True
    return out


def _and_first_true(s1: pd.Series, s2: pd.Series) -> pd.Series:
    both = s1 & s2
    first_true = both & ~both.shift(1, fill_value=False)
    return first_true.fillna(False)


def combo_a(day_df: pd.DataFrame) -> pd.Series:
    """macd_cross AND adx_di_cross, само първия бар на деня в който и двата са True."""
    return _and_first_true(sig_macd_cross(day_df), sig_adx_di_cross(day_df))


def combo_b(day_df: pd.DataFrame) -> pd.Series:
    """cci_reversal AND orb_mid_bounce, само първия бар на деня в който и двата са True."""
    return _and_first_true(sig_cci_reversal(day_df), sig_orb_mid_bounce(day_df))


def top2_or_signal(day_df: pd.DataFrame) -> pd.Series:
    """ИЛИ (OR) от combo_a/combo_b - само ПЪРВОТО огъване на деня на коя да е от двете."""
    out = combo_a(day_df) | combo_b(day_df)
    first_true = out & ~out.shift(1, fill_value=False)
    return first_true.fillna(False)


def collect_historical_moves(days: list) -> list:
    """Точно analyze_signal от qqq_intraday_search.py/gld_intraday_search.py:
    за всеки минал ден, в който сигналът гръмне (само ПЪРВИЯТ), влизаме на
    OPEN на следващия бар и мерим mfe/mae/close до края на деня (без
    overnight). days трябва да е списъкът от split_by_ny_day МИНУС
    текущия (все още незавършен) ден."""
    moves = []
    for date, day_df in days:
        try:
            sig = top2_or_signal(day_df)
        except Exception:
            continue
        fired = sig[sig].index
        if len(fired) == 0:
            continue
        pos = day_df.index.get_loc(fired[0])
        entry_pos = pos + 1
        if entry_pos >= len(day_df):
            continue
        entry_price = day_df["open"].iloc[entry_pos]
        if not entry_price or entry_price != entry_price:
            continue
        window = day_df.iloc[entry_pos:]
        if window.empty:
            continue
        mfe_pct = (window["high"].max() - entry_price) / entry_price * 100.0
        mae_pct = (window["low"].min() - entry_price) / entry_price * 100.0
        close_pct = (window["close"].iloc[-1] - entry_price) / entry_price * 100.0
        moves.append({"date": str(date), "mfe_pct": mfe_pct, "mae_pct": mae_pct, "close_pct": close_pct})
    return moves


def derive_tp_sl(moves: list, tp_percentile: float, sl_percentile: float):
    """ТОЧНО derive_tp_sl от gld_intraday_finalize.py - 50-ти перцентил на
    mfe -> TP, 20-ти перцентил на |mae| -> SL (процента, не цена)."""
    mdf = pd.DataFrame(moves)
    tp = float(mdf["mfe_pct"].quantile(tp_percentile / 100.0))
    sl = abs(float(mdf["mae_pct"].quantile(sl_percentile / 100.0)))
    return round(tp, 3), round(sl, 3)


@dataclass
class Top2Result:
    direction: str  # "LONG" или "FLAT"
    price: float = 0.0
    timestamp: object = None
    tp_pct: float = None
    sl_pct: float = None
    take_profit: float = None
    stop_loss: float = None
    historical_signals_used: int = 0
    reasons: list = field(default_factory=list)
    enough_data: bool = True
    which_combo: str = None  # "A" (macd+adx) или "B" (cci+orb)


def evaluate(raw_df: pd.DataFrame) -> Top2Result:
    """raw_df - суров OHLCV df с капитализирани колони (High/Low/Close/Open),
    какъвто връща data_fetch.get_candles. Гледа само последния наличен бар
    на ТЕКУЩИЯ (все още незавършен) NY ден - причинно коректно, без
    lookahead."""
    session = to_ny_session_bars(raw_df)
    days = split_by_ny_day(session)
    if len(days) < 2:
        return Top2Result(direction="FLAT", enough_data=False, reasons=["Няма достатъчно дни история още."])

    today_date, today_df = days[-1]
    history_days = days[:-1]

    if len(today_df) == 0:
        return Top2Result(direction="FLAT", enough_data=False, reasons=["Днешният ден още няма бари в сесийния прозорец (09:30-16:00 ET)."])

    moves = collect_historical_moves(history_days)
    n_signals = len(moves)

    cur_price = float(today_df["close"].iloc[-1])
    cur_ts = today_df.index[-1]

    if n_signals < config.TOP2_MIN_SIGNALS_FOR_TPSL:
        return Top2Result(
            direction="FLAT", price=cur_price, timestamp=cur_ts,
            historical_signals_used=n_signals, enough_data=False,
            reasons=[
                f"Недостатъчна история за надеждно TP/SL: намерени {n_signals} исторически сигнала, "
                f"трябват >= {config.TOP2_MIN_SIGNALS_FOR_TPSL}. Честно FLAT, без форсирана догадка за TP/SL - "
                "изчакай повече дни история (или виж config.TOP2_TWELVEDATA_OUTPUTSIZE)."
            ],
        )

    tp_pct, sl_pct = derive_tp_sl(moves, config.TOP2_TP_PERCENTILE, config.TOP2_SL_PERCENTILE)

    sig_a_today = combo_a(today_df)
    sig_b_today = combo_b(today_df)
    fired_a = bool(sig_a_today.iloc[-1])
    fired_b = bool(sig_b_today.iloc[-1])

    if fired_a or fired_b:
        which = "A (macd_cross+adx_di_cross)" if fired_a else "B (cci_reversal+orb_mid_bounce)"
        take_profit = round(cur_price * (1 + tp_pct / 100.0), 2)
        stop_loss = round(cur_price * (1 - sl_pct / 100.0), 2)
        return Top2Result(
            direction="LONG", price=cur_price, timestamp=cur_ts,
            tp_pct=tp_pct, sl_pct=sl_pct, take_profit=take_profit, stop_loss=stop_loss,
            historical_signals_used=n_signals, which_combo=which,
            reasons=[
                f"Комбинация {which} гръмна точно на този бар (сигналът е LONG-ONLY, никога SHORT).",
                f"TP/SL изведени от историческото mfe/mae на {n_signals} минали сигнала "
                f"({config.TOP2_TP_PERCENTILE:.0f}-ти перцентил -> TP, {config.TOP2_SL_PERCENTILE:.0f}-ти -> SL).",
            ],
        )

    return Top2Result(
        direction="FLAT", price=cur_price, timestamp=cur_ts,
        historical_signals_used=n_signals,
        reasons=[f"Няма огъване на A или B точно сега (история: {n_signals} сигнала, достатъчно за TP/SL, но днес още нищо)."],
    )
