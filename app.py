"""
GoldSignalBot - by default watches USA Tech 100 (Nasdaq-100) only (gold
support is still fully in place - set ENABLE_GOLD=true in Render to bring
it back) and emails you when a confluence of technical indicators, plus an
analog-match against similar recent setups, points to a buy opportunity.

Runs as a Flask web service (for Render's free tier, which requires an
HTTP port to stay awake) with a background thread doing the actual
market checks on a timer. Pair with an external keep-alive pinger
(e.g. cron-job.org hitting "/" every ~10 min) so Render's free tier
doesn't spin the service down from inactivity.
"""
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from flask import Flask, jsonify

import champion_strategy
import config
import data_fetch
import notifier
import signals
import top2_gold_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gold_signal_bot")

app = Flask(__name__)

# One state block per configured instrument (config.INSTRUMENTS), keyed by
# instrument "key" (e.g. "TECH100" - gold no longer runs through this old
# continuous confluence engine, see "champion" state block below).
state = {
    "started_at": datetime.now(timezone.utc).isoformat(),
    "instruments": {
        inst["key"]: {
            "name": inst["name"],
            "last_check": None,
            "last_price": None,
            "last_direction": "NONE",
            "last_score": 0,
            "last_signal_sent": {"BUY": None, "SELL": None},  # direction -> {"time": datetime, "score": int}
            "consecutive_errors": 0,
            "last_error": None,
            "last_data_source": None,  # which provider actually served the last successful check - see data_fetch.get_candles
        }
        for inst in config.INSTRUMENTS
    },
    # По заявка (10.10): златото вече не е в "instruments" по-горе - то има
    # собствен checkpoint-базиран цикъл (champion_strategy.py), а не
    # continuous confluence скан. Виж check_champion_gold()/background_loop_champion()
    # по-долу.
    "champion": {
        "name": config.GOLD_INSTRUMENT["name"],
        "enabled": config.ENABLE_GOLD,
        "checkpoints": [f"{h:02d}:{m:02d}" for h, m in config.CHAMPION_CHECKPOINTS],
        "checkpoints_tz": config.CHAMPION_CHECKPOINTS_TZ,
        "last_check": None,
        "last_price": None,
        "last_direction": "NONE",
        "last_data_source": None,
        "consecutive_errors": 0,
        "last_error": None,
    },
    # Добавено 21.11 - отделен, НЕЗАВИСИМ цикъл за TOP2 OR
    # вътрешно-дневната стратегия (5-мин, NY сесия, виж
    # top2_gold_strategy.py). day_key/signaled_today/flat_sent_today
    # следят ЕДИН NY календарен ден - нулират се автоматично при смяна на
    # деня (виж check_top2_gold()), за да се праща най-много 1 LONG имейл
    # + 1 "без сигнал" обобщение на ден, не на всяка 5-мин проверка.
    "top2": {
        "name": config.GOLD_INSTRUMENT["name"],
        "enabled": config.ENABLE_TOP2_GOLD,
        "day_key": None,
        "signaled_today": False,
        "flat_sent_today": False,
        "last_check": None,
        "last_price": None,
        "last_direction": "NONE",
        "last_historical_signals": None,
        "last_data_source": None,
        "consecutive_errors": 0,
        "last_error": None,
    },
}


def _within_active_window() -> bool:
    """By user request (18.09): Monday-Friday only, config.ACTIVE_START_*
    to config.ACTIVE_END_* in config.ACTIVE_HOURS_TZ. If the timezone data
    is somehow unavailable, fail open (check anyway) rather than going
    silent because of a bug here."""
    try:
        tz = ZoneInfo(config.ACTIVE_HOURS_TZ)
    except Exception:
        log.warning("ACTIVE_HOURS_TZ (%s) invalid - skipping the active-window check.", config.ACTIVE_HOURS_TZ)
        return True
    now_local = datetime.now(tz)
    if config.ACTIVE_DAYS_ONLY and now_local.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    start = now_local.replace(hour=config.ACTIVE_START_HOUR, minute=config.ACTIVE_START_MINUTE, second=0, microsecond=0)
    end = now_local.replace(hour=config.ACTIVE_END_HOUR, minute=config.ACTIVE_END_MINUTE, second=0, microsecond=0)
    return start <= now_local <= end


def _cooldown_hours_for(score: int) -> float:
    """By user request (24.09): a strong/high-conviction signal gets a
    longer cooldown before the next alert - see the comment on
    STRONG_SIGNAL_SCORE_THRESHOLD in config.py."""
    if score >= config.STRONG_SIGNAL_SCORE_THRESHOLD:
        return config.STRONG_SIGNAL_COOLDOWN_HOURS
    return config.REALERT_COOLDOWN_HOURS


def _should_send(inst_state: dict, direction: str) -> bool:
    """Dedupe: only re-alert the same direction after a cooldown that
    scales with how strong the LAST SENT alert for that direction was
    (not the current check's score - the wait is about giving that earlier
    trade room to develop)."""
    last_sent = inst_state["last_signal_sent"].get(direction)
    if last_sent is None:
        return True
    cooldown_hours = _cooldown_hours_for(last_sent["score"])
    return datetime.now(timezone.utc) - last_sent["time"] > timedelta(hours=cooldown_hours)


def check_instrument(instrument: dict):
    key = instrument["key"]
    inst_state = state["instruments"][key]
    try:
        source_info = {}
        df = data_fetch.get_candles(instrument, interval=config.CANDLE_INTERVAL, period="60d", source_info=source_info)
        result = signals.evaluate(df)

        inst_state["last_check"] = datetime.now(timezone.utc).isoformat()
        inst_state["last_price"] = result.price
        inst_state["last_direction"] = result.direction
        inst_state["last_score"] = result.score
        inst_state["last_data_source"] = source_info.get("source")
        inst_state["consecutive_errors"] = 0
        inst_state["last_error"] = None

        log.info(
            "[%s] Check complete: price=%.2f direction=%s score=%d/%d",
            key, result.price, result.direction, result.score, result.max_score,
        )

        alertable_directions = ("BUY",) if config.ALERT_ONLY_BUY else ("BUY", "SELL")
        if result.direction in alertable_directions:
            if _should_send(inst_state, result.direction):
                sent = notifier.send_signal_email(result, instrument)
                if sent:
                    inst_state["last_signal_sent"][result.direction] = {
                        "time": datetime.now(timezone.utc), "score": result.score,
                    }
            else:
                # By user request (24.09, after a "log shows BUY but no email
                # arrived" report): make the cooldown skip visible in the
                # logs too, not just the ALERT_ONLY_BUY skip below - a
                # qualifying signal that doesn't email can otherwise look
                # identical to a bug.
                last_sent = inst_state["last_signal_sent"].get(result.direction)
                minutes_ago = (datetime.now(timezone.utc) - last_sent["time"]).total_seconds() / 60 if last_sent else None
                cooldown_minutes = _cooldown_hours_for(last_sent["score"]) * 60 if last_sent else config.REALERT_COOLDOWN_HOURS * 60
                log.info(
                    "[%s] %s setup seen (score=%d/%d) but still in cooldown (last %s alert was %.1f min ago, "
                    "scored %s/7 so cooldown is %.0f min) - not re-emailing.",
                    key, result.direction, result.score, result.max_score, result.direction,
                    minutes_ago if minutes_ago is not None else -1,
                    last_sent["score"] if last_sent else "?", cooldown_minutes,
                )
        elif result.direction == "SELL" and config.ALERT_ONLY_BUY:
            log.info("[%s] SELL setup seen (score=%d/%d) but ALERT_ONLY_BUY=true - not emailing.", key, result.score, result.max_score)

    except Exception as exc:  # noqa: BLE001
        inst_state["consecutive_errors"] += 1
        inst_state["last_error"] = str(exc)
        log.exception("[%s] Check failed", key)
        # Only email about infrastructure trouble after repeated failures,
        # so a single flaky request doesn't spam the inbox.
        if inst_state["consecutive_errors"] == 5:
            notifier.send_error_email(key, str(exc))


def check_champion_gold():
    """Една checkpoint проверка на шампионската стратегия за злато - вика
    се точно на часовете в config.CHAMPION_CHECKPOINTS (09:00 Europe/Sofia
    по подразбиране - единственият час, на който quiet_accumulation+
    ema_trend е честно валидирана, виж champion_strategy.py), не на
    continuous timer. Изпраща имейл ВИНАГИ - LONG/SHORT или честно FLAT -
    по изрична заявка на потребителя да получава отговор на всяка проверка."""
    cs = state["champion"]
    try:
        source_info = {}
        df = data_fetch.get_candles(
            config.GOLD_INSTRUMENT, interval=config.GOLD_CANDLE_INTERVAL,
            period=config.GOLD_FETCH_PERIOD, source_info=source_info,
        )
        result = champion_strategy.evaluate(df)

        cs["last_check"] = datetime.now(timezone.utc).isoformat()
        cs["last_price"] = result.price
        cs["last_direction"] = result.direction
        cs["last_data_source"] = source_info.get("source")
        cs["consecutive_errors"] = 0
        cs["last_error"] = None

        log.info(
            "[GOLD/champion] Checkpoint complete: price=%.2f direction=%s quiet=%s trend=%s",
            result.price, result.direction, result.quiet_vote, result.trend_vote,
        )

        if result.direction in ("LONG", "SHORT"):
            notifier.send_champion_signal_email(result)
        else:
            notifier.send_champion_flat_email(result)

    except Exception as exc:  # noqa: BLE001
        cs["consecutive_errors"] += 1
        cs["last_error"] = str(exc)
        log.exception("[GOLD/champion] Checkpoint check failed")
        if cs["consecutive_errors"] == 5:
            notifier.send_error_email("GOLD", str(exc))


def _next_champion_checkpoint() -> tuple:
    """Връща (datetime на следващия checkpoint, секунди до него)."""
    try:
        tz = ZoneInfo(config.CHAMPION_CHECKPOINTS_TZ)
    except Exception:
        log.warning("CHAMPION_CHECKPOINTS_TZ (%s) invalid - using UTC.", config.CHAMPION_CHECKPOINTS_TZ)
        tz = timezone.utc
    now_local = datetime.now(tz)
    candidates = []
    for h, m in config.CHAMPION_CHECKPOINTS:
        candidate = now_local.replace(hour=h, minute=m, second=0, microsecond=0)
        if candidate <= now_local:
            candidate += timedelta(days=1)
        candidates.append(candidate)
    next_dt = min(candidates)
    return next_dt, max(1.0, (next_dt - now_local).total_seconds())


def background_loop_champion():
    """Отделен, checkpoint-базиран цикъл за златото - спи до следващия от
    config.CHAMPION_CHECKPOINTS, прави проверка, спи до следващия, и т.н.
    Независим от стария CHECK_INTERVAL_MINUTES continuous скан по-долу
    (background_loop(), за Tech100/стария confluence engine)."""
    while True:
        next_dt, seconds = _next_champion_checkpoint()
        log.info("[GOLD/champion] Следващ checkpoint: %s (след %.0f мин)", next_dt.isoformat(), seconds / 60)
        time.sleep(seconds)
        check_champion_gold()


def _top2_session_status(now_ny) -> str:
    """"before" (сесията още не е почнала за днес), "in" (вътре в
    09:30-16:00 ET), "after" (сесията е приключила за днес). Събота/неделя
    се връща като "after" (нищо за правене, просто чакаме деня да се
    смени от check_top2_gold())."""
    if now_ny.weekday() >= 5:
        return "after"
    start = now_ny.replace(hour=config.TOP2_SESSION_START_HOUR, minute=config.TOP2_SESSION_START_MINUTE, second=0, microsecond=0)
    end = now_ny.replace(hour=config.TOP2_SESSION_END_HOUR, minute=config.TOP2_SESSION_END_MINUTE, second=0, microsecond=0)
    if now_ny < start:
        return "before"
    if now_ny > end:
        return "after"
    return "in"


def check_top2_gold():
    """Проверка на TOP2 OR стратегията (виж top2_gold_strategy.py) - вика
    се на всеки config.TOP2_CHECK_INTERVAL_MINUTES, но реално тегли данни
    САМО докато сме вътре в NY сесията (09:30-16:00 ET) - извън нея просто
    следи дали денят се е сменил (за да нулира signaled_today/
    flat_sent_today) и дали трябва да прати honest "без сигнал днес"
    обобщение веднъж, точно след затварянето."""
    t2 = state["top2"]
    try:
        tz = ZoneInfo(config.TOP2_SESSION_TZ)
    except Exception:
        log.warning("TOP2_SESSION_TZ (%s) invalid - skipping this TOP2 check.", config.TOP2_SESSION_TZ)
        return
    now_ny = datetime.now(tz)
    day_key = now_ny.date().isoformat()
    if t2["day_key"] != day_key:
        t2["day_key"] = day_key
        t2["signaled_today"] = False
        t2["flat_sent_today"] = False

    status = _top2_session_status(now_ny)
    if status == "before":
        return
    if status == "after":
        if now_ny.weekday() < 5 and not t2["signaled_today"] and not t2["flat_sent_today"]:
            # Mon-Fri only - събота/неделя също се връщат като "after"
            # (виж _top2_session_status), но няма сесия през уикенда,
            # затова не пращаме "без сигнал днес" имейл за дни, в които
            # пазарът изобщо не е бил отворен.
            # Не теглим нови данни само за да кажем "нямаше сигнал" -
            # вече знаем, защото signaled_today е False цял ден.
            result = top2_gold_strategy.Top2Result(
                direction="FLAT", price=t2.get("last_price") or 0.0,
                historical_signals_used=t2.get("last_historical_signals") or 0,
                reasons=["Сесията (09:30-16:00 ET) приключи без LONG огъване на TOP2 OR днес."],
            )
            notifier.send_top2_flat_email(result)
            t2["flat_sent_today"] = True
        return

    # status == "in"
    try:
        source_info = {}
        df = data_fetch.get_candles(
            config.GOLD_INSTRUMENT, interval=config.TOP2_CANDLE_INTERVAL, period=config.TOP2_FETCH_PERIOD,
            source_info=source_info, twelvedata_outputsize=config.TOP2_TWELVEDATA_OUTPUTSIZE,
            twelvedata_timezone=config.TOP2_TWELVEDATA_TIMEZONE,
        )
        result = top2_gold_strategy.evaluate(df)

        t2["last_check"] = datetime.now(timezone.utc).isoformat()
        t2["last_price"] = result.price
        t2["last_direction"] = result.direction
        t2["last_historical_signals"] = result.historical_signals_used
        t2["last_data_source"] = source_info.get("source")
        t2["consecutive_errors"] = 0
        t2["last_error"] = None

        log.info(
            "[GOLD/top2] Check complete: price=%s direction=%s historical_signals=%d enough_data=%s",
            result.price, result.direction, result.historical_signals_used, result.enough_data,
        )

        if result.direction == "LONG" and not t2["signaled_today"]:
            sent = notifier.send_top2_signal_email(result)
            if sent:
                t2["signaled_today"] = True

    except Exception as exc:  # noqa: BLE001
        t2["consecutive_errors"] += 1
        t2["last_error"] = str(exc)
        log.exception("[GOLD/top2] Check failed")
        if t2["consecutive_errors"] == 5:
            notifier.send_error_email("GOLD_TOP2", str(exc))


def background_loop_top2():
    """НЕЗАВИСИМ цикъл от background_loop_champion() по-горе - проверява
    на всеки config.TOP2_CHECK_INTERVAL_MINUTES (по подразбиране 5 мин),
    не на 1 checkpoint/ден. check_top2_gold() сам решава дали реално да
    тегли данни (само докато сме в NY сесията)."""
    time.sleep(max(30, config.TOP2_CHECK_INTERVAL_MINUTES * 60))
    while True:
        check_top2_gold()
        time.sleep(max(30, config.TOP2_CHECK_INTERVAL_MINUTES * 60))


def check_once():
    if not _within_active_window():
        log.info(
            "Outside the active window (%02d:%02d-%02d:%02d %s, Mon-Fri only=%s) - skipping this cycle.",
            config.ACTIVE_START_HOUR, config.ACTIVE_START_MINUTE,
            config.ACTIVE_END_HOUR, config.ACTIVE_END_MINUTE,
            config.ACTIVE_HOURS_TZ, config.ACTIVE_DAYS_ONLY,
        )
        return
    for instrument in config.INSTRUMENTS:
        check_instrument(instrument)


def background_loop():
    # By user request (25.09, after "всеки път когато се пусне ми изпраща
    # съобщение бай"): a restart used to trigger an INSTANT check_once()
    # with a freshly wiped cooldown (state only lives in memory, so a new
    # process has no idea a signal was already emailed minutes ago). While
    # the cron-job.org keepalive was disabled earlier today, the service
    # kept restarting every ~15 min - each restart re-ran check_once()
    # immediately with zero cooldown protection, which is exactly what
    # made it look like "every single time it starts, I get a BUY email".
    # That root cause is fixed now (keepalive re-enabled), but waiting one
    # full interval before the very first check is a cheap safety net
    # against the same symptom if the service ever restarts again for any
    # other reason (a manual deploy, a brief Render hiccup, ...) - worst
    # case it costs one delayed check, not a burst of un-cooled-down
    # alerts. Hit /check-now any time to trigger an immediate check
    # manually instead of waiting.
    time.sleep(max(60, config.CHECK_INTERVAL_MINUTES * 60))
    while True:
        check_once()
        time.sleep(max(60, config.CHECK_INTERVAL_MINUTES * 60))


@app.route("/")
def health():
    """Health/keep-alive endpoint. Point your uptime pinger (cron-job.org,
    UptimeRobot, ...) at this URL every ~10 minutes."""
    return jsonify({
        "status": "ok",
        "bot": "GoldSignalBot",
        **state,
    })


def _check_all_instruments_now():
    """Same as check_once() but ignores the active window - a manual
    /check-now hit is an explicit request to test right now, weekend or
    3am included. Also triggers an immediate champion-strategy gold
    checkpoint, regardless of its normal schedule."""
    for instrument in config.INSTRUMENTS:
        check_instrument(instrument)
    if config.ENABLE_GOLD:
        check_champion_gold()
    if config.ENABLE_TOP2_GOLD:
        check_top2_gold()


@app.route("/check-now")
def check_now():
    """Manually trigger an immediate check of every instrument (+ gold's
    champion strategy, if enabled), regardless of the active window/next
    scheduled checkpoint (useful for testing after deploy, any time)."""
    threading.Thread(target=_check_all_instruments_now, daemon=True).start()
    return jsonify({"status": "check triggered, see /"})


def start_background_thread():
    if config.SEND_TEST_EMAIL_ON_START:
        notifier.send_test_email()
    # По заявка (10.10): два независими фонови цикъла - стария continuous
    # confluence скан (Tech100, ако е включен) и новия checkpoint-базиран
    # цикъл за златото (шампионската стратегия). Всеки старира само ако
    # реално има какво да проверява.
    if config.INSTRUMENTS:
        threading.Thread(target=background_loop, daemon=True).start()
    if config.ENABLE_GOLD:
        threading.Thread(target=background_loop_champion, daemon=True).start()
    if config.ENABLE_TOP2_GOLD:
        threading.Thread(target=background_loop_top2, daemon=True).start()


start_background_thread()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT)
