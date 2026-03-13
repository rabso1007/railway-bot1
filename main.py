#!/usr/bin/env python3
"""
QUANTUM FLOW TRADING BOT v1.8.9 - ULTIMATE INSTITUTIONAL EDITION WITH EDGE INTELLIGENCE ENGINE
تم دمج جميع التحسينات الاحترافية المطلوبة والإصلاحات الحرجة وفقًا للمراجعة الشاملة:
- إصلاح deadlock في SmartCache (نقل await future خارج القفل + جلب فعلي)
- إصلاح is_blacklisted (إزالة async with من دالة متزامنة)
- إصلاح عدد ? في save_trade
- إصلاح إدارة القفل في partial_exit
- إضافة deepcopy في close_trade_full
- إزالة المعامل غير المستخدم tp1 من apply_be_and_trail
- إزالة الكود الميت _ORDER_FLOW_SAMPLING_OK
- فصل التحذيرات عن الأخطاء في validate_config
- تعديل save_emergency_checkpoint لتكون غير متزامنة
- إزالة حلقة إعادة التشغيل من main() والاعتماد على معالجة الأخطاء داخل async_main
"""

import asyncio
import aiohttp
from aiohttp import web
import aiodns
import ccxt.async_support as ccxt
import pandas as pd
import ta
import numpy as np
import time
import json
import logging
import os
import sqlite3
import hashlib
import re
import signal
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Deque
from collections import deque, defaultdict
from decimal import Decimal
import traceback
import tracemalloc
from functools import wraps
import copy
from contextlib import contextmanager, asynccontextmanager
import random
import math
from logging.handlers import RotatingFileHandler
import html  # for escaping

# التحقق من إصدار ccxt المطلوب
REQUIRED_CCXT_VERSION = "4.3.74"
try:
    installed_version = ccxt.__version__
    if installed_version != REQUIRED_CCXT_VERSION:
        print(f"⚠️ إصدار ccxt المثبت ({installed_version}) يختلف عن المطلوب ({REQUIRED_CCXT_VERSION}). يُنصح بتثبيت الإصدار المحدد.")
except Exception:
    print("⚠️ تعذر التحقق من إصدار ccxt.")

# ===================== EDGE ENGINE IMPORTS =====================
try:
    from edge_engine import get_edge_engine, EdgeState as RealEdgeState
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("edge_engine غير موجود - استخدام الوضع الافتراضي")
    class EdgeState:
        system_state = "NORMAL"
        def risk_multiplier(self): return 1.0
        def should_trade(self): return True
        async def record_trade(self, r_multiple, quantum_score, exit_type): return None
        def get_telemetry_report(self): return "EdgeEngine (fallback): no data"
    async def get_edge_engine(): return EdgeState()

# ------------------------------------------------------------
# التحقق من المكتبات الاختيارية
# ------------------------------------------------------------
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy غير متوفر - بعض الميزات ستكون محدودة")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ sklearn غير متوفر - ميزات ML معطلة")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib غير متوفر - سيتم استخدام ta")

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler('quantum_flow_institutional.log', maxBytes=10_000_000, backupCount=5, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ===================== GRACEFUL SHUTDOWN HANDLER =====================
class GracefulShutdown:
    def __init__(self):
        self.should_stop = False
        self.tasks = []
    def add_task(self, task):
        self.tasks.append(task)
    async def shutdown(self):
        self.should_stop = True
        logger.info("🛑 Starting graceful shutdown...")
        for task in self.tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("✅ All tasks cancelled")

shutdown_manager = GracefulShutdown()

# ===================== AUDIT LOGGING =====================
audit_handler = RotatingFileHandler("audit.log", maxBytes=10_000_000, backupCount=5, encoding='utf-8')
audit_logger = logging.getLogger('audit')
audit_logger.setLevel(logging.INFO)
audit_logger.addHandler(audit_handler)
audit_logger.propagate = False

def log_order_audit(order_type: str, symbol: str, price: float, amount: float, status: str = ""):
    try:
        audit_logger.info(f"{datetime.now(timezone.utc).isoformat()},{order_type},{symbol},{price},{amount},{status}")
    except Exception as e:
        logger.error(f"[Audit Log Error] {str(e)}")

# ===================== METRICS COLLECTOR =====================
class MetricsCollector:
    MAX_METRIC_HISTORY = 1000
    def __init__(self):
        self.metrics: Dict[str, List[Dict]] = defaultdict(list)
        self.lock = asyncio.Lock()
    def record_latency(self, operation: str):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start = time.perf_counter()
                success = True
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    raise
                finally:
                    end = time.perf_counter()
                    async with self.lock:
                        self.metrics[operation].append({
                            "timestamp": time.time(),
                            "duration": end - start,
                            "success": success
                        })
                        if len(self.metrics[operation]) > self.MAX_METRIC_HISTORY:
                            self.metrics[operation] = self.metrics[operation][-self.MAX_METRIC_HISTORY:]
            return wrapper
        return decorator
    async def record_error(self, operation: str, error_type: str):
        async with self.lock:
            self.metrics[f"{operation}_errors"].append({
                "timestamp": time.time(),
                "error_type": error_type
            })
            if len(self.metrics[f"{operation}_errors"]) > self.MAX_METRIC_HISTORY:
                self.metrics[f"{operation}_errors"] = self.metrics[f"{operation}_errors"][-self.MAX_METRIC_HISTORY:]
    def get_percentiles(self, metric: str) -> Dict[str, float]:
        if metric not in self.metrics or not self.metrics[metric]:
            return {}
        durations = [m["duration"] for m in self.metrics[metric] if "duration" in m]
        if not durations:
            return {}
        durations.sort()
        return {
            "p50": durations[len(durations)//2],
            "p95": durations[int(len(durations)*0.95)] if len(durations)>1 else durations[0],
            "p99": durations[int(len(durations)*0.99)] if len(durations)>1 else durations[0],
            "max": durations[-1],
            "count": len(durations)
        }
    def get_summary(self) -> Dict:
        summary = {}
        for operation, records in self.metrics.items():
            if records and "duration" in records[0]:
                percentiles = self.get_percentiles(operation)
                if percentiles:
                    summary[operation] = percentiles
        return summary

metrics = MetricsCollector()

# ===================== EXPONENTIAL BACKOFF RETRY =====================
class ExponentialBackoffRetry:
    def __init__(self, max_retries=3, base_delay=1, max_delay=60):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    async def execute(self, func, *args, **kwargs):
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except (asyncio.TimeoutError, aiohttp.ClientError, ccxt.NetworkError,
                    ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection,
                    ccxt.RateLimitExceeded) as e:
                last_exception = e
                if attempt == self.max_retries - 1:
                    break
                delay = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                logger.warning(f"Retry {attempt+1}/{self.max_retries} after {delay:.1f}s: {e}")
                await asyncio.sleep(delay)
        if last_exception is None:
            last_exception = RuntimeError("Retry attempts exhausted")
        raise last_exception

# ===================== ENHANCED LOCK MANAGER =====================
class EnhancedLockManager:
    def __init__(self):
        self.trade_locks: Dict[str, asyncio.Lock] = {}
        self.global_lock = asyncio.Lock()
        self.failed_locks: Dict[str, Dict] = {}  # key -> {'count': int, 'last_failure': timestamp}
        self.MAX_LOCK_RETRIES = 3
        self.LOCK_TIMEOUT = 10
        self.BLACKLIST_TTL = 3600  # 1 hour

    async def acquire_trade_lock(self, symbol: str) -> bool:
        async with self.global_lock:
            if symbol not in self.trade_locks:
                self.trade_locks[symbol] = asyncio.Lock()
            lock = self.trade_locks[symbol]
        for attempt in range(self.MAX_LOCK_RETRIES):
            try:
                await asyncio.wait_for(lock.acquire(), timeout=self.LOCK_TIMEOUT)
                # Reset failure count on success
                async with self.global_lock:
                    self.failed_locks.pop(symbol, None)
                return True
            except asyncio.TimeoutError:
                if attempt == self.MAX_LOCK_RETRIES - 1:
                    now = time.time()
                    async with self.global_lock:
                        if symbol not in self.failed_locks:
                            self.failed_locks[symbol] = {'count': 1, 'last_failure': now}
                        else:
                            self.failed_locks[symbol]['count'] += 1
                            self.failed_locks[symbol]['last_failure'] = now
                        logger.critical(f"🚨 DEADLOCK: {symbol} (failures: {self.failed_locks[symbol]['count']})")
                        if self.failed_locks[symbol]['count'] >= 3:
                            asyncio.create_task(send_telegram(
                                f"🚨 CRITICAL DEADLOCK\n\nSymbol: {symbol}\nFailures: {self.failed_locks[symbol]['count']}\nAction: Symbol temporarily blacklisted",
                                critical=True
                            ))
                    return False
                await asyncio.sleep(2 ** attempt)
        return False

    def release_trade_lock(self, symbol: str):
        if symbol in self.trade_locks:
            try:
                self.trade_locks[symbol].release()
            except RuntimeError:
                pass

    def is_blacklisted(self, symbol: str) -> bool:
        # هذه دالة متزامنة (لا تستخدم async with)
        now = time.time()
        if symbol not in self.failed_locks:
            return False
        # TTL check
        if now - self.failed_locks[symbol]['last_failure'] > self.BLACKLIST_TTL:
            # Auto-expire - but we need to modify dict, so we need lock? but this is sync function.
            # Better to move expiry to a background task, but for simplicity we'll just return False
            # and let the expiry happen in the next acquire attempt.
            # Actually we can't modify safely without lock, so we return False and ignore.
            # The acquire_trade_lock will eventually remove it on success.
            return False
        return self.failed_locks[symbol]['count'] >= 3

    async def acquire_all_locks(self):
        acquired = []
        failed = False
        try:
            async with self.global_lock:
                symbols = list(self.trade_locks.keys())
            for symbol in symbols:
                if await self.acquire_trade_lock(symbol):
                    acquired.append(symbol)
                else:
                    failed = True
                    logger.error(f"[LockManager] Failed to acquire lock for {symbol}")
                    break
            if failed:
                for s in acquired:
                    self.release_trade_lock(s)
                return []
            return acquired
        except Exception as e:
            logger.error(f"[LockManager Error] {str(e)}")
            for s in acquired:
                self.release_trade_lock(s)
            return []

    def release_all_locks(self, symbols: List[str]):
        for symbol in symbols:
            self.release_trade_lock(symbol)

@asynccontextmanager
async def trade_lock(bot_instance: 'TradingBot', symbol: str):
    acquired = await bot_instance.get_trade_lock(symbol)
    if not acquired:
        raise RuntimeError(f"Could not acquire lock for {symbol}")
    try:
        yield
    finally:
        bot_instance.release_trade_lock(symbol)

# ===================== TRADING BOT CLASS =====================
class TradingBot:
    def __init__(self):
        self.active_trades: Dict[str, 'TradeState'] = {}
        self.stats: Dict[str, Any] = {
            "signals_generated": 0,
            "signals_a_plus": 0,
            "signals_a": 0,
            "signals_b": 0,
            "daily_a_plus_count": 0,
            "trades_won": 0,
            "trades_lost": 0,
            "trades_partial": 0,
            "total_r_multiple": 0.0,
            "tp1_hits": 0,
            "tp2_hits": 0,
            "hard_gates_passed": 0,
            "hard_gates_failed": 0,
            "api_errors": 0,
            "retries_count": 0,
            "avg_quantum_score": 0.0,
            "live_orders_placed": 0,
            "live_orders_filled": 0,
            "live_orders_canceled": 0,
            "live_sells_executed": 0,
            "live_order_errors": 0,
            "live_emergencies": 0,
            "paper_trades_opened": 0,
            "loop_count": 0,
            "last_reset_date": None,
            "global_consecutive_losses": 0,
        }
        self.lock_manager = EnhancedLockManager()
        self.symbol_cooldown: Dict[str, float] = {}
        self.btc_trend: Optional[Dict] = None
        self.btc_last_check: float = 0
        self.btc_trend_lock = asyncio.Lock()
        self.consecutive_losses: Dict[str, List[float]] = defaultdict(list)
        self.consecutive_loss_blacklist: Dict[str, float] = {}
        self.correlation_cache: Dict[str, Tuple[float, float]] = {}
        self.correlation_cache_lock = asyncio.Lock()
        self.edge_engine = None

    async def get_trade_lock(self, symbol: str):
        return await self.lock_manager.acquire_trade_lock(symbol)

    def release_trade_lock(self, symbol: str):
        self.lock_manager.release_trade_lock(symbol)

bot = TradingBot()
STATS = bot.stats
ACTIVE_TRADES = bot.active_trades

# ===================== DAILY COUNTER RESET =====================
def reset_daily_counters():
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    if STATS.get("last_reset_date") != today:
        STATS["daily_a_plus_count"] = 0
        STATS["last_reset_date"] = today
        logger.info(f"♻️ تم إعادة تعيين عدادات A+ اليومية لـ {today}")

# ===================== API CIRCUIT BREAKER =====================
class APICircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failures = 0
        self.last_failure = 0
        self.threshold = failure_threshold
        self.timeout = timeout
        self.state = "CLOSED"
    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"
    def record_failure(self):
        self.failures += 1
        self.last_failure = time.time()
        if self.failures >= self.threshold:
            self.state = "OPEN"
            logger.warning(f"⚠️ API Circuit Breaker OPEN ({self.failures} failures)")
    def can_attempt(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit Breaker -> HALF_OPEN")
                return True
            return False
        return True
    def get_state(self) -> Dict:
        return {
            "state": self.state,
            "failures": self.failures,
            "last_failure": self.last_failure
        }

api_circuit = APICircuitBreaker(failure_threshold=5, timeout=60)

# ===================== ENHANCED HELPER FUNCTIONS =====================
def escape_html(text: Any) -> str:
    """استخدام html.escape من المكتبة القياسية بدلاً من الدوال المعقدة"""
    return html.escape(str(text))

def safe_float(value: Any, default: float = 0.0, log_invalid: bool = False) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and not np.isfinite(value):
            if log_invalid:
                logger.warning(f"[safe_float] Invalid float: {value}")
            return default
        result = float(value)
        if not np.isfinite(result):
            if log_invalid:
                logger.warning(f"[safe_float] Non-finite result: {result} from {value}")
            return default
        return result
    except (TypeError, ValueError) as e:
        if log_invalid:
            logger.warning(f"[safe_float] Conversion error: {value} -> {e}")
        return default

def validate_price(price: float) -> bool:
    if not isinstance(price, (int, float)):
        return False
    if not (np.isfinite(price) and price > 0):
        return False
    if price > 1e8:
        return False
    return True

def validate_volume(volume: float) -> bool:
    if not isinstance(volume, (int, float)):
        return False
    if not np.isfinite(volume):
        return False
    if volume < 0:
        return False
    if volume > 1e12:
        return False
    return True

def clamp(value: float, low: float, high: float) -> float:
    try:
        v = float(value)
        return max(low, min(high, v))
    except Exception:
        return low

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def is_live_trading_enabled() -> bool:
    return bool(CONFIG.get("ENABLE_LIVE_TRADING", False))

def is_paper_trading_enabled() -> bool:
    if is_live_trading_enabled():
        return False
    return bool(CONFIG.get("PAPER_TRADING_MODE", False))

def get_execution_mode_badge() -> str:
    if is_live_trading_enabled():
        return "✅ LIVE"
    if is_paper_trading_enabled():
        return "🟨 PAPER"
    return "🟦 SIGNAL"

def _coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def _validate_symbol(symbol: str) -> bool:
    pattern = r'^[A-Z0-9]{2,20}/[A-Z]{2,10}$'
    return bool(re.match(pattern, symbol))

# ===================== RUNTIME PATH FALLBACK =====================
def _ensure_runtime_paths():
    try:
        is_colab = False
        try:
            import google.colab
            is_colab = True
        except Exception:
            is_colab = False
        if is_colab:
            return
        runtime_dir = os.getenv("QUANTUM_RUNTIME_DIR", os.path.join(os.getcwd(), "quantum_runtime"))
        os.makedirs(runtime_dir, exist_ok=True)
        def _fix_path(key: str, default_filename: str):
            p = CONFIG.get(key, "")
            if isinstance(p, str) and (p.startswith("/content/") or not os.path.isabs(p)):
                newp = os.path.join(runtime_dir, default_filename)
                CONFIG[key] = newp
        _fix_path("CHECKPOINT_PATH", "quantum_checkpoint.json")
        _fix_path("EMERGENCY_CHECKPOINT_PATH", "quantum_emergency_checkpoint.json")
        _fix_path("EMERGENCY_SELL_LOG", "emergency_sells.json")
    except Exception:
        pass

# ===================== TIMESTAMP HELPER FUNCTIONS =====================
def find_candle_by_timestamp(df: pd.DataFrame, timestamp: int) -> Optional[int]:
    if df is None or len(df) == 0:
        return None
    if 't' in df.columns:
        col = 't'
    elif 'timestamp' in df.columns:
        col = 'timestamp'
    else:
        return None
    try:
        matches = df.index[df[col].astype("int64") == int(timestamp)].tolist()
        return matches[0] if matches else None
    except Exception:
        return None

def ensure_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    if 't' in df.columns and 'timestamp' not in df.columns:
        df['timestamp'] = df['t'].astype("int64")
    elif 'timestamp' not in df.columns:
        df['timestamp'] = df.index
    return df

# ===================== STABLE HASH FUNCTION =====================
def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)

# ===================== DATA CLASSES =====================
@dataclass
class MarketStructure:
    structure: str
    bos_bullish: bool
    choch: bool
    order_block: Optional[Dict[str, float]]
    fvg_zone: Optional[Dict[str, float]]
    liquidity_high: float
    liquidity_low: float
    swing_high: float
    swing_low: float
    trend_strength: float
    def __post_init__(self):
        self.trend_strength = max(0.0, min(100.0, safe_float(self.trend_strength, 0.0)))
        if not validate_price(self.swing_high):
            self.swing_high = 0.0
        if not validate_price(self.swing_low):
            self.swing_low = 0.0

@dataclass
class OrderFlowData:
    imbalance: float
    delta: float
    divergence: bool
    bid_strength: float
    ask_strength: float
    volume_profile: str
    signal: str
    confidence: float
    def __post_init__(self):
        self.imbalance = max(-1.0, min(1.0, safe_float(self.imbalance, 0.0)))
        self.delta = max(-1.0, min(1.0, safe_float(self.delta, 0.0)))
        self.confidence = max(0.0, min(100.0, safe_float(self.confidence, 50.0)))

@dataclass
class VolumeProfileData:
    poc: float
    vah: float
    val: float
    current_position: str
    vwap: float
    vwap_upper: float
    vwap_lower: float
    hvn_levels: List[float]
    lvn_levels: List[float]
    volume_trend: str
    def __post_init__(self):
        if not validate_price(self.poc):
            self.poc = 0.0
        if not validate_price(self.vwap):
            self.vwap = 0.0

@dataclass
class LiquidityGrab:
    detected: bool
    grab_type: str
    grab_level: float
    wick_strength: float
    volume_spike: float
    recovery_strength: float
    confidence: float
    equal_lows: bool = False
    equal_lows_range: float = 0.0
    sweep_candle_close: float = 0.0
    sweep_index: int = -1
    sweep_timestamp: Optional[int] = None
    def __post_init__(self):
        self.wick_strength = max(0.0, min(1.0, safe_float(self.wick_strength, 0.0)))
        self.confidence = max(0.0, min(100.0, safe_float(self.confidence, 50.0)))

@dataclass
class TradeState:
    symbol: str
    entry: float
    original_sl: float
    current_sl: float
    tp1: float
    tp2: float
    atr: float = 0.0
    tp1_hit: bool = False
    tp2_hit: bool = False
    remaining_position: float = 1.0
    be_moved: bool = False
    trailing_active: bool = False
    entry_time: str = field(default_factory=now_utc_iso)
    total_realized_r: float = 0.0
    last_update: str = field(default_factory=now_utc_iso)
    signal_class: str = ""
    quantum_score: float = 0.0
    gates_passed: List[str] = field(default_factory=list)
    entry_order_id: str = ""
    entry_filled: bool = False
    entry_fill_price: float = 0.0
    entry_fill_amount: float = 0.0
    tp1_order_done: bool = False
    tp2_order_done: bool = False
    sl_order_done: bool = False
    sl_order_id: str = ""
    emergency_state: bool = False
    emergency_reason: str = ""
    emergency_last_attempt: str = ""
    emergency_attempts: int = 0
    is_paper: bool = False
    execution_mode: str = "SIGNAL"
    entry_assumed: bool = False
    is_exiting: bool = False
    _version: int = 0
    order_block_low: float = 0.0
    order_block_high: float = 0.0
    liquidity_grab_level: float = 0.0

    def update_timestamp(self):
        self.last_update = now_utc_iso()

@dataclass
class QuantumSignal:
    symbol: str
    mode: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    atr: float
    position_size_usdt: float
    position_size_pct: float
    quantum_score: float
    confidence: float
    signal_class: str
    market_structure: MarketStructure
    order_flow: Optional[OrderFlowData]
    volume_profile: Optional[VolumeProfileData]
    liquidity_grab: Optional[LiquidityGrab]
    mtf_alignment: int
    trend_1h: str
    structure_15m: str
    entry_5m: str
    risk_reward: float
    win_probability: float
    gates_passed: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=now_utc_iso)

# ===================== CONFIGURATION المُحسّنة =====================
CONFIG = {
    # Exchange
    "EXCHANGE": "mexc",
    "QUOTE": "/USDT",
    # Timeframes
    "TF_TREND": "1h",
    "TF_STRUCTURE": "15m",
    "TF_ENTRY": "5m",
    # Market Structure Settings
    "SWING_WINDOW": 5,
    "BOS_CONFIRMATION_CANDLES": 2,
    "BOS_CONFIRMATION_MULTIPLIER": 1.001,
    "ORDER_BLOCK_LOOKBACK": 20,
    "FVG_MIN_SIZE_ATR": 0.5,
    # Order Flow Settings
    "ORDERBOOK_DEPTH": 40,
    "ORDERBOOK_WEIGHTED_LEVELS": 15,
    "TRADES_SAMPLE_SIZE": 100,
    "IMBALANCE_THRESHOLD": 0.25,
    "DELTA_THRESHOLD": 0.15,
    "DIVERGENCE_THRESHOLD": 0.3,
    "ORDER_FLOW_ENABLED": True,
    "ORDER_FLOW_AS_BOOSTER": True,
    "ORDER_FLOW_ONLY_HIGH_ALIGNMENT": True,
    "ORDER_FLOW_ABSORPTION_LOGIC_V2": True,
    "ORDER_FLOW_SAMPLING_ENABLED": True,
    "ORDER_FLOW_SAMPLE_EVERY_N_LOOPS": 3,
    # Daily Reset
    "DAILY_RESET_UTC_HOUR": 0,
    # Price Acceptance Gate
    "ENABLE_PRICE_ACCEPTANCE_GATE": True,
    "OB_ACCEPT_REQUIRE_RETEST_5M": True,
    "OB_ACCEPT_CLOSE_ABOVE_MID": True,
    "OB_ACCEPT_REJECTION_WICK": True,
    "OB_REJECTION_WICK_BODY_RATIO": 1.2,
    "LG_CONFIRM_NEXT_CANDLE_ONLY": True,
    "LG_CONFIRM_CLOSE_ABOVE_SWEEP_HIGH": True,
    "LG_CONFIRM_VOLUME_LOWER_THAN_SWEEP": True,
    "OB_ENTRY_MODE": "mid",
    "OB_ENTRY_BUFFER_PCT": 0.0005,
    # Risk Management
    "RISK_PER_TRADE_PCT": 1.0,
    "MAX_SL_PCT": 3.0,
    "ATR_SL_MULT": 1.5,
    "TP1_RR": 1.5,
    "TP2_RR": 3.0,
    "TP1_EXIT_PCT": 0.6,
    "TP2_EXIT_PCT": 0.4,
    "BE_AT_R": 1.0,
    "BE_ATR_MULT": 0.5,
    "TRAIL_START_R": 2.0,
    "TRAIL_ATR_MULT": 1.0,
    # Position Sizing
    "ACCOUNT_SIZE_USDT": 1000,
    "MIN_POSITION_SIZE_USDT": 10,
    "MAX_POSITION_SIZE_USDT": 200,
    # Trading Settings
    "LONG_ONLY": True,
    "MIN_QUANTUM_SCORE": 68,
    "QUANTUM_A_SCORE": 75,
    "QUANTUM_A_PLUS_SCORE": 80,
    "MAX_DAILY_A_PLUS": 7,
    # Hard Gates
    "ENABLE_HARD_GATES": True,
    "HARD_GATE_1_MIN_TREND_STRENGTH": 70,
    "HARD_GATE_1_MIN_MTF_ALIGNMENT": 2,
    "HARD_GATE_2_REQUIRE_ZONE": True,
    "HARD_GATE_2_MIN_LG_CONFIDENCE": 70,
    "HARD_GATE_2_OB_FRESHNESS": 8,
    "HARD_GATE_3_REQUIRE_BOOSTER": False,
    # Liquidity Grab Settings
    "LG_WICK_MIN_RATIO": 0.45,
    "LG_RECOVERY_MIN": 0.5,
    "LG_VOLUME_MULTIPLIER": 1.7,
    "LG_EQUAL_LOWS_REQUIRED": 3,
    "LG_EQUAL_LOWS_RANGE_ATR_MULT": 0.5,
    # Volume Profile
    "ENABLE_VOLUME_PROFILE": True,
    "VOLUME_PROFILE_BINS": 50,
    "VALUE_AREA_PCT": 0.7,
    "HVN_THRESHOLD": 1.0,
    "LVN_THRESHOLD": 0.5,
    # BASELINE
    "ENABLE_VOLUME_PROFILE_BASELINE": False,
    "ENABLE_ORDER_FLOW_BASELINE": False,
    # Multi-Timeframe
    "MIN_MTF_ALIGNMENT": 2,
    # Market Regime Filter
    "ENABLE_MARKET_REGIME_FILTER": True,
    "MIN_ADX_FOR_TREND": 16,
    "MAX_CHASE_MOVE_PCT": 0.03,
    # BTC Filter
    "ENABLE_BTC_FILTER": True,
    "BTC_CRASH_THRESHOLD": -3.0,
    "BTC_WARNING_THRESHOLD": -1.5,
    "BTC_CORRELATION_THRESHOLD": 0.3,
    # Live Trading
    "ENABLE_LIVE_TRADING": False,
    "MEXC_API_KEY": "",
    "MEXC_API_SECRET": "",
    "LIVE_MAX_OPEN_TRADES": 5,
    "MAX_SPREAD_PCT": 0.10,
    "ENTRY_ORDER_TYPE": "limit",
    "ENTRY_LIMIT_TIMEOUT_SEC": 120,
    "ENTRY_LIMIT_POLL_SEC": 3,
    "LIVE_REQUIRE_SPREAD_FILTER": True,
    "LIVE_RECALIBRATE_LEVELS_ON_FILL": True,
    "LIVE_RECALIBRATION_MODE": "rr",
    "LIVE_REQUIRE_BALANCE_RECONCILIATION": True,
    "MIN_DUST_THRESHOLD": 0.000001,
    "LIVE_PLACE_SL_ORDER": True,
    "LIVE_SL_ORDER_TYPE": "stop-loss",
    # Paper Trading
    "PAPER_TRADING_MODE": False,
    "PAPER_MAX_OPEN_TRADES": 25,
    # Entry Quality Filter
    "ENABLE_ENTRY_QUALITY_FILTER": True,
    "ENTRY_QUALITY_MAX_ATR_PCT_5M": 5.5,
    "ENTRY_QUALITY_MAX_BB_WIDTH_5M": 0.08,
    "ENTRY_QUALITY_MAX_DISTANCE_FROM_ZONE_ATR": 1.6,
    "RSI_MIN": 30,
    "RSI_MAX": 75,
    # Cooldown
    "SYMBOL_COOLDOWN_SEC": 1200,
    # Daily Circuit Breaker
    "ENABLE_DAILY_MAX_LOSS": True,
    "DAILY_MAX_LOSS_R": -4.0,
    # Telegram
    "TG_TOKEN": "",
    "TG_CHAT": "",
    "TG_SEND_CRITICAL_ALERTS": True,
    "SILENT_MODE": False,
    "AUTO_DISABLE_SILENT_WHEN_TG_OK": True,
    # Database
    "ENABLE_DB_PERSISTENCE": True,
    "DB_PATH": "quantum_trades_institutional.db",
    # Rate Limiting
    "REQUESTS_PER_MINUTE": 1200,
    "MAX_WEIGHT_PER_MINUTE": 6000,
    "REQUESTS_PER_SECOND": 10,
    "TICKER_WEIGHT": 2,
    "ORDERBOOK_WEIGHT": 10,
    "MAX_RETRIES": 3,
    "EXPONENTIAL_BACKOFF": True,
    # Checkpoints
    "ENABLE_CHECKPOINTS": True,
    "CHECKPOINT_INTERVAL_SEC": 300,
    "CHECKPOINT_PATH": "/content/quantum_checkpoint.json",
    "EMERGENCY_CHECKPOINT_PATH": "/content/quantum_emergency_checkpoint.json",
    "EMERGENCY_SELL_LOG": "/content/emergency_sells.json",
    # Debug
    "DEBUG_MODE": False,
    # Batch Processing
    "BATCH_SIZE": 15,
    # Advanced
    "ORDER_FLOW_ENABLE_FOR_ALIGNMENT_2_IF_STRONG_SCORE": True,
    "ORDER_FLOW_PRECHECK_MIN_SCORE": 68.0,
    # INSTITUTIONAL FEATURES
    "ENABLE_MEMORY_MONITORING": False,
    "ENABLE_HEALTH_CHECK": False,
    "HEALTH_CHECK_PORT": 8080,
    "CIRCUIT_BREAKER_ENABLED": True,
    "CIRCUIT_BREAKER_FAILURE_THRESHOLD": 5,
    "CIRCUIT_BREAKER_TIMEOUT": 60,
    "LOCK_MAX_RETRIES": 3,
    "LOCK_TIMEOUT_SECONDS": 10,
    "RETRY_MAX_RETRIES": 3,
    "RETRY_BASE_DELAY": 1,
    "RETRY_MAX_DELAY": 60,
    # Reconciliation
    "RECONCILIATION_INTERVAL_SEC": 300,
    # Volatility filter
    "MAX_ATR_PCT_15M": 8.0,
    # Slippage protection
    "MAX_SLIPPAGE_PCT": 0.0035,
    # Global consecutive losses circuit breaker
    "MAX_CONSECUTIVE_LOSSES": 3,
    # Liquidity filter configurable
    "MIN_VOLUME_24H": 2000000,
    # Enable liquidity grab
    "ENABLE_LIQUIDITY_GRAB": True,
    # Max symbol scan
    "MAX_SYMBOL_SCAN": 80,
    # Buffer for trailing stop (نسبة مئوية)
    "TRAILING_BUFFER_PCT": 0.001,
}

# ===================== TELEGRAM ENV AUTO-LOAD =====================
def load_telegram_from_env():
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_TOKEN") or
             os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN"))
    chat = (os.getenv("TELEGRAM_CHAT_ID") or os.getenv("TG_CHAT") or
            os.getenv("TELEGRAM_CHAT") or os.getenv("CHAT_ID"))
    if token and not CONFIG.get("TG_TOKEN"):
        CONFIG["TG_TOKEN"] = token.strip()
    if chat and not CONFIG.get("TG_CHAT"):
        CONFIG["TG_CHAT"] = chat.strip()
    CONFIG.setdefault("_TG_WARNED_ONCE", False)
    if not CONFIG.get("TG_TOKEN") or not CONFIG.get("TG_CHAT"):
        if not CONFIG["_TG_WARNED_ONCE"]:
            logger.warning("⚠️ Telegram غير مكوّن بالكامل - سيتم تشغيل الوضع الصامت تلقائياً إذا لزم")
            CONFIG["_TG_WARNED_ONCE"] = True
        CONFIG["SILENT_MODE"] = True
    else:
        if CONFIG.get("AUTO_DISABLE_SILENT_WHEN_TG_OK", True):
            CONFIG["SILENT_MODE"] = False

def load_mexc_from_env():
    key = os.getenv("MEXC_API_KEY") or os.getenv("MEXC_KEY")
    secret = os.getenv("MEXC_API_SECRET") or os.getenv("MEXC_SECRET")
    if key and not CONFIG.get("MEXC_API_KEY"):
        CONFIG["MEXC_API_KEY"] = key.strip()
    if secret and not CONFIG.get("MEXC_API_SECRET"):
        CONFIG["MEXC_API_SECRET"] = secret.strip()

# ===================== GLOBAL STATE =====================
HTTP_SESSION: Optional[aiohttp.ClientSession] = None
_http_session_lock = asyncio.Lock()
BTC_TREND: Optional[Dict] = None
BTC_LAST_CHECK: float = 0

# ===================== SYMBOL COOLDOWN =====================
def is_in_cooldown(symbol: str) -> bool:
    cd = safe_float(CONFIG.get("SYMBOL_COOLDOWN_SEC", 0), 0.0)
    if cd <= 0:
        return False
    last = bot.symbol_cooldown.get(symbol, 0)
    if time.time() - last < cd:
        return True
    return False

def set_symbol_cooldown(symbol: str):
    bot.symbol_cooldown[symbol] = time.time()

# ===================== PRICE ACCEPTANCE GATE =====================
def _candle_body_wick_ratios(candle: Dict) -> Tuple[float, float, float]:
    o = safe_float(candle.get("open", 0.0))
    c = safe_float(candle.get("close", 0.0))
    h = safe_float(candle.get("high", 0.0))
    l = safe_float(candle.get("low", 0.0))
    body = abs(c - o)
    lower_wick = max(0.0, min(o, c) - l)
    upper_wick = max(0.0, h - max(o, c))
    return body, lower_wick, upper_wick

def _ob_entry_price(ob: Dict[str, Any]) -> float:
    low = safe_float(ob.get("body_low", ob.get("low", 0.0)))
    high = safe_float(ob.get("body_high", ob.get("high", 0.0)))
    if not (validate_price(low) and validate_price(high) and high > low):
        return 0.0
    mid = (low + high) / 2.0
    mode = CONFIG.get("OB_ENTRY_MODE", "mid")
    buf = safe_float(CONFIG.get("OB_ENTRY_BUFFER_PCT", 0.0), 0.0)
    if mode == "low":
        return low * (1.0 + buf)
    return mid * (1.0 + buf)

def price_acceptance_gate_5m(df_5m: pd.DataFrame, ob: Optional[Dict[str, Any]], lg: Optional['LiquidityGrab']) -> Tuple[bool, str]:
    if not CONFIG.get("ENABLE_PRICE_ACCEPTANCE_GATE", True):
        return True, ""
    if df_5m is None or len(df_5m) < 2:
        return False, "no_data"
    last = df_5m.iloc[-1]
    prev = df_5m.iloc[-2]
    if ob:
        low = safe_float(ob.get("body_low", ob.get("low", 0.0)))
        high = safe_float(ob.get("body_high", ob.get("high", 0.0)))
        if not (validate_price(low) and validate_price(high) and high > low):
            return False, "ob_invalid"
        mid = (low + high) / 2.0
        current_atr = safe_float(df_5m['atr'].iloc[-1]) if 'atr' in df_5m.columns else 0
        last_low = safe_float(last.get("low"))
        last_high = safe_float(last.get("high"))
        last_close = safe_float(last.get("close"))
        if current_atr > 0:
            atr_tolerance = current_atr * 1.5
            tolerance_low = mid - atr_tolerance
            tolerance_high = mid + atr_tolerance
            price_in_zone = (last_low <= tolerance_high and last_high >= tolerance_low)
            close_near = abs(last_close - mid) <= atr_tolerance
        else:
            price_in_zone = (last_low <= high and last_high >= low)
            close_near = (last_close >= low) and (last_close <= high)
        if not price_in_zone:
            return False, "ob_no_retest"
        acceptance = False
        if CONFIG.get("OB_ACCEPT_CLOSE_ABOVE_MID", True) and close_near and last_close > mid:
            acceptance = True
        if (not acceptance) and CONFIG.get("OB_ACCEPT_REJECTION_WICK", True):
            body, lower_wick, upper_wick = _candle_body_wick_ratios(last)
            ratio = safe_float(CONFIG.get("OB_REJECTION_WICK_BODY_RATIO", 1.2), 1.2)
            open_ = safe_float(last.get("open"))
            close = safe_float(last.get("close"))
            if body > 0 and lower_wick >= ratio * body and (close >= open_):
                acceptance = True
        if not acceptance:
            return False, "ob_no_acceptance"
        vol_slice = df_5m['volume'].iloc[-20:-1]
        if len(vol_slice) > 0:
            avg_vol = vol_slice.mean()
            if np.isfinite(avg_vol) and last['volume'] < avg_vol * 0.7:
                return False, "ob_volume_too_low"
        else:
            return False, "ob_volume_insufficient_data"
        return True, "ok_ob_acceptance"
    if lg and lg.detected and lg.grab_type == "BULLISH":
        sweep_ts = lg.sweep_timestamp
        sweep_i = None
        if sweep_ts:
            sweep_i = find_candle_by_timestamp(df_5m, sweep_ts)
        if sweep_i is None or sweep_i == -1 or sweep_i >= len(df_5m) - 1:
            sweep = prev
            confirm = last
            sweep_i = len(df_5m) - 2
        else:
            sweep = df_5m.iloc[sweep_i]
            confirm = df_5m.iloc[sweep_i + 1]
        sweep_high = safe_float(sweep.get("high"))
        sweep_vol = safe_float(sweep.get("volume"))
        confirm_close = safe_float(confirm.get("close"))
        confirm_vol = safe_float(confirm.get("volume"))
        if CONFIG.get("LG_CONFIRM_CLOSE_ABOVE_SWEEP_HIGH", True):
            if confirm_close <= sweep_high:
                return False, "lg_confirm_close_below_sweep"
        if CONFIG.get("LG_CONFIRM_VOLUME_LOWER_THAN_SWEEP", True):
            if sweep_vol > 0 and confirm_vol >= sweep_vol:
                return False, "lg_confirm_volume_not_lower"
        return True, "ok_lg_confirmed"
    return False, "no_zone_no_trade"

# ===================== CONFIG VALIDATION =====================
def validate_config():
    errors = []      # أخطاء تمنع التشغيل
    warnings = []    # تحذيرات لا تمنع التشغيل

    if CONFIG.get("ENABLE_LIVE_TRADING"):
        if not CONFIG.get("MEXC_API_KEY"):
            errors.append("❌ MEXC_API_KEY مفقود (LIVE TRADING ON)")
        if not CONFIG.get("MEXC_API_SECRET"):
            errors.append("❌ MEXC_API_SECRET مفقود (LIVE TRADING ON)")

    if CONFIG.get("TP1_RR", 0) >= CONFIG.get("TP2_RR", 0):
        errors.append("❌ TP2_RR يجب أن يكون أكبر من TP1_RR")

    if CONFIG.get("RISK_PER_TRADE_PCT", 0) > 5:
        warnings.append("⚠️ المخاطرة عالية جداً (>5%)")

    if not CONFIG.get("TG_TOKEN") or not CONFIG.get("TG_CHAT"):
        CONFIG.setdefault("_TG_WARNED_IN_VALIDATE", False)
        if not CONFIG["_TG_WARNED_IN_VALIDATE"]:
            warnings.append("⚠️ Telegram غير مكوّن بالكامل - سيتم تشغيل الوضع الصامت تلقائياً إذا لزم")
            CONFIG["_TG_WARNED_IN_VALIDATE"] = True

    if CONFIG.get("LIVE_RECALIBRATION_MODE") not in ("rr",):
        errors.append("❌ LIVE_RECALIBRATION_MODE غير صحيح (القيمة المسموحة فقط: rr)")

    if errors:
        for err in errors:
            logger.error(err)
        raise ValueError("Configuration validation failed")
    if warnings:
        for warn in warnings:
            logger.warning(warn)
    logger.info("✅ Configuration validated (with warnings ignored)")

# ===================== WEIGHTED RATE LIMITER =====================
class WeightedRateLimiter:
    def __init__(self, max_per_minute: int, max_weight_per_minute: int, max_per_second: int):
        self.request_times = deque(maxlen=max_per_minute * 3)
        self.weights = deque(maxlen=2000)
        self.max_per_minute = int(max_per_minute)
        self.max_weight_per_minute = int(max_weight_per_minute)
        self.max_per_second = int(max_per_second)
        self.lock = asyncio.Lock()
        self.consecutive_errors = 0
        self.sec_times = deque(maxlen=max_per_second * 5)
        self.total_weight = 0

    async def wait_if_needed(self, weight: int = 1):
        backoff_sleep = 0.0
        if self.consecutive_errors > 0 and CONFIG.get("EXPONENTIAL_BACKOFF", True):
            backoff_sleep = min(2 ** self.consecutive_errors, 32)
        if backoff_sleep > 0:
            await asyncio.sleep(backoff_sleep)

        async with self.lock:
            now = time.time()
            # تنظيف القوائم
            while self.request_times and now - self.request_times[0] > 60:
                self.request_times.popleft()
            while self.weights and now - self.weights[0][0] > 60:
                _, w = self.weights.popleft()
                self.total_weight -= w
            while self.sec_times and now - self.sec_times[0] > 1:
                self.sec_times.popleft()

            sec_sleep = 0.0
            if len(self.sec_times) >= self.max_per_second:
                sec_sleep = 1 - (now - self.sec_times[0]) + 0.05

            min_sleep = 0.0
            if len(self.request_times) >= self.max_per_minute:
                oldest = self.request_times[0]
                min_sleep = 60 - (now - oldest) + 0.1

            weight_sleep = 0.0
            if self.total_weight + weight > self.max_weight_per_minute:
                if self.weights:
                    oldest_time = self.weights[0][0]
                    weight_sleep = 60 - (now - oldest_time) + 0.5

        # النوم خارج القفل
        max_sleep = max(sec_sleep, min_sleep, weight_sleep)
        if max_sleep > 0:
            logger.warning(f"⏸️ Rate limit: waiting {max_sleep:.1f}s")
            await asyncio.sleep(max_sleep)

        async with self.lock:
            now = time.time()
            while self.request_times and now - self.request_times[0] > 60:
                self.request_times.popleft()
            while self.weights and now - self.weights[0][0] > 60:
                _, w = self.weights.popleft()
                self.total_weight -= w
            while self.sec_times and now - self.sec_times[0] > 1:
                self.sec_times.popleft()
            self.request_times.append(now)
            self.sec_times.append(now)
            self.weights.append((now, int(weight)))
            self.total_weight += int(weight)

    def record_error(self):
        self.consecutive_errors += 1
        STATS["api_errors"] += 1

    def reset_errors(self):
        self.consecutive_errors = 0

rate_limiter = WeightedRateLimiter(
    max_per_minute=CONFIG["REQUESTS_PER_MINUTE"],
    max_weight_per_minute=CONFIG["MAX_WEIGHT_PER_MINUTE"],
    max_per_second=CONFIG["REQUESTS_PER_SECOND"]
)

# ===================== ENHANCED SMART CACHE (مع إصلاح deadlock) =====================
class SmartCache:
    def __init__(self):
        self.cache: Dict[str, Tuple[float, Any, int]] = {}
        self.lock = asyncio.Lock()
        self.pending_fetches: Dict[str, asyncio.Future] = {}
        self.hits = 0
        self.misses = 0
        self.max_size = 1000

    def _get_ttl(self, timeframe: str) -> int:
        ttl_map = {"1m": 15, "5m": 30, "15m": 60, "1h": 180, "4h": 600}
        return ttl_map.get(timeframe, 30)

    async def smart_cache_cleanup(self):
        async with self.lock:
            now = time.time()
            self.cache = {k: v for k, v in self.cache.items() if now - v[0] <= 600}
            if len(self.cache) >= self.max_size:
                sorted_items = sorted(
                    self.cache.items(),
                    key=lambda x: (x[1][2], x[1][0])
                )
                keep_count = int(self.max_size * 0.8)
                self.cache = dict(sorted_items[-keep_count:])
                logger.info(f"🧹 Cache: {keep_count} items after aggressive cleanup")
            else:
                logger.info(f"🧹 Cache: {len(self.cache)} items after cleanup")

    async def get_ohlcv(self, exchange, symbol: str, timeframe: str, limit: int = 150) -> Optional[List]:
        cache_key = f"{symbol}:{timeframe}:{limit}"
        ttl = self._get_ttl(timeframe)

        async with self.lock:
            if cache_key in self.pending_fetches:
                future = self.pending_fetches[cache_key]
            elif cache_key in self.cache:
                cache_time, data, access_count = self.cache[cache_key]
                if time.time() - cache_time < ttl:
                    self.cache[cache_key] = (cache_time, data, access_count + 1)
                    self.hits += 1
                    return data
                else:
                    del self.cache[cache_key]
                    self.misses += 1
                    future = asyncio.get_running_loop().create_future()
                    self.pending_fetches[cache_key] = future
                    asyncio.create_task(self._do_fetch(exchange, symbol, timeframe, limit, cache_key, future))
            else:
                self.misses += 1
                future = asyncio.get_running_loop().create_future()
                self.pending_fetches[cache_key] = future
                asyncio.create_task(self._do_fetch(exchange, symbol, timeframe, limit, cache_key, future))

        # انتظر خارج القفل
        try:
            data = await asyncio.shield(future)
            return data
        except Exception:
            return None
        finally:
            async with self.lock:
                self.pending_fetches.pop(cache_key, None)

    async def _do_fetch(self, exchange, symbol: str, timeframe: str, limit: int, cache_key: str, future: asyncio.Future):
        try:
            data = await self._fetch_ohlcv_with_rate_limit(exchange, symbol, timeframe, limit)
            async with self.lock:
                self.cache[cache_key] = (time.time(), data, 1)
            if not future.done():
                future.set_result(data)
        except Exception as e:
            if not future.done():
                future.set_exception(e)
        finally:
            async with self.lock:
                self.pending_fetches.pop(cache_key, None)

    async def _fetch_ohlcv_with_rate_limit(self, exchange, symbol: str, timeframe: str, limit: int):
        await rate_limiter.wait_if_needed(weight=2)
        return await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    async def clear_old_entries(self):
        async with self.lock:
            now = time.time()
            to_delete = [k for k, (t, _, _) in self.cache.items() if now - t > 600]
            for key in to_delete:
                del self.cache[key]
            if to_delete:
                logger.info(f"[Cache] Cleared {len(to_delete)} expired entries")

    def get_stats(self) -> Dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        avg_access = 0
        if self.cache:
            avg_access = sum(access_count for _, _, access_count in self.cache.values()) / len(self.cache)
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "avg_access_count": avg_access
        }

cache = SmartCache()

# ===================== ENHANCED HTTP SESSION =====================
async def get_session() -> aiohttp.ClientSession:
    global HTTP_SESSION
    async with _http_session_lock:
        if HTTP_SESSION is None or HTTP_SESSION.closed:
            timeout = aiohttp.ClientTimeout(
                total=30,
                connect=10,
                sock_read=15,
                sock_connect=10
            )
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300,
                enable_cleanup_closed=True
            )
            HTTP_SESSION = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                raise_for_status=False
            )
        return HTTP_SESSION

async def close_session():
    global HTTP_SESSION
    try:
        if HTTP_SESSION and not HTTP_SESSION.closed:
            await HTTP_SESSION.close()
            await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"[Session Close Error] {str(e)}")
    finally:
        HTTP_SESSION = None

# ===================== TELEGRAM =====================
async def send_telegram(msg: str, retry: int = 0, critical: bool = False):
    if not CONFIG["TG_TOKEN"] or not CONFIG["TG_CHAT"] or CONFIG["SILENT_MODE"]:
        logger.info(f"[TG] {msg[:200]}")
        return
    if critical and not CONFIG["TG_SEND_CRITICAL_ALERTS"]:
        logger.critical(f"[CRITICAL] {msg[:200]}")
        return
    try:
        session = await get_session()
        token = CONFIG['TG_TOKEN']
        chat_id = CONFIG['TG_CHAT']
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        if len(msg) > 4000:
            msg = msg[:3900] + "\n\n...[تم الاختصار]"
        async with session.post(url, json={
            "chat_id": chat_id,
            "text": msg,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                return
            error_text = await resp.text()
            logger.error(f"[TG Error] {resp.status}: {error_text[:100]}")
            if retry < 2:
                await asyncio.sleep(1)
                await send_telegram(msg, retry + 1, critical)
    except Exception as e:
        logger.error(f"[TG Exception] {type(e).__name__}: {str(e)[:100]}")
        if retry < 2:
            await asyncio.sleep(1)
            await send_telegram(msg, retry + 1, critical)

# ===================== ENHANCED LIVE TRADING HELPERS =====================
async def ensure_live_trading_ready(exchange) -> bool:
    if not is_live_trading_enabled():
        return False
    if not CONFIG.get("MEXC_API_KEY") or not CONFIG.get("MEXC_API_SECRET"):
        logger.error("[LIVE] Missing MEXC_API_KEY / MEXC_API_SECRET in ENV")
        await send_telegram(
            "❌ Live Trading مفعّل لكن مفاتيح MEXC غير موجودة\n"
            "• تأكد من ENV: MEXC_API_KEY و MEXC_API_SECRET\n"
            "• سيتم تعطيل التنفيذ الحقيقي تلقائيًا حفاظًا على الأمان.",
            critical=True
        )
        CONFIG["ENABLE_LIVE_TRADING"] = False
        return False
    return True

def _get_market(exchange, symbol: str) -> Optional[Dict[str, Any]]:
    try:
        return exchange.markets.get(symbol)
    except Exception:
        return None

def _round_amount_to_precision(exchange, symbol: str, amount: float) -> float:
    try:
        return safe_float(exchange.amount_to_precision(symbol, amount))
    except Exception:
        return safe_float(amount)

def _round_price_to_precision(exchange, symbol: str, price: float) -> float:
    try:
        return safe_float(exchange.price_to_precision(symbol, price))
    except Exception:
        return safe_float(price)

async def get_spread_pct(exchange, symbol: str) -> Optional[float]:
    try:
        await rate_limiter.wait_if_needed(weight=CONFIG.get("ORDERBOOK_WEIGHT", 10))
        ob = await exchange.fetch_order_book(symbol, 5)
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if not bids or not asks:
            return None
        best_bid = safe_float(bids[0][0])
        best_ask = safe_float(asks[0][0])
        if best_bid <= 0 or best_ask <= 0:
            return None
        rate_limiter.reset_errors()
        return ((best_ask - best_bid) / best_bid) * 100
    except Exception:
        rate_limiter.record_error()
        return None

async def compute_order_amount_base(exchange, symbol: str, usdt_size: float, price: float) -> Tuple[float, str]:
    if usdt_size <= 0 or price <= 0:
        return 0.0, "invalid_size_or_price"
    market = _get_market(exchange, symbol)
    if not market:
        return 0.0, "market_not_found"
    try:
        amount = usdt_size / price
        amount = _round_amount_to_precision(exchange, symbol, amount)
        notional = amount * price
        min_notional = safe_float(market.get("limits", {}).get("cost", {}).get("min", 5))
        if notional < min_notional:
            return 0.0, f"notional_too_small({notional:.2f}<{min_notional})"
        min_amount = safe_float(market.get("limits", {}).get("amount", {}).get("min", 0))
        if amount < min_amount:
            return 0.0, f"amount_too_small({amount}<{min_amount})"
        return amount, ""
    except Exception as e:
        return 0.0, f"calc_error: {str(e)[:50]}"

async def place_limit_buy_entry(exchange, symbol: str, price: float, amount: float) -> Optional[Dict[str, Any]]:
    try:
        px = _round_price_to_precision(exchange, symbol, price)
        amt = _round_amount_to_precision(exchange, symbol, amount)
        await rate_limiter.wait_if_needed(weight=1)
        order = await exchange.create_order(symbol, "limit", "buy", amt, px)
        STATS["live_orders_placed"] += 1
        log_order_audit("LIMIT_BUY", symbol, px, amt, "PLACED")
        logger.info(f"[ENTRY] Limit buy placed for {symbol}: price={px}, amount={amt}")
        rate_limiter.reset_errors()
        return order
    except Exception as e:
        STATS["live_order_errors"] += 1
        logger.error(f"[LIVE Entry Error] {symbol}: {str(e)[:150]}")
        log_order_audit("LIMIT_BUY", symbol, price, amount, f"ERROR: {str(e)[:50]}")
        rate_limiter.record_error()
        return None

def exchange_supports_stop_orders(exchange) -> bool:
    try:
        h = getattr(exchange, "has", {}) or {}
        keys = [
            "createStopLossOrder",
            "createStopMarketOrder",
            "createStopOrder",
            "createTriggerOrder",
            "createOrder"
        ]
        for k in keys:
            if h.get(k) is True:
                return True
    except Exception:
        pass
    return False

async def validate_stop_loss_capability(exchange):
    if not is_live_trading_enabled():
        return
    if not CONFIG.get("LIVE_PLACE_SL_ORDER", True):
        return
    supported = exchange_supports_stop_orders(exchange)
    if not supported:
        CONFIG["LIVE_PLACE_SL_ORDER"] = False
        logger.warning("[LIVE] Stop-loss orders appear unsupported on this exchange/market. Disabling LIVE_PLACE_SL_ORDER.")
        await send_telegram(
            "⚠️ تنبيه LIVE\n\n"
            "أوامر Stop-Loss على المنصة قد لا تكون مدعومة/غير مضمونة في Spot عبر CCXT.\n"
            "تم تعطيل LIVE_PLACE_SL_ORDER تلقائيًا.\n"
            "سيتم الاعتماد على المراقبة الداخلية والخروج Market Sell Safe فقط.",
            critical=False
        )

async def place_stop_loss_order(exchange, symbol: str, stop_price: float, amount: float) -> Optional[Dict[str, Any]]:
    if not CONFIG.get("LIVE_PLACE_SL_ORDER"):
        return None
    order_type = CONFIG.get("LIVE_SL_ORDER_TYPE", "stop-loss")
    try:
        amt = _round_amount_to_precision(exchange, symbol, amount)
        params = {'stopPrice': _round_price_to_precision(exchange, symbol, stop_price)}
        await rate_limiter.wait_if_needed(weight=1)
        order = await exchange.create_order(symbol, order_type, "sell", amt, None, params)
        log_order_audit("STOP_LOSS", symbol, stop_price, amt, "PLACED")
        logger.info(f"[SL] Stop loss placed for {symbol} at {stop_price}")
        rate_limiter.reset_errors()
        return order
    except Exception as e:
        logger.error(f"[LIVE SL Order Error] {symbol}: {str(e)[:150]}")
        log_order_audit("STOP_LOSS", symbol, stop_price, amount, f"ERROR: {str(e)[:50]}")
        rate_limiter.record_error()
        if CONFIG.get("LIVE_PLACE_SL_ORDER", True):
            CONFIG["LIVE_PLACE_SL_ORDER"] = False
            await send_telegram(
                "⚠️ LIVE SL Disabled\n\n"
                "فشل وضع Stop Loss على المنصة، وتم تعطيل LIVE_PLACE_SL_ORDER تلقائيًا لتفادي تكرار الأخطاء.\n"
                f"• Symbol: {escape_html(symbol)}\n"
                f"• Error: {escape_html(str(e)[:120])}",
                critical=False
            )
        return None

async def cancel_stop_loss_order(exchange, symbol: str, order_id: str) -> bool:
    try:
        await rate_limiter.wait_if_needed(weight=1)
        await exchange.cancel_order(order_id, symbol)
        log_order_audit("CANCEL_SL", symbol, 0, 0, "CANCELLED")
        rate_limiter.reset_errors()
        return True
    except Exception as e:
        logger.error(f"[LIVE Cancel SL Error] {symbol}: {str(e)[:150]}")
        rate_limiter.record_error()
        return False

async def wait_for_order_fill_or_cancel(exchange, symbol: str, order_id: str, timeout_sec: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
    start = time.time()
    last = None
    poll = max(1, int(CONFIG.get("ENTRY_LIMIT_POLL_SEC", 3)))
    while time.time() - start < timeout_sec:
        try:
            await rate_limiter.wait_if_needed(weight=1)
            last = await exchange.fetch_order(order_id, symbol)
            rate_limiter.reset_errors()
            status = last.get("status")
            filled = safe_float(last.get("filled"), 0.0)
            amount = safe_float(last.get("amount"), 0.0)
            if status == "closed" and amount > 0 and filled >= amount * 0.999:
                STATS["live_orders_filled"] += 1
                log_order_audit("LIMIT_BUY", symbol, last.get('price', 0), filled, "FILLED")
                logger.info(f"[FILL] Order filled for {symbol}: price={last.get('price', 0)}, amount={filled}")
                return True, last
            if status in ("canceled", "rejected", "expired"):
                log_order_audit("LIMIT_BUY", symbol, last.get('price', 0), filled, status)
                return False, last
        except Exception as e:
            rate_limiter.record_error()
        await asyncio.sleep(poll)
    try:
        await rate_limiter.wait_if_needed(weight=1)
        await exchange.cancel_order(order_id, symbol)
        STATS["live_orders_canceled"] += 1
        try:
            await rate_limiter.wait_if_needed(weight=1)
            last = await exchange.fetch_order(order_id, symbol)
            rate_limiter.reset_errors()
        except Exception:
            rate_limiter.record_error()
            pass
        log_order_audit("LIMIT_BUY", symbol, 0, 0, "TIMEOUT_CANCELED")
        logger.info(f"[FILL] Order {order_id} for {symbol} timed out and canceled")
        rate_limiter.reset_errors()
    except Exception as e:
        logger.warning(f"[LIVE Cancel Warn] {symbol}: {str(e)[:120]}")
        rate_limiter.record_error()
    return False, last

async def market_sell(exchange, symbol: str, amount: float) -> Optional[Dict[str, Any]]:
    try:
        amt = _round_amount_to_precision(exchange, symbol, amount)
        if amt <= 0:
            return None
        await rate_limiter.wait_if_needed(weight=1)
        order = await exchange.create_order(symbol, "market", "sell", amt)
        STATS["live_sells_executed"] += 1
        log_order_audit("MARKET_SELL", symbol, 0, amt, "EXECUTED")
        logger.info(f"[EXIT] Market sell executed for {symbol}: amount={amt}")
        rate_limiter.reset_errors()
        return order
    except Exception as e:
        STATS["live_order_errors"] += 1
        logger.error(f"[LIVE Sell Error] {symbol}: {str(e)[:150]}")
        log_order_audit("MARKET_SELL", symbol, 0, amount, f"ERROR: {str(e)[:50]}")
        rate_limiter.record_error()
        return None

async def market_sell_safe(exchange, symbol: str, amount: float, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    for attempt in range(max_retries):
        try:
            amt = _round_amount_to_precision(exchange, symbol, amount)
            if amt <= 0:
                return None
            order = await market_sell(exchange, symbol, amt)
            if order:
                return order
            await asyncio.sleep(1)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.critical(f"[LIVE Sell Critical] {symbol}: {str(e)[:200]}")
                STATS["live_emergencies"] += 1
                await send_telegram(
                    f"🚨 CRITICAL: فشل البيع بعد {max_retries} محاولات!\n\n"
                    f"الرمز: {escape_html(symbol)}\n"
                    f"الكمية: {amount}\n"
                    f"الخطأ: {escape_html(str(e)[:200])}\n\n"
                    f"⚠️ يرجى البيع يدوياً فوراً!",
                    critical=True
                )
            return None
        await asyncio.sleep(1)
    return None

# ===================== ENTRY QUALITY FILTER HELPERS =====================
def _get_entry_reference_zone(signal: 'QuantumSignal') -> Optional[Tuple[float, float, str]]:
    try:
        ms = signal.market_structure
        if ms and ms.order_block:
            zl = safe_float(ms.order_block.get("body_low", ms.order_block.get("low", 0.0)), 0.0)
            zh = safe_float(ms.order_block.get("body_high", ms.order_block.get("high", 0.0)), 0.0)
            if validate_price(zl) and validate_price(zh) and zh >= zl:
                return zl, zh, "ORDER_BLOCK"
        lg = signal.liquidity_grab
        if lg and lg.detected and lg.grab_type == "BULLISH":
            lvl = safe_float(lg.grab_level, 0.0)
            if validate_price(lvl):
                return lvl * 0.999, lvl * 1.001, "LIQUIDITY_GRAB"
    except Exception:
        pass
    return None

def entry_quality_filter_5m(df_5m: pd.DataFrame, signal: 'QuantumSignal') -> Tuple[bool, str]:
    if not CONFIG.get("ENABLE_ENTRY_QUALITY_FILTER", True):
        return True, ""
    try:
        if df_5m is None or len(df_5m) < 2:
            return False, "no_data"
        last = df_5m.iloc[-1]
        prev = df_5m.iloc[-2]
        atr_pct = safe_float(last.get("atr_pct", 0.0), 0.0)
        if atr_pct > safe_float(CONFIG.get("ENTRY_QUALITY_MAX_ATR_PCT_5M", 5.5), 5.5):
            return False, f"atr_pct_too_high({atr_pct:.2f}%)"
        bb_width = safe_float(last.get("bb_width", 0.0), 0.0)
        if bb_width > safe_float(CONFIG.get("ENTRY_QUALITY_MAX_BB_WIDTH_5M", 0.08), 0.08):
            return False, f"bb_width_too_high({bb_width:.3f})"
        price_change_2_candles = (last['close'] - prev['close']) / prev['close']
        if abs(price_change_2_candles) > 0.02:
            return False, "momentum_too_fast"
        zone = _get_entry_reference_zone(signal)
        if zone and 'atr' in df_5m.columns:
            zone_low, zone_high, zt = zone
            atr = safe_float(df_5m['atr'].iloc[-1], 0.0)
            if atr > 0:
                px = safe_float(df_5m['close'].iloc[-1], 0.0)
                dist = 0.0
                if px < zone_low:
                    dist = zone_low - px
                elif px > zone_high:
                    dist = px - zone_high
                max_dist = atr * safe_float(CONFIG.get("ENTRY_QUALITY_MAX_DISTANCE_FROM_ZONE_ATR", 1.2), 1.2)
                if dist > max_dist:
                    return False, f"too_far_from_{zt.lower()}(dist={dist:.6f},max={max_dist:.6f})"
        return True, ""
    except Exception:
        return True, ""

# ===================== INSTITUTIONAL RECALIBRATION =====================
def recalibrate_levels_on_fill(fill_price: float, signal: 'QuantumSignal') -> Tuple[float, float, float, float]:
    """
    تعيد (sl, tp1, tp2, position_size_usdt) بعد إعادة المعايرة.
    """
    if not validate_price(fill_price):
        return signal.sl, signal.tp1, signal.tp2, signal.position_size_usdt
    mode = CONFIG.get("LIVE_RECALIBRATION_MODE", "rr")
    orig_entry = safe_float(signal.entry, 0.0)
    orig_sl = safe_float(signal.sl, 0.0)
    orig_risk = orig_entry - orig_sl
    if orig_risk <= 0:
        return signal.sl, signal.tp1, signal.tp2, signal.position_size_usdt
    if mode == "rr":
        tp1_rr = (safe_float(signal.tp1) - orig_entry) / orig_risk if orig_risk > 0 else CONFIG["TP1_RR"]
        tp2_rr = (safe_float(signal.tp2) - orig_entry) / orig_risk if orig_risk > 0 else CONFIG["TP2_RR"]
        new_sl = fill_price - orig_risk
        if new_sl <= 0:
            logger.warning(f"[Recalibrate] SL would be negative: {new_sl:.6f}, using fallback")
            new_sl = fill_price * 0.95
        max_sl_distance = fill_price * (CONFIG["MAX_SL_PCT"] / 100)
        hard_sl = fill_price - max_sl_distance
        if new_sl < hard_sl:
            logger.info(f"[Recalibrate] SL clamped to MAX_SL_PCT: {new_sl:.6f} -> {hard_sl:.6f}")
            new_sl = hard_sl
        new_tp1 = fill_price + (orig_risk * tp1_rr)
        new_tp2 = fill_price + (orig_risk * tp2_rr)
        return new_sl, new_tp1, new_tp2, signal.position_size_usdt
    return signal.sl, signal.tp1, signal.tp2, signal.position_size_usdt

# ===================== ENHANCED DATABASE MANAGER =====================
db_manager = None

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection_pool_size = 5
        self.init_database()

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES,
                timeout=10.0,
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"[DB Error] {str(e)}")
            raise
        finally:
            if conn:
                conn.close()

    def init_database(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS active_trades (
                    symbol TEXT PRIMARY KEY,
                    entry REAL,
                    original_sl REAL,
                    current_sl REAL,
                    tp1 REAL,
                    tp2 REAL,
                    tp3 REAL,
                    atr REAL,
                    tp1_hit INTEGER,
                    tp2_hit INTEGER,
                    tp3_hit INTEGER,
                    remaining_position REAL,
                    be_moved INTEGER,
                    trailing_active INTEGER,
                    entry_time TEXT,
                    last_update TEXT,
                    total_realized_r REAL,
                    signal_data TEXT,
                    emergency_state INTEGER,
                    emergency_reason TEXT,
                    emergency_last_attempt TEXT,
                    emergency_attempts INTEGER,
                    version INTEGER DEFAULT 0,
                    sl_order_id TEXT,
                    order_block_low REAL DEFAULT 0,
                    order_block_high REAL DEFAULT 0,
                    liquidity_grab_level REAL DEFAULT 0
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    entry REAL,
                    exit REAL,
                    exit_type TEXT,
                    profit_pct REAL,
                    r_multiple REAL,
                    quantum_score REAL,
                    signal_class TEXT,
                    gates_passed TEXT,
                    entry_time TEXT,
                    exit_time TEXT,
                    execution_mode TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    date TEXT,
                    realized_r REAL,
                    blocked INTEGER,
                    last_updated TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON active_trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_exit_time ON trade_history(exit_time DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_version ON active_trades(version)")
            conn.commit()

    def save_daily_state(self, date: str, realized_r: float, blocked: bool):
        if not CONFIG["ENABLE_DB_PERSISTENCE"]:
            return
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO daily_state (id, date, realized_r, blocked, last_updated)
                    VALUES (1, ?, ?, ?, ?)
                """, (date, realized_r, int(blocked), now_utc_iso()))
        except Exception as e:
            logger.error(f"[DB Daily State Save Error] {str(e)}")

    def load_daily_state(self) -> Tuple[Optional[str], float, bool]:
        if not CONFIG["ENABLE_DB_PERSISTENCE"]:
            return None, 0.0, False
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT date, realized_r, blocked FROM daily_state WHERE id = 1")
                row = cursor.fetchone()
                if row:
                    return row["date"], safe_float(row["realized_r"]), bool(row["blocked"])
        except Exception as e:
            logger.error(f"[DB Daily State Load Error] {str(e)}")
        return None, 0.0, False

    def save_trade(self, trade_state: TradeState, signal_data: Optional[Dict] = None):
        if not CONFIG["ENABLE_DB_PERSISTENCE"]:
            return
        if not _validate_symbol(trade_state.symbol):
            logger.error(f"[DB] Invalid symbol format: {trade_state.symbol}")
            return
        try:
            signal_data_copy = copy.deepcopy(signal_data) if signal_data else None
            signal_data_json = json.dumps(signal_data_copy, default=str) if signal_data_copy else None
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO active_trades (
                        symbol,
                        entry,
                        original_sl,
                        current_sl,
                        tp1,
                        tp2,
                        tp3,
                        atr,
                        tp1_hit,
                        tp2_hit,
                        tp3_hit,
                        remaining_position,
                        be_moved,
                        trailing_active,
                        entry_time,
                        last_update,
                        total_realized_r,
                        signal_data,
                        emergency_state,
                        emergency_reason,
                        emergency_last_attempt,
                        emergency_attempts,
                        version,
                        sl_order_id,
                        order_block_low,
                        order_block_high,
                        liquidity_grab_level
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_state.symbol,
                    trade_state.entry,
                    trade_state.original_sl,
                    trade_state.current_sl,
                    trade_state.tp1,
                    trade_state.tp2,
                    0.0,  # tp3 غير مستخدم
                    trade_state.atr,
                    int(trade_state.tp1_hit),
                    int(trade_state.tp2_hit),
                    0,  # tp3_hit غير مستخدم
                    trade_state.remaining_position,
                    int(trade_state.be_moved),
                    int(trade_state.trailing_active),
                    trade_state.entry_time,
                    trade_state.last_update,
                    trade_state.total_realized_r,
                    signal_data_json,
                    int(trade_state.emergency_state),
                    trade_state.emergency_reason,
                    trade_state.emergency_last_attempt,
                    trade_state.emergency_attempts,
                    trade_state._version,
                    trade_state.sl_order_id,
                    trade_state.order_block_low,
                    trade_state.order_block_high,
                    trade_state.liquidity_grab_level
                ))
        except Exception as e:
            logger.error(f"[DB Save Error] {str(e)}")

    def remove_trade(self, symbol: str):
        if not CONFIG["ENABLE_DB_PERSISTENCE"]:
            return
        if not _validate_symbol(symbol):
            logger.error(f"[DB] Invalid symbol format: {symbol}")
            return
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM active_trades WHERE symbol = ?", (symbol,))
        except Exception as e:
            logger.error(f"[DB Remove Error] {str(e)}")

    def record_trade_history(self, symbol: str, trade_state: TradeState, exit_price: float, exit_type: str, r_multiple: float, execution_mode: str = ""):
        if not CONFIG["ENABLE_DB_PERSISTENCE"]:
            return
        if not _validate_symbol(symbol):
            logger.error(f"[DB] Invalid symbol format: {symbol}")
            return
        try:
            profit_pct = ((exit_price - trade_state.entry) / trade_state.entry) * 100
            signal_data = self.get_active_trade_signal_data(symbol)
            quantum_score = 0.0
            signal_class = ""
            gates_passed = []
            if signal_data:
                quantum_score = safe_float(signal_data.get("quantum_score"), 0.0)
                signal_class = signal_data.get("signal_class") or ""
                gates_passed = signal_data.get("gates_passed") or []
            gates_passed_json = json.dumps(gates_passed) if gates_passed else None
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO trade_history (
                        symbol,
                        entry,
                        exit,
                        exit_type,
                        profit_pct,
                        r_multiple,
                        quantum_score,
                        signal_class,
                        gates_passed,
                        entry_time,
                        exit_time,
                        execution_mode
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    trade_state.entry,
                    exit_price,
                    exit_type,
                    profit_pct,
                    r_multiple,
                    quantum_score,
                    signal_class,
                    gates_passed_json,
                    trade_state.entry_time,
                    now_utc_iso(),
                    execution_mode or (trade_state.execution_mode if hasattr(trade_state, 'execution_mode') else "")
                ))
        except Exception as e:
            logger.error(f"[DB History Error] {str(e)}")

    def load_active_trades(self) -> Dict[str, TradeState]:
        if not CONFIG["ENABLE_DB_PERSISTENCE"]:
            return {}
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM active_trades")
                rows = cursor.fetchall()
                trades = {}
                for row in rows:
                    sd = None
                    try:
                        raw = row["signal_data"]
                        if raw:
                            sd = json.loads(raw)
                    except Exception:
                        sd = None
                    version = row["version"] if "version" in row.keys() else 0
                    trade_state = TradeState(
                        symbol=row['symbol'],
                        entry=row['entry'],
                        original_sl=row['original_sl'],
                        current_sl=row['current_sl'],
                        tp1=row['tp1'],
                        tp2=row['tp2'],
                        atr=row['atr'],
                        tp1_hit=bool(row['tp1_hit']),
                        tp2_hit=bool(row['tp2_hit']),
                        remaining_position=row['remaining_position'],
                        be_moved=bool(row['be_moved']),
                        trailing_active=bool(row['trailing_active']),
                        entry_time=row['entry_time'],
                        last_update=row['last_update'],
                        total_realized_r=row['total_realized_r'],
                        quantum_score=safe_float(sd.get("quantum_score"), 0.0) if isinstance(sd, dict) else 0.0,
                        signal_class=(sd.get("signal_class") if isinstance(sd, dict) else "") or "",
                        gates_passed=(sd.get("gates_passed") if isinstance(sd, dict) and isinstance(sd.get("gates_passed"), list) else []),
                        emergency_state=bool(row['emergency_state']),
                        emergency_reason=row['emergency_reason'] or "",
                        emergency_last_attempt=row['emergency_last_attempt'] or "",
                        emergency_attempts=row['emergency_attempts'] or 0,
                        is_paper=bool(sd.get("is_paper")) if isinstance(sd, dict) else False,
                        execution_mode=(sd.get("execution_mode") if isinstance(sd, dict) else "") or ("PAPER" if not is_live_trading_enabled() else "LIVE"),
                        entry_assumed=bool(sd.get("entry_assumed")) if isinstance(sd, dict) else False,
                        _version=version,
                        sl_order_id=row['sl_order_id'] if 'sl_order_id' in row.keys() else "",
                        order_block_low=row['order_block_low'] if 'order_block_low' in row.keys() else 0.0,
                        order_block_high=row['order_block_high'] if 'order_block_high' in row.keys() else 0.0,
                        liquidity_grab_level=row['liquidity_grab_level'] if 'liquidity_grab_level' in row.keys() else 0.0,
                    )
                    trade_state.is_exiting = False
                    trades[row['symbol']] = trade_state
                logger.info(f"[DB] Loaded {len(trades)} active trades")
                return trades
        except Exception as e:
            logger.error(f"[DB Load Error] {str(e)}")
            return {}

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        if not CONFIG["ENABLE_DB_PERSISTENCE"]:
            return []
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT symbol, entry, exit, exit_type, profit_pct, r_multiple, quantum_score, signal_class, gates_passed, entry_time, exit_time, execution_mode
                    FROM trade_history
                    ORDER BY exit_time DESC
                    LIMIT ?
                """, (limit,))
                columns = [column[0] for column in cursor.description]
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"[DB History Load Error] {str(e)}")
            return []

    def get_active_trade_signal_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not CONFIG["ENABLE_DB_PERSISTENCE"]:
            return None
        if not _validate_symbol(symbol):
            logger.error(f"[DB] Invalid symbol format: {symbol}")
            return None
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT signal_data FROM active_trades WHERE symbol = ?", (symbol,))
                row = cursor.fetchone()
                if not row:
                    return None
                raw = row["signal_data"]
                if not raw:
                    return None
                try:
                    return json.loads(raw)
                except Exception:
                    return None
        except Exception as e:
            logger.error(f"[DB SignalData Error] {symbol}: {str(e)}")
            return None

    def update_trade_with_version(self, symbol: str, updates: Dict[str, Any], expected_version: int) -> bool:
        if not CONFIG["ENABLE_DB_PERSISTENCE"]:
            return False
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                set_clauses = []
                params = []
                for key, value in updates.items():
                    if key.startswith('_'):
                        continue
                    set_clauses.append(f"{key} = ?")
                    params.append(value)
                set_clauses.append("version = version + 1")
                params.append(symbol)
                params.append(expected_version)
                query = f"UPDATE active_trades SET {', '.join(set_clauses)} WHERE symbol = ? AND version = ?"
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"[DB Update Version Error] {symbol}: {str(e)}")
            return False

# ===================== DAILY CIRCUIT BREAKER =====================
class DailyCircuitBreaker:
    def __init__(self, db=None):
        self.db = db
        self.date = None
        self.realized_r = 0.0
        self.blocked = False
        if self.db is not None and CONFIG.get("ENABLE_DB_PERSISTENCE", True):
            self.load_state()

    def attach_db(self, db):
        self.db = db
        if self.db is not None and CONFIG.get("ENABLE_DB_PERSISTENCE", True):
            self.load_state()

    def load_state(self):
        if not CONFIG.get("ENABLE_DB_PERSISTENCE", True):
            return
        if self.db is None:
            logger.warning("[Daily Circuit] db_manager not ready yet; skipping load_state")
            return
        db_date, db_realized_r, db_blocked = self.db.load_daily_state()
        if db_date:
            self.date = db_date
            self.realized_r = db_realized_r
            self.blocked = db_blocked
            logger.info(f"[Daily Circuit] Loaded state: date={self.date}, R={self.realized_r:.2f}, blocked={self.blocked}")

    def save_state(self):
        if not CONFIG.get("ENABLE_DB_PERSISTENCE", True):
            return
        if self.db is None:
            return
        self.db.save_daily_state(self.date or "", self.realized_r, self.blocked)

    def reset_if_needed(self):
        now = datetime.now(timezone.utc)
        current_date = now.date().isoformat()
        if self.date is None:
            self.date = current_date
            self.save_state()
            return
        if current_date != self.date:
            logger.info(f"♻️ Daily reset - new date: {current_date}")
            self.date = current_date
            self.realized_r = 0.0
            self.blocked = False
            self.save_state()

    def record_daily_r(self, r_value: float):
        self.reset_if_needed()
        self.realized_r = safe_float(self.realized_r) + safe_float(r_value)
        max_loss = safe_float(CONFIG.get("DAILY_MAX_LOSS_R", -4.0))
        if self.realized_r <= max_loss:
            self.blocked = True
            logger.warning(f"🛑 Daily loss limit reached: {self.realized_r:.2f}R <= {max_loss}R")
            self.save_state()

    def is_blocked(self) -> bool:
        if not CONFIG.get("ENABLE_DAILY_MAX_LOSS", True):
            return False
        self.reset_if_needed()
        max_loss_r = safe_float(CONFIG.get("DAILY_MAX_LOSS_R", -4.0), -4.0)
        if self.blocked:
            return True
        if safe_float(self.realized_r, 0.0) <= max_loss_r:
            self.blocked = True
            self.save_state()
            return True
        return False

    def get_state(self) -> Dict:
        return {
            "date": self.date,
            "realized_r": self.realized_r,
            "blocked": self.blocked
        }

daily_circuit = DailyCircuitBreaker(db=None)

def record_daily_r(r_value: float):
    daily_circuit.record_daily_r(r_value)

def is_daily_loss_blocked() -> bool:
    return daily_circuit.is_blocked()

# ===================== CHECKPOINT SYSTEM =====================
async def checkpoint_saver():
    if not CONFIG.get("ENABLE_CHECKPOINTS", True):
        return
    interval = int(CONFIG.get("CHECKPOINT_INTERVAL_SEC", 300))
    path = CONFIG.get("CHECKPOINT_PATH", "/content/quantum_checkpoint.json")
    while not shutdown_manager.should_stop:
        try:
            await asyncio.sleep(interval)
            acquired_symbols = await bot.lock_manager.acquire_all_locks()
            if not acquired_symbols:
                logger.error("[Checkpoint] Failed to acquire locks, skipping")
                continue
            try:
                active_trades_copy = {k: asdict(v) for k, v in ACTIVE_TRADES.items()}
            finally:
                bot.lock_manager.release_all_locks(acquired_symbols)
            checkpoint = {
                'timestamp': now_utc_iso(),
                'active_trades': active_trades_copy,
                'stats': STATS.copy(),
                'loop_count': STATS.get('loop_count', 0),
                'config_live': bool(CONFIG.get("ENABLE_LIVE_TRADING", False)),
            }
            def write_checkpoint(path, data):
                with open(path, 'w') as f:
                    json.dump(data, f, default=str, indent=2)
            await asyncio.to_thread(write_checkpoint, path, checkpoint)
            logger.info(f"💾 Checkpoint saved - {len(active_trades_copy)} trades")
        except Exception as e:
            logger.error(f"Checkpoint save error: {e}")
            await asyncio.sleep(60)

def load_checkpoint() -> bool:
    if not CONFIG.get("ENABLE_CHECKPOINTS", True):
        return False
    path = CONFIG.get("CHECKPOINT_PATH", "/content/quantum_checkpoint.json")
    try:
        with open(path, 'r') as f:
            checkpoint = json.load(f)
        logger.info(f"♻️ Checkpoint loaded from {checkpoint.get('timestamp')}")
        STATS.update(checkpoint.get('stats', {}))
        recovered = 0
        at = checkpoint.get('active_trades', {}) or {}
        for symbol, trade_data in at.items():
            try:
                if 'order_block_low' not in trade_data:
                    trade_data['order_block_low'] = 0.0
                if 'order_block_high' not in trade_data:
                    trade_data['order_block_high'] = 0.0
                if 'liquidity_grab_level' not in trade_data:
                    trade_data['liquidity_grab_level'] = 0.0
                trade_data['is_exiting'] = False
                ACTIVE_TRADES[symbol] = TradeState(**trade_data)
                recovered += 1
            except Exception:
                continue
        logger.info(f"✅ Recovered {recovered} active trades from checkpoint")
        return recovered > 0
    except FileNotFoundError:
        logger.info("ℹ️ No checkpoint found - starting fresh")
        return False
    except Exception as e:
        logger.error(f"Checkpoint load error: {e}")
        return False

async def save_emergency_checkpoint_async(err: Exception):
    """نسخة غير متزامنة لحفظ checkpoint الطوارئ"""
    try:
        acquired = await bot.lock_manager.acquire_all_locks()
        if not acquired:
            logger.error("[Emergency] Failed to acquire locks for checkpoint")
            return
        try:
            active_trades_copy = {k: asdict(v) for k, v in ACTIVE_TRADES.items()}
        finally:
            bot.lock_manager.release_all_locks(acquired)
        checkpoint = {
            'timestamp': now_utc_iso(),
            'active_trades': active_trades_copy,
            'stats': STATS.copy(),
            'error': str(err)
        }
        path = CONFIG.get("EMERGENCY_CHECKPOINT_PATH", "/content/quantum_emergency_checkpoint.json")
        with open(path, 'w') as f:
            json.dump(checkpoint, f, default=str, indent=2)
        logger.info("💾 Emergency checkpoint saved (async)")
    except Exception as e:
        logger.error(f"Emergency checkpoint failed: {e}")

# دالة متزامنة للتوافق مع الكود القديم، تقوم بإنشاء مهمة
def save_emergency_checkpoint(err: Exception):
    asyncio.create_task(save_emergency_checkpoint_async(err))

# ===================== INDICATORS (مع إصلاح TALIB_FALLBACK) =====================
def calculate_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) < 20:
        return None
    try:
        df = df.copy()
        df = df[df['close'] > 0].copy()
        if len(df) < 20:
            return None
        TALIB_FALLBACK = False
        if TALIB_AVAILABLE:
            try:
                highs = df['high'].values
                lows = df['low'].values
                closes = df['close'].values
                df['ema9'] = talib.EMA(closes, timeperiod=9)
                df['ema21'] = talib.EMA(closes, timeperiod=21)
                df['ema50'] = talib.EMA(closes, timeperiod=50)
                df['ema200'] = talib.EMA(closes, timeperiod=200)
                df['atr'] = talib.ATR(highs, lows, closes, timeperiod=14)
                df['atr_pct'] = (df['atr'] / df['close'].replace(0, np.nan)) * 100
                df['atr_pct'] = df['atr_pct'].fillna(0)
                df['rsi'] = talib.RSI(closes, timeperiod=14)
                df['rsi'] = df['rsi'].fillna(50)
                macd, macdsignal, macdhist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
                df['macd'] = pd.Series(macdhist, index=df.index).fillna(0)
            except Exception:
                TALIB_FALLBACK = True

        if not TALIB_AVAILABLE or TALIB_FALLBACK:
            if len(df) >= 9:
                df['ema9'] = ta.trend.ema_indicator(df['close'], 9)
            else:
                df['ema9'] = df['close']
            if len(df) >= 21:
                df['ema21'] = ta.trend.ema_indicator(df['close'], 21)
            else:
                df['ema21'] = df['close']
            if len(df) >= 50:
                df['ema50'] = ta.trend.ema_indicator(df['close'], 50)
            else:
                df['ema50'] = df['close']
            if len(df) >= 200:
                df['ema200'] = ta.trend.ema_indicator(df['close'], 200)
            else:
                df['ema200'] = df['close']
            if len(df) >= 14:
                df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
                df['atr_pct'] = (df['atr'] / df['close'].replace(0, np.nan)) * 100
                df['atr_pct'] = df['atr_pct'].fillna(0)
            else:
                df['atr'] = 0
                df['atr_pct'] = 0
            if len(df) >= 14:
                df['rsi'] = ta.momentum.rsi(df['close'], 14)
                df['rsi'] = df['rsi'].fillna(50)
            else:
                df['rsi'] = 50
            if len(df) >= 26:
                df['macd'] = ta.trend.macd_diff(df['close'])
                df['macd'] = df['macd'].fillna(0)
            else:
                df['macd'] = 0
            if len(df) >= 20:
                df['volume_sma'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan)
                df['volume_ratio'] = df['volume_ratio'].fillna(1)
            else:
                df['volume_sma'] = df['volume']
                df['volume_ratio'] = 1
            if len(df) >= 20:
                bb = ta.volatility.BollingerBands(df['close'], 20, 2)
                df['bb_upper'] = bb.bollinger_hband()
                df['bb_lower'] = bb.bollinger_lband()
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close'].replace(0, np.nan)
                df['bb_width'] = df['bb_width'].fillna(0)
            else:
                df['bb_upper'] = df['close']
                df['bb_lower'] = df['close']
                df['bb_width'] = 0
            if len(df) >= 14:
                df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], 14)
                df['adx'] = df['adx'].fillna(25)
            else:
                df['adx'] = 25

        df = df.ffill().bfill()
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32]:
                if col.startswith('ema'):
                    df[col] = df[col].fillna(df['close'])
                elif 'ratio' in col.lower() or 'pct' in col.lower():
                    df[col] = df[col].fillna(1 if 'ratio' in col.lower() else 0)
                else:
                    df[col] = df[col].fillna(0)
        if df.isna().any().any():
            nan_cols = df.columns[df.isna().any()].tolist()
            logger.warning(f"[Indicators] Still NaN in columns: {nan_cols}")
            df = df.dropna()
        if len(df) < 20:
            return None
        return df
    except Exception as e:
        logger.error(f"[Indicators Error] {str(e)}")
        return None

# ===================== MARKET STRUCTURE ANALYSIS =====================
def analyze_market_structure(df: pd.DataFrame) -> Optional[MarketStructure]:
    if df is None or len(df) < 50:
        return None
    try:
        window = CONFIG["SWING_WINDOW"]
        high_indices = []
        low_indices = []
        for i in range(window, len(df) - window):
            if all(df['high'].iloc[i] >= df['high'].iloc[i-j] and
                   df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window+1)):
                high_indices.append(i)
            if all(df['low'].iloc[i] <= df['low'].iloc[i-j] and
                   df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window+1)):
                low_indices.append(i)
        if not high_indices or not low_indices:
            return None
        swing_high = safe_float(df['high'].iloc[high_indices[-1]])
        swing_low = safe_float(df['low'].iloc[low_indices[-1]])
        prev_swing_high = safe_float(df['high'].iloc[high_indices[-2]] if len(high_indices) > 1 else swing_high)
        prev_swing_low = safe_float(df['low'].iloc[low_indices[-2]] if len(low_indices) > 1 else swing_low)
        if not all(validate_price(p) for p in [swing_high, swing_low, prev_swing_high, prev_swing_low]):
            return None
        current_price = safe_float(df['close'].iloc[-1])
        bos_confirm_candles = CONFIG["BOS_CONFIRMATION_CANDLES"]
        recent_high = safe_float(df['high'].iloc[-bos_confirm_candles:].max())
        recent_low = safe_float(df['low'].iloc[-bos_confirm_candles:].min())
        recent_close = safe_float(df['close'].iloc[-1])
        mult = CONFIG["BOS_CONFIRMATION_MULTIPLIER"]
        bos_bullish = (recent_high > prev_swing_high and recent_close > prev_swing_high * mult)
        bos_bearish = (recent_low < prev_swing_low and recent_close < prev_swing_low * mult)
        choch = False
        if len(high_indices) >= 3 and len(low_indices) >= 3:
            immediate_high = safe_float(df['high'].iloc[high_indices[-2]])
            immediate_low = safe_float(df['low'].iloc[low_indices[-2]])
            if current_price > immediate_high and not bos_bullish:
                choch = True
            elif current_price < immediate_low and not bos_bearish:
                choch = True
        order_block = None
        lookback = CONFIG["ORDER_BLOCK_LOOKBACK"]
        if len(df) > lookback:
            for i in range(len(df) - 1, len(df) - lookback - 1, -1):
                candle = df.iloc[i]
                if candle['close'] < candle['open']:
                    continue
                if i > 0 and df.iloc[i-1]['close'] > df.iloc[i-1]['open']:
                    continue
                volume_spike = candle['volume'] > df['volume'].iloc[max(0, i-10):i].mean() * 1.3
                displacement_found = False
                for j in range(i+1, min(i+6, len(df))):
                    if df.iloc[j]['high'] > candle['high'] * 1.005:
                        displacement_found = True
                        break
                is_fresh = (len(df) - 1 - i) < CONFIG["HARD_GATE_2_OB_FRESHNESS"]
                if volume_spike and displacement_found and is_fresh:
                    order_block = {
                        'high': safe_float(candle['high']),
                        'low': safe_float(candle['low']),
                        'body_high': safe_float(max(candle['open'], candle['close'])),
                        'body_low': safe_float(min(candle['open'], candle['close'])),
                        'index': i,
                        'freshness': len(df) - 1 - i
                    }
                    break
        fvg_zone = None
        if len(df) > 3:
            for i in range(len(df) - 2, 0, -1):
                prev_candle = df.iloc[i-1]
                current_candle = df.iloc[i]
                if prev_candle['high'] < current_candle['low']:
                    gap_low = prev_candle['high']
                    gap_high = current_candle['low']
                    min_gap_size = safe_float(df['atr'].iloc[i] * CONFIG["FVG_MIN_SIZE_ATR"])
                    if (gap_high - gap_low) >= min_gap_size:
                        fvg_zone = {
                            'high': gap_high,
                            'low': gap_low,
                            'type': 'BULLISH',
                            'index': i
                        }
                        break
                elif prev_candle['low'] > current_candle['high']:
                    gap_low = current_candle['high']
                    gap_high = prev_candle['low']
                    min_gap_size = safe_float(df['atr'].iloc[i] * CONFIG["FVG_MIN_SIZE_ATR"])
                    if (gap_high - gap_low) >= min_gap_size:
                        fvg_zone = {
                            'high': gap_high,
                            'low': gap_low,
                            'type': 'BEARISH',
                            'index': i
                        }
                        break
        window_liquidity = df.iloc[-21:-1] if len(df) >= 21 else df.iloc[:-1]
        liquidity_high = safe_float(window_liquidity['high'].max()) if len(window_liquidity) > 0 else 0
        liquidity_low = safe_float(window_liquidity['low'].min()) if len(window_liquidity) > 0 else 0
        structure = "NEUTRAL"
        if bos_bullish or (swing_high > prev_swing_high and swing_low > prev_swing_low):
            structure = "BULLISH"
        elif bos_bearish or (swing_high < prev_swing_high and swing_low < prev_swing_low):
            structure = "BEARISH"
        ema_alignment = 0
        if 'ema9' in df.columns and 'ema21' in df.columns and 'ema50' in df.columns:
            if df['ema9'].iloc[-1] > df['ema21'].iloc[-1] > df['ema50'].iloc[-1]:
                ema_alignment = 100
            elif df['ema9'].iloc[-1] > df['ema21'].iloc[-1]:
                ema_alignment = 66
            elif df['ema9'].iloc[-1] > df['ema50'].iloc[-1]:
                ema_alignment = 33
        adx_strength = min(safe_float(df['adx'].iloc[-1]) / 50 * 100, 100) if 'adx' in df.columns else 50
        price_position = 0
        if 'ema50' in df.columns and current_price > df['ema50'].iloc[-1]:
            price_position = 20
        trend_strength = safe_float(ema_alignment * 0.5 + adx_strength * 0.3 + price_position * 0.2)
        return MarketStructure(
            structure=structure,
            bos_bullish=bos_bullish,
            choch=choch,
            order_block=order_block,
            fvg_zone=fvg_zone,
            liquidity_high=liquidity_high,
            liquidity_low=liquidity_low,
            swing_high=swing_high,
            swing_low=swing_low,
            trend_strength=trend_strength
        )
    except Exception as e:
        logger.error(f"[Market Structure Error] {str(e)}")
        return None

# ===================== ENHANCED ORDER FLOW ANALYSIS =====================
@metrics.record_latency("order_flow_analysis")
async def analyze_order_flow(exchange, symbol: str, mtf_alignment: int = 0, allow_low_alignment: bool = False) -> Optional[OrderFlowData]:
    if not CONFIG["ORDER_FLOW_ENABLED"]:
        return None
    if CONFIG["ORDER_FLOW_ONLY_HIGH_ALIGNMENT"] and mtf_alignment < 3 and not allow_low_alignment:
        return None
    if CONFIG.get("CIRCUIT_BREAKER_ENABLED", True) and not api_circuit.can_attempt():
        logger.warning(f"[Circuit Breaker] Skipping order flow for {symbol}")
        return None
    try:
        await rate_limiter.wait_if_needed(weight=CONFIG.get("ORDERBOOK_WEIGHT", 10))
        orderbook = await exchange.fetch_order_book(symbol, CONFIG["ORDERBOOK_DEPTH"])
        rate_limiter.reset_errors()
        bids = orderbook.get("bids") or []
        asks = orderbook.get("asks") or []
        if not bids or not asks:
            return None
        bid_volume = sum(bid[1] for bid in bids[:CONFIG["ORDERBOOK_WEIGHTED_LEVELS"]])
        ask_volume = sum(ask[1] for ask in asks[:CONFIG["ORDERBOOK_WEIGHTED_LEVELS"]])
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return None
        imbalance = (bid_volume - ask_volume) / total_volume
        try:
            await rate_limiter.wait_if_needed(weight=5)
            trades = await exchange.fetch_trades(symbol, limit=CONFIG["TRADES_SAMPLE_SIZE"])
            rate_limiter.reset_errors()
            buy_volume = 0
            sell_volume = 0
            for trade in trades:
                if trade.get('side') == 'buy':
                    buy_volume += trade.get('amount', 0)
                else:
                    sell_volume += trade.get('amount', 0)
            total_trade_volume = buy_volume + sell_volume
            delta = (buy_volume - sell_volume) / total_trade_volume if total_trade_volume > 0 else 0
        except Exception:
            delta = 0
            rate_limiter.record_error()
        bid_strength = bid_volume / total_volume if total_volume > 0 else 0.5
        ask_strength = ask_volume / total_volume if total_volume > 0 else 0.5
        divergence = abs(imbalance - delta) > CONFIG["DIVERGENCE_THRESHOLD"]
        volume_profile = "NEUTRAL"
        if imbalance > CONFIG["IMBALANCE_THRESHOLD"] and delta > CONFIG["DELTA_THRESHOLD"]:
            volume_profile = "AGGRESSIVE_BUYING"
        elif imbalance > CONFIG["IMBALANCE_THRESHOLD"] and delta < -CONFIG["DELTA_THRESHOLD"]:
            volume_profile = "AGGRESSIVE_SELLING"
        elif imbalance > CONFIG["IMBALANCE_THRESHOLD"] and abs(delta) < CONFIG["DELTA_THRESHOLD"]:
            volume_profile = "ABSORPTION"
        elif imbalance < -CONFIG["IMBALANCE_THRESHOLD"] and abs(delta) < CONFIG["DELTA_THRESHOLD"]:
            volume_profile = "DISTRIBUTION"
        signal = "NEUTRAL"
        confidence = 50.0
        if volume_profile == "AGGRESSIVE_BUYING":
            signal = "BULLISH"
            confidence = 70.0
        elif volume_profile == "DISTRIBUTION":
            signal = "BEARISH"
            confidence = 75.0
        elif volume_profile == "AGGRESSIVE_SELLING":
            signal = "BEARISH"
            confidence = 70.0
        elif volume_profile == "ABSORPTION":
            if CONFIG.get("ORDER_FLOW_ABSORPTION_LOGIC_V2", True):
                if imbalance > 0.30 and delta < 0.10:
                    signal = "BULLISH"
                    confidence = 65.0
                elif imbalance < -0.30 and delta > -0.10:
                    signal = "BEARISH"
                    confidence = 65.0
            else:
                if imbalance > 0.20 and delta < 0.05:
                    signal = "BULLISH"
                    confidence = 60.0
                elif not divergence:
                    if imbalance > CONFIG["IMBALANCE_THRESHOLD"]:
                        signal = "BULLISH"
                        confidence = 55.0
                    elif imbalance < -CONFIG["IMBALANCE_THRESHOLD"]:
                        signal = "BEARISH"
                        confidence = 55.0
        api_circuit.record_success()
        return OrderFlowData(
            imbalance=imbalance,
            delta=delta,
            divergence=divergence,
            bid_strength=bid_strength,
            ask_strength=ask_strength,
            volume_profile=volume_profile,
            signal=signal,
            confidence=confidence
        )
    except Exception as e:
        api_circuit.record_failure()
        rate_limiter.record_error()
        await metrics.record_error("order_flow_analysis", type(e).__name__)
        logger.error(f"[Order Flow Error] {symbol}: {str(e)[:100]}")
        return None

# ===================== VOLUME PROFILE ANALYSIS =====================
def analyze_volume_profile(df: pd.DataFrame, precheck_score: float = 0.0) -> Optional[VolumeProfileData]:
    if not CONFIG.get("ENABLE_VOLUME_PROFILE", False) or df is None or len(df) < 50:
        return None
    # نستخدم متغيراً بديلاً لأخذ العينات
    if not CONFIG.get("_VOLUME_PROFILE_SAMPLING_OK", True):
        return None
    try:
        if len(df) > 100:
            df = df.iloc[-100:]
        close_prices = df['close'].values
        volumes = df['volume'].values
        if len(close_prices) == 0:
            return None
        price_min = np.min(close_prices)
        price_max = np.max(close_prices)
        if price_max <= price_min:
            return None
        bins = min(CONFIG["VOLUME_PROFILE_BINS"], 20)
        hist, bin_edges = np.histogram(close_prices, bins=bins, weights=volumes, density=False)
        if np.sum(hist) == 0:
            return None
        poc_idx = np.argmax(hist)
        poc = safe_float(bin_edges[poc_idx])
        total_volume = np.sum(hist)
        target_volume = total_volume * CONFIG["VALUE_AREA_PCT"]
        sorted_indices = np.argsort(hist)[::-1]
        cumulative_volume = 0
        value_area_indices = []
        for idx in sorted_indices:
            cumulative_volume += hist[idx]
            value_area_indices.append(idx)
            if cumulative_volume >= target_volume:
                break
        vah = safe_float(bin_edges[max(value_area_indices) + 1])
        val = safe_float(bin_edges[min(value_area_indices)])
        volume_mean = hist.mean()
        volume_std = hist.std() if hist.std() > 0 else 1.0
        hvn_threshold = volume_mean + CONFIG["HVN_THRESHOLD"] * volume_std
        lvn_threshold = volume_mean - CONFIG["LVN_THRESHOLD"] * volume_std
        hvn_levels = [safe_float(bin_edges[i]) for i in range(len(hist)) if hist[i] > hvn_threshold]
        lvn_levels = [safe_float(bin_edges[i]) for i in range(len(hist)) if hist[i] < lvn_threshold]
        subset = df.iloc[-100:] if len(df) > 100 else df
        if len(subset) < 5:
            return None
        df_copy = subset.copy()
        vol_cum = df_copy['volume'].cumsum().replace(0, np.nan)
        df_copy['vwap'] = (df_copy['close'] * df_copy['volume']).cumsum() / vol_cum
        df_copy['vwap'] = df_copy['vwap'].ffill().bfill().fillna(df_copy['close'])
        df_copy['vwap_diff_sq'] = ((df_copy['close'] - df_copy['vwap']) ** 2) * df_copy['volume']
        vwap_var = df_copy['vwap_diff_sq'].cumsum() / vol_cum
        vwap_var = vwap_var.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        vwap_std = np.sqrt(safe_float(vwap_var.iloc[-1]))
        vwap = safe_float(df_copy['vwap'].iloc[-1])
        vwap_upper = safe_float(vwap + 2 * vwap_std)
        vwap_lower = safe_float(vwap - 2 * vwap_std)
        current_price = safe_float(df['close'].iloc[-1])
        if current_price > vah:
            position = "ABOVE_VALUE"
        elif current_price >= val:
            position = "IN_VALUE"
        else:
            position = "BELOW_VALUE"
        recent_volume = safe_float(subset['volume'].iloc[-10:].mean())
        older_volume = safe_float(subset['volume'].iloc[-30:-10].mean())
        if older_volume == 0:
            volume_trend = "NEUTRAL"
        elif recent_volume > older_volume * 1.2:
            volume_trend = "INCREASING"
        elif recent_volume < older_volume * 0.8:
            volume_trend = "DECREASING"
        else:
            volume_trend = "NEUTRAL"
        return VolumeProfileData(
            poc=poc,
            vah=vah,
            val=val,
            current_position=position,
            vwap=vwap,
            vwap_upper=vwap_upper,
            vwap_lower=vwap_lower,
            hvn_levels=hvn_levels[:5],
            lvn_levels=lvn_levels[:5],
            volume_trend=volume_trend
        )
    except Exception as e:
        logger.error(f"[Volume Profile Error] {str(e)}")
        return None

# ===================== LIQUIDITY GRAB DETECTION =====================
def detect_liquidity_grab(df: pd.DataFrame, symbol: str = "") -> Optional[LiquidityGrab]:
    if not CONFIG.get("ENABLE_LIQUIDITY_GRAB", True) or df is None or len(df) < 30:
        return None
    try:
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2] if len(df) > 1 else last_candle
        window = df.iloc[-21:-1] if len(df) >= 21 else df.iloc[:-1]
        if len(window) == 0:
            return None
        support = safe_float(window['low'].min())
        resistance = safe_float(window['high'].max())
        current_atr = safe_float(df['atr'].iloc[-1]) if 'atr' in df.columns else 0
        equal_lows = False
        equal_lows_range = 0.0
        if 'low' in df.columns:
            recent_lows = df['low'].iloc[-8:].tolist()
            if len(recent_lows) >= 5:
                required = int(CONFIG["LG_EQUAL_LOWS_REQUIRED"])
                level = float(np.median(recent_lows[-5:]))
                tol = current_atr * CONFIG["LG_EQUAL_LOWS_RANGE_ATR_MULT"]
                touches = sum(1 for x in recent_lows[-5:] if abs(x - level) <= tol)
                equal_lows = touches >= required
                equal_lows_range = tol * 2
        avg_volume = safe_float(df['volume'].iloc[-20:-1].mean())
        volume_spike = False
        if avg_volume > 0:
            volume_spike = last_candle['volume'] > avg_volume * CONFIG["LG_VOLUME_MULTIPLIER"]
        candle_range = safe_float(last_candle['high'] - last_candle['low'])
        if candle_range == 0:
            return None
        lower_wick = safe_float(min(last_candle['open'], last_candle['close']) - last_candle['low'])
        wick_ratio = lower_wick / candle_range
        recovery = safe_float((last_candle['close'] - last_candle['low']) / candle_range)
        sweep_candle_close_in_range = (last_candle['close'] > support and last_candle['close'] < resistance)
        bullish_grab = (
            last_candle['low'] < support * 0.998 and
            volume_spike and
            wick_ratio >= CONFIG["LG_WICK_MIN_RATIO"] and
            recovery >= CONFIG["LG_RECOVERY_MIN"] and
            sweep_candle_close_in_range
        )
        broke_resistance = last_candle['high'] > resistance * 1.002
        upper_wick = safe_float(last_candle['high'] - max(last_candle['open'], last_candle['close']))
        upper_wick_ratio = upper_wick / candle_range
        bearish_grab = (
            broke_resistance and
            volume_spike and
            upper_wick_ratio >= CONFIG["LG_WICK_MIN_RATIO"] and
            last_candle['close'] < resistance * 0.998
        )
        if not (bullish_grab or bearish_grab):
            return None
        grab_type = "BULLISH" if bullish_grab else "BEARISH"
        grab_level = support if bullish_grab else resistance
        confidence = 50.0
        if bullish_grab:
            confidence = min(
                wick_ratio * 100 + recovery * 50 + (last_candle['volume'] / avg_volume / CONFIG["LG_VOLUME_MULTIPLIER"]) * 20,
                100
            )
            if equal_lows and len(df) >= CONFIG["LG_EQUAL_LOWS_REQUIRED"]:
                confidence = min(confidence * 1.2, 100)
            if confidence < CONFIG["HARD_GATE_2_MIN_LG_CONFIDENCE"]:
                return None
            sweep_idx_in_recent = len(df) - 1
            sweep_timestamp = int(df['t'].iloc[-1]) if 't' in df.columns else None
            if symbol:
                logger.info(f"[LIQUIDITY] {symbol} bullish grab detected with confidence {confidence:.1f}")
            else:
                logger.info(f"[LIQUIDITY] bullish grab detected with confidence {confidence:.1f}")
            return LiquidityGrab(
                detected=True,
                grab_type=grab_type,
                grab_level=grab_level,
                wick_strength=wick_ratio,
                volume_spike=last_candle['volume'] / avg_volume if avg_volume > 0 else 1.0,
                recovery_strength=recovery,
                confidence=confidence,
                equal_lows=equal_lows and len(df) >= CONFIG["LG_EQUAL_LOWS_REQUIRED"],
                equal_lows_range=equal_lows_range,
                sweep_candle_close=last_candle['close'],
                sweep_index=sweep_idx_in_recent,
                sweep_timestamp=sweep_timestamp
            )
        elif bearish_grab:
            confidence = min(
                upper_wick_ratio * 100 + (last_candle['volume'] / avg_volume / CONFIG["LG_VOLUME_MULTIPLIER"]) * 20,
                100
            )
            if confidence < CONFIG["HARD_GATE_2_MIN_LG_CONFIDENCE"]:
                return None
            return LiquidityGrab(
                detected=True,
                grab_type=grab_type,
                grab_level=grab_level,
                wick_strength=upper_wick_ratio,
                volume_spike=last_candle['volume'] / avg_volume if avg_volume > 0 else 1.0,
                recovery_strength=0.0,
                confidence=confidence
            )
    except Exception as e:
        logger.error(f"[Liquidity Grab Error] {str(e)}")
        return None
    return None

# ===================== MULTI TIMEFRAME ANALYSIS =====================
@metrics.record_latency("mtf_analysis")
async def analyze_multi_timeframe(exchange, symbol: str) -> Optional[Dict]:
    try:
        tasks = [
            cache.get_ohlcv(exchange, symbol, CONFIG["TF_TREND"], 150),
            cache.get_ohlcv(exchange, symbol, CONFIG["TF_STRUCTURE"], 150),
            cache.get_ohlcv(exchange, symbol, CONFIG["TF_ENTRY"], 150),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        data_1h, data_15m, data_5m = results
        if any(isinstance(r, Exception) or r is None for r in results):
            return None
        df_1h = calculate_indicators(pd.DataFrame(data_1h, columns=['t','open','high','low','close','volume']))
        df_15m = calculate_indicators(pd.DataFrame(data_15m, columns=['t','open','high','low','close','volume']))
        df_5m = calculate_indicators(pd.DataFrame(data_5m, columns=['t','open','high','low','close','volume']))
        if not all(df is not None for df in [df_1h, df_15m, df_5m]):
            return None
        if CONFIG.get("ENABLE_MARKET_REGIME_FILTER", True):
            if df_15m is not None and len(df_15m) > 0:
                adx_value = df_15m['adx'].iloc[-1]
                if adx_value < CONFIG.get("MIN_ADX_FOR_TREND", 20):
                    logger.info(f"[Regime Filter] {symbol}: ADX too low ({adx_value:.1f}) - skipping")
                    return None
        structure_1h = analyze_market_structure(df_1h)
        structure_15m = analyze_market_structure(df_15m)
        structure_5m = analyze_market_structure(df_5m)
        if not all(s is not None for s in [structure_1h, structure_15m, structure_5m]):
            return None
        if CONFIG["LONG_ONLY"] and structure_1h.structure != "BULLISH":
            return None
        if structure_1h.trend_strength < 50:
            return None
        alignment = 0
        if structure_1h.structure == "BULLISH":
            alignment += 1
        if structure_15m.structure == "BULLISH":
            alignment += 1
        if structure_5m.structure == "BULLISH":
            alignment += 1
        liquidity_grab = detect_liquidity_grab(df_5m, symbol)
        return {
            'structure_1h': structure_1h,
            'structure_15m': structure_15m,
            'structure_5m': structure_5m,
            'df_1h': df_1h,
            'df_15m': df_15m,
            'df_5m': df_5m,
            'liquidity_grab': liquidity_grab,
            'alignment': alignment,
            'signal': "ENTER" if alignment >= CONFIG["MIN_MTF_ALIGNMENT"] else "WAIT"
        }
    except Exception as e:
        logger.error(f"[MTF Error] {symbol}: {str(e)[:150]}")
        return None

# ===================== INSTITUTIONAL HARD GATES =====================
def evaluate_hard_gates(
        market_structure: MarketStructure,
        order_flow: Optional[OrderFlowData],
        volume_profile: Optional[VolumeProfileData],
        liquidity_grab: Optional[LiquidityGrab],
        mtf_alignment: int,
        df: pd.DataFrame
) -> Tuple[bool, List[str]]:
    if not CONFIG["ENABLE_HARD_GATES"]:
        return True, []
    gates_passed = []
    all_gates_passed = True

    # GATE 1: Trend strength and MTF alignment
    if mtf_alignment == 3:
        gate1_passed = (market_structure.trend_strength >= CONFIG["HARD_GATE_1_MIN_TREND_STRENGTH"] and
                        market_structure.structure == "BULLISH")
    elif mtf_alignment == 2:
        gate1_passed = (market_structure.trend_strength > 70 and
                        market_structure.structure == "BULLISH")
    else:
        gate1_passed = False
    if gate1_passed:
        gates_passed.append("GATE_1_TREND")
    else:
        all_gates_passed = False
        if CONFIG["DEBUG_MODE"]:
            logger.info(f"[Hard Gates] GATE_1 failed: Trend={market_structure.trend_strength:.1f}, "
                        f"MTF={mtf_alignment}, Structure={market_structure.structure}")

    # GATE 2: Strong entry zone (Liquidity Grab or fresh Order Block with BOS)
    has_strong_lg = (liquidity_grab and liquidity_grab.detected and
                     liquidity_grab.confidence >= CONFIG.get("HARD_GATE_2_MIN_LG_CONFIDENCE", 70) and
                     liquidity_grab.grab_type == "BULLISH")
    has_fresh_ob = (market_structure.order_block and
                    market_structure.order_block.get("freshness", 999) <= CONFIG.get("HARD_GATE_2_OB_FRESHNESS", 8) and
                    market_structure.trend_strength >= 65 and
                    market_structure.bos_bullish)
    gate2_passed = has_strong_lg or has_fresh_ob
    if gate2_passed:
        if has_strong_lg:
            gates_passed.append("GATE_2_STRONG_LIQUIDITY_GRAB")
        if has_fresh_ob:
            gates_passed.append("GATE_2_FRESH_ORDER_BLOCK")
    else:
        all_gates_passed = False
        if CONFIG["DEBUG_MODE"]:
            logger.info(f"[Hard Gates] GATE_2 failed: No strong Liquidity Grab or fresh Order Block with BOS")

    # GATE 3: Boosters (optional)
    if CONFIG["ENABLE_HARD_GATES"] and CONFIG["HARD_GATE_3_REQUIRE_BOOSTER"]:
        boosters = []
        if order_flow and order_flow.signal == "BULLISH" and order_flow.confidence >= 70:
            boosters.append("ORDER_FLOW")
            gates_passed.append("BOOSTER_ORDER_FLOW")
        if volume_profile and volume_profile.current_position == "BELOW_VALUE":
            boosters.append("VOLUME_PROFILE")
            gates_passed.append("BOOSTER_VOLUME_PROFILE")
        if 'rsi' in df.columns:
            rsi = safe_float(df['rsi'].iloc[-1])
            if 30 < rsi < 70:
                boosters.append("RSI")
                gates_passed.append("BOOSTER_RSI")
        gate3_passed = len(boosters) >= 1
        if not gate3_passed:
            all_gates_passed = False
            if CONFIG["DEBUG_MODE"]:
                logger.info(f"[Hard Gates] GATE_3 failed: No boosters")

    if all_gates_passed:
        STATS["hard_gates_passed"] += 1
    else:
        STATS["hard_gates_failed"] += 1
    return all_gates_passed, gates_passed

# ===================== ENHANCED QUANTUM SCORING =====================
def calculate_quantum_score(
        market_structure: MarketStructure,
        order_flow: Optional[OrderFlowData],
        volume_profile: Optional[VolumeProfileData],
        liquidity_grab: Optional[LiquidityGrab],
        mtf_alignment: int,
        df: pd.DataFrame,
        gates_passed: Optional[List[str]] = None
) -> Tuple[float, float, str]:
    if gates_passed is None:
        gates_ok, gates_passed = evaluate_hard_gates(
            market_structure, order_flow, volume_profile, liquidity_grab, mtf_alignment, df
        )
        if CONFIG["ENABLE_HARD_GATES"] and not gates_ok:
            return 0.0, 0.0, "REJECT"
    score = 0.0
    confidence_factors = []

    # 1. Liquidity Grab (max 25)
    if liquidity_grab and liquidity_grab.detected and liquidity_grab.grab_type == "BULLISH":
        lg_score = 15 + min(liquidity_grab.confidence * 0.1, 10)
        if hasattr(liquidity_grab, 'equal_lows') and liquidity_grab.equal_lows:
            lg_score = min(lg_score + 3, 25)
        score += min(lg_score, 25)
        confidence_factors.append(liquidity_grab.confidence)

    # 2. Order Block (max 20)
    if market_structure.order_block:
        ob_score = 10
        freshness = market_structure.order_block.get('freshness', 100)
        if freshness < 5:
            ob_score += 10
        elif freshness < 10:
            ob_score += 5
        ob_score = min(ob_score, 20)
        score += ob_score
        confidence_factors.append(80)

    # 3. Break of Structure (max 15)
    if market_structure.bos_bullish:
        bos_score = 15
        score += bos_score
        confidence_factors.append(90)

    # 4. Order Flow (max 15)
    if order_flow:
        of_score = 0
        if order_flow.signal == "BULLISH":
            of_score += 5
        if order_flow.volume_profile == "AGGRESSIVE_BUYING":
            of_score += 10
        elif order_flow.volume_profile == "ABSORPTION":
            of_score += 10
        elif order_flow.volume_profile == "DISTRIBUTION":
            of_score -= 5
        if order_flow.imbalance > 0.3:
            of_score += 2
        of_score = max(0, min(15, of_score))
        score += of_score
        confidence_factors.append(order_flow.confidence)

    # 5. MTF Alignment (max 10)
    alignment_score = mtf_alignment * 3.33
    alignment_score = min(10, alignment_score)
    score += alignment_score
    confidence_factors.append(50 + mtf_alignment * 15)

    # 6. Volume Profile (max 10)
    if volume_profile:
        vp_score = 0
        if volume_profile.current_position == "BELOW_VALUE":
            vp_score = 10
        elif volume_profile.current_position == "IN_VALUE":
            vp_score = 5
        score += vp_score
        confidence_factors.append(75 if vp_score > 0 else 50)

    # 7. RSI (max 5)
    if 'rsi' in df.columns:
        rsi = safe_float(df['rsi'].iloc[-1])
        if 30 < rsi < 70:
            rsi_score = 5
        else:
            rsi_score = 0
        score += rsi_score
        confidence_factors.append(65)

    # 8. Extra bonus: BELOW_VALUE + BULLISH_FLOW (max 2)
    if (volume_profile and volume_profile.current_position == "BELOW_VALUE" and
            order_flow and order_flow.signal == "BULLISH"):
        score += 2
        confidence_factors.append(95)

    quantum_score = min(100, max(0, score))
    confidence = np.mean(confidence_factors) if confidence_factors else 50.0
    confidence = min(100, confidence * 0.9)

    if quantum_score >= CONFIG["QUANTUM_A_PLUS_SCORE"]:
        signal_class = "QUANTUM_A+"
    elif quantum_score >= CONFIG["QUANTUM_A_SCORE"]:
        signal_class = "QUANTUM_A"
    elif quantum_score >= CONFIG["MIN_QUANTUM_SCORE"]:
        signal_class = "QUANTUM_B"
    else:
        signal_class = "REJECT"
    return quantum_score, confidence, signal_class

# ===================== دالة حساب مستويات وقف الخسارة والأهداف (بهدفين) =====================
def compute_sl_and_tp_from_structure(entry: float, market_structure: MarketStructure, atr: float, liquidity_grab: Optional[LiquidityGrab]) -> Tuple[float, float, float]:
    if not validate_price(entry) or entry <= 0:
        return 0.0, 0.0, 0.0
    if not validate_price(atr) or atr <= 0:
        atr = entry * 0.02
    sl = 0.0
    if market_structure.order_block:
        ob_sl = market_structure.order_block['low'] * 0.995
        if 0 < ob_sl < entry:
            sl = ob_sl
    if sl == 0:
        sl = entry - (atr * CONFIG["ATR_SL_MULT"])
    if sl <= 0:
        sl = entry * 0.95
    if sl >= entry or sl <= 0:
        sl = entry * 0.95
    min_sl_distance = entry * 0.005
    if (entry - sl) < min_sl_distance:
        sl = entry - min_sl_distance
    max_sl_distance = entry * (CONFIG["MAX_SL_PCT"] / 100)
    hard_sl = entry - max_sl_distance
    if sl < hard_sl:
        logger.warning(f"[SL] SL {sl:.6f} exceeds MAX_SL_PCT, clamping to {hard_sl:.6f}")
        sl = hard_sl
    risk = entry - sl
    if risk <= 0:
        logger.error(f"[SL] Invalid risk: {risk:.6f}")
        return 0.0, 0.0, 0.0
    tp1 = entry + (risk * CONFIG["TP1_RR"])
    tp2 = entry + (risk * CONFIG["TP2_RR"])
    return sl, tp1, tp2

# ===================== ASYNC POSITION SIZING WITH DYNAMIC RISK (BY SCORE) =====================
async def calculate_position_size(entry: float, sl: float, account_size: Optional[float] = None, quantum_score: float = 0.0) -> Tuple[float, float]:
    if account_size is None:
        account_size = CONFIG["ACCOUNT_SIZE_USDT"]
    if not validate_price(entry) or not validate_price(sl) or sl >= entry:
        return 0.0, 0.0
    if quantum_score < 70:
        risk_pct = 0.6
    elif quantum_score <= 75:
        risk_pct = 0.8
    elif quantum_score <= 85:
        risk_pct = 1.1
    elif quantum_score <= 92:
        risk_pct = 1.4
    else:
        risk_pct = 1.7
    base_risk_amount = account_size * (risk_pct / 100)
    try:
        edge_engine = await get_edge_engine()
        risk_multiplier = edge_engine.risk_multiplier()
        risk_amount = base_risk_amount * risk_multiplier
        logger.debug("[EdgeEngine] Position sizing: base=%.2f, multiplier=%.2f, final=%.2f", base_risk_amount, risk_multiplier, risk_amount)
    except Exception as e:
        logger.error(f"[EdgeEngine] Error applying risk multiplier: {e}")
        risk_amount = base_risk_amount
    risk_per_unit = entry - sl
    if risk_per_unit <= 0:
        return 0.0, 0.0
    position_size = risk_amount / risk_per_unit
    position_value = position_size * entry
    min_size = CONFIG["MIN_POSITION_SIZE_USDT"]
    max_size = CONFIG["MAX_POSITION_SIZE_USDT"]
    if position_value < min_size:
        return 0.0, 0.0
    if position_value > max_size:
        position_value = max_size
        position_size = position_value / entry
    position_pct = (position_value / account_size) * 100
    return position_value, position_pct

# ===================== دالة موحدة لفحص BTC =====================
async def check_btc_conditions(exchange) -> Dict:
    if not CONFIG["ENABLE_BTC_FILTER"]:
        return {"trend": "NEUTRAL", "change_1h": 0, "safe_to_trade": True}
    async with bot.btc_trend_lock:
        if bot.btc_trend and (time.time() - bot.btc_last_check) < 300:
            return bot.btc_trend
    try:
        await rate_limiter.wait_if_needed(weight=2)
        data = await cache.get_ohlcv(exchange, "BTC/USDT", "1h", 100)
        rate_limiter.reset_errors()
        if not data or len(data) < 20:
            return {"trend": "NEUTRAL", "change_1h": 0, "safe_to_trade": True}
        df = pd.DataFrame(data, columns=['t','open','high','low','close','volume'])
        df = calculate_indicators(df)
        if df is None:
            return {"trend": "NEUTRAL", "change_1h": 0, "safe_to_trade": True}
        current_price = safe_float(df['close'].iloc[-1])
        price_1h_ago = safe_float(df['close'].iloc[-2]) if len(df) >= 2 else current_price
        price_4h_ago = safe_float(df['close'].iloc[-5]) if len(df) >= 5 else price_1h_ago
        change_1h = ((current_price - price_1h_ago) / price_1h_ago) * 100 if price_1h_ago > 0 else 0
        change_4h = ((current_price - price_4h_ago) / price_4h_ago) * 100 if price_4h_ago > 0 else 0
        rsi_1h = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
        ema50_1h = df['ema50'].iloc[-1] if 'ema50' in df.columns else current_price
        ema200_1h = df['ema200'].iloc[-1] if 'ema200' in df.columns else current_price

        if change_1h <= CONFIG["BTC_CRASH_THRESHOLD"] or change_4h <= CONFIG["BTC_CRASH_THRESHOLD"] * 1.5:
            trend = "CRASH"
            safe_to_trade = False
        elif change_1h <= CONFIG["BTC_WARNING_THRESHOLD"]:
            trend = "WARNING"
            safe_to_trade = True
        elif change_1h >= -CONFIG["BTC_WARNING_THRESHOLD"]:
            trend = "BULLISH"
            safe_to_trade = True
        else:
            trend = "NEUTRAL"
            safe_to_trade = True

        btc_filter_ok = True
        if rsi_1h < 40:
            btc_filter_ok = False
        if current_price < ema50_1h:
            btc_filter_ok = False
        if ema50_1h <= ema200_1h:
            btc_filter_ok = False

        result = {
            "trend": trend,
            "change_1h": round(change_1h, 2),
            "change_4h": round(change_4h, 2),
            "safe_to_trade": safe_to_trade,
            "price": current_price,
            "rsi": rsi_1h,
            "ema50": ema50_1h,
            "ema200": ema200_1h,
            "btc_filter_ok": btc_filter_ok
        }
        async with bot.btc_trend_lock:
            bot.btc_trend = result
            bot.btc_last_check = time.time()
        if trend == "CRASH":
            logger.warning(f"[BTC] 🚨 تحذير: انهيار! 1H: {change_1h:.2f}%, 4H: {change_4h:.2f}%")
            await send_telegram(
                f"⚠️ تحذير انهيار BTC\n\n"
                f"📉 التغير خلال ساعة: {change_1h:.2f}%\n"
                f"📉 التغير خلال 4 ساعات: {change_4h:.2f}%\n\n"
                f"🛑 تم إيقاف التداول مؤقتاً!",
                critical=True
            )
        return result
    except Exception as e:
        rate_limiter.record_error()
        logger.error(f"[BTC Check Error] {str(e)[:100]}")
        return {"trend": "NEUTRAL", "change_1h": 0, "safe_to_trade": True, "btc_filter_ok": True}

# ===================== FIXED ORDER FLOW SAMPLING LOGIC =====================
def should_run_order_flow(symbol: str, mtf_alignment: int, precheck_score: float, loop_count: int) -> bool:
    if not CONFIG.get("ORDER_FLOW_ENABLED", True):
        return False
    if mtf_alignment >= 3:
        return True
    if mtf_alignment == 2 and CONFIG.get("ORDER_FLOW_ENABLE_FOR_ALIGNMENT_2_IF_STRONG_SCORE", True):
        if precheck_score >= safe_float(CONFIG.get("ORDER_FLOW_PRECHECK_MIN_SCORE", 68.0), 68.0):
            return True
    if mtf_alignment >= 2 and CONFIG.get("ORDER_FLOW_SAMPLING_ENABLED", True):
        sample_every = CONFIG.get("ORDER_FLOW_SAMPLE_EVERY_N_LOOPS", 3)
        symbol_hash = stable_hash(symbol) % sample_every
        if symbol_hash == loop_count % sample_every:
            return True
    return False

# ===================== BTC CORRELATION FILTER =====================
async def get_btc_correlation(exchange, symbol: str) -> Optional[float]:
    if symbol == "BTC/USDT":
        return 1.0
    cache_key = f"{symbol}_BTC_corr"
    now = time.time()
    async with bot.correlation_cache_lock:
        if cache_key in bot.correlation_cache:
            corr, ts = bot.correlation_cache[cache_key]
            if now - ts < 3600:
                return corr
    try:
        btc_data = await cache.get_ohlcv(exchange, "BTC/USDT", "1h", 50)
        if not btc_data or len(btc_data) < 20:
            return None
        btc_df = pd.DataFrame(btc_data, columns=['t','open','high','low','close','volume'])
        sym_data = await cache.get_ohlcv(exchange, symbol, "1h", 50)
        if not sym_data or len(sym_data) < 20:
            return None
        sym_df = pd.DataFrame(sym_data, columns=['t','open','high','low','close','volume'])
        if len(btc_df) != len(sym_df):
            min_len = min(len(btc_df), len(sym_df))
            btc_df = btc_df.iloc[-min_len:]
            sym_df = sym_df.iloc[-min_len:]
        btc_returns = btc_df['close'].pct_change().dropna()
        sym_returns = sym_df['close'].pct_change().dropna()
        if len(btc_returns) < 10 or len(sym_returns) < 10:
            return None
        min_len = min(len(btc_returns), len(sym_returns))
        btc_returns = btc_returns.iloc[-min_len:]
        sym_returns = sym_returns.iloc[-min_len:]
        if SCIPY_AVAILABLE:
            corr, _ = stats.pearsonr(btc_returns, sym_returns)
        else:
            btc_mean = np.mean(btc_returns)
            sym_mean = np.mean(sym_returns)
            num = np.sum((btc_returns - btc_mean) * (sym_returns - sym_mean))
            den = np.sqrt(np.sum((btc_returns - btc_mean)**2) * np.sum((sym_returns - sym_mean)**2))
            corr = num / den if den != 0 else 0.0
        if np.isnan(corr):
            logger.warning(f"[BTC Correlation] NaN for {symbol}, skipping filter")
            return None
        corr = safe_float(corr, 0.0)
        async with bot.correlation_cache_lock:
            bot.correlation_cache[cache_key] = (corr, now)
        return corr
    except Exception as e:
        logger.error(f"[BTC Correlation Error] {symbol}: {str(e)[:100]}")
        return None

# ===================== INSTITUTIONAL SIGNAL GENERATOR =====================
@metrics.record_latency("signal_generation")
async def generate_quantum_signal(exchange, symbol: str) -> Optional[QuantumSignal]:
    try:
        if symbol == "BTC/USDT":
            return None
        if CONFIG.get("ENABLE_MARKET_REGIME_FILTER", True):
            try:
                data_5m = await cache.get_ohlcv(exchange, symbol, "5m", 10)
                if data_5m and len(data_5m) >= 3:
                    df_temp = pd.DataFrame(data_5m, columns=['t','open','high','low','close','volume'])
                    if len(df_temp) >= 3:
                        last_3 = df_temp['close'].iloc[-3:]
                        pct_change = last_3.pct_change().sum()
                        if abs(pct_change) > CONFIG.get("MAX_CHASE_MOVE_PCT", 0.03):
                            logger.info(f"[Chase Filter] {symbol}: large move detected ({pct_change:.2%}) - skipping")
                            return None
            except Exception:
                pass
        btc_info = await check_btc_conditions(exchange)
        if not btc_info.get('btc_filter_ok', True):
            logger.info(f"[BTC Filter] {symbol} rejected due to BTC conditions")
            return None
        corr = await get_btc_correlation(exchange, symbol)
        if corr is not None and corr < CONFIG.get("BTC_CORRELATION_THRESHOLD", 0.3):
            logger.info(f"[REJECT] {symbol} BTC correlation {corr:.2f} < {CONFIG.get('BTC_CORRELATION_THRESHOLD', 0.3)}")
            return None
        mtf = await analyze_multi_timeframe(exchange, symbol)
        if not mtf or mtf['signal'] != "ENTER":
            return None
        structure_1h = mtf['structure_1h']
        structure_15m = mtf['structure_15m']
        structure_5m = mtf['structure_5m']
        liquidity_grab = mtf['liquidity_grab']
        df_1h = mtf['df_1h']
        df_15m = mtf['df_15m']
        df_5m = mtf['df_5m']
        if CONFIG["LONG_ONLY"] and structure_1h.structure != "BULLISH":
            return None
        if 'atr' in df_15m.columns and 'close' in df_15m.columns:
            atr_15m = df_15m['atr'].iloc[-1]
            price_15m = df_15m['close'].iloc[-1]
            if price_15m > 0:
                atr_pct_15m = (atr_15m / price_15m) * 100
                if atr_pct_15m > CONFIG.get("MAX_ATR_PCT_15M", 8.0):
                    logger.info(f"[Volatility Filter] {symbol} 15m ATR% {atr_pct_15m:.2f}% > {CONFIG['MAX_ATR_PCT_15M']}% - skipping")
                    return None
        ob = structure_15m.order_block if (structure_15m and structure_15m.order_block) else None
        lg = liquidity_grab if (liquidity_grab and liquidity_grab.detected and liquidity_grab.grab_type == "BULLISH") else None
        if lg:
            last_close = df_5m['close'].iloc[-1]
            prev_high = df_5m['high'].iloc[-2]
            prev_low = df_5m['low'].iloc[-2]
            mid = (prev_high + prev_low) / 2.0
            if last_close < mid and df_5m['volume'].iloc[-1] <= df_5m['volume'].iloc[-2]:
                logger.info(f"[REJECT] {symbol} Micro Structure: last close below midpoint and volume not increasing")
                return None
        ok_accept, reason = price_acceptance_gate_5m(df_5m, ob, lg)
        if not ok_accept:
            if CONFIG.get("DEBUG_MODE"):
                logger.info(f"[REJECT] {symbol} EntryGate: {reason}")
            return None
        if ob:
            entry = _ob_entry_price(ob)
        elif lg:
            entry = safe_float(lg.grab_level) * 1.0005
        else:
            return None
        if not validate_price(entry):
            return None
        atr = safe_float(df_15m['atr'].iloc[-1])
        sl, tp1, tp2 = compute_sl_and_tp_from_structure(entry, structure_15m, atr, liquidity_grab)
        if sl == 0:
            return None
        if 'rsi' in df_5m.columns:
            rsi_5m = df_5m['rsi'].iloc[-1]
            rsi_min = CONFIG.get("RSI_MIN", 30)
            rsi_max = CONFIG.get("RSI_MAX", 75)
            if rsi_5m < rsi_min or rsi_5m > rsi_max:
                logger.info(f"[REJECT] {symbol} RSI 5m = {rsi_5m:.1f} outside {rsi_min}-{rsi_max}")
                return None
        gates_ok, gates_list = evaluate_hard_gates(
            structure_15m, None, None, liquidity_grab, mtf['alignment'], df_15m
        )
        if CONFIG["ENABLE_HARD_GATES"] and not gates_ok:
            return None
        pre_qs, pre_conf, pre_class = calculate_quantum_score(
            structure_15m, None, None, liquidity_grab, mtf['alignment'], df_15m, gates_passed=gates_list
        )
        if pre_class == "QUANTUM_A+":
            reset_daily_counters()
            max_daily_a_plus = CONFIG.get("MAX_DAILY_A_PLUS", 7)
            if STATS.get("daily_a_plus_count", 0) >= max_daily_a_plus:
                logger.info(f"[REJECT] {symbol} A+ cap reached ({STATS['daily_a_plus_count']}/{max_daily_a_plus})")
                return None

        # ===================== فلتر EMA50 > EMA200 المرن =====================
        if df_1h is not None and 'ema50' in df_1h.columns and 'ema200' in df_1h.columns:
            ema50_1h = df_1h['ema50'].iloc[-1]
            ema200_1h = df_1h['ema200'].iloc[-1]
            close_1h = df_1h['close'].iloc[-1]
            if len(df_1h) >= 3:
                ema200_slope = df_1h['ema200'].iloc[-1] > df_1h['ema200'].iloc[-2] > df_1h['ema200'].iloc[-3]
            else:
                ema200_slope = False
            if pre_class == "QUANTUM_A+":
                if ema50_1h <= ema200_1h:
                    logger.info(f"[REJECT] {symbol} A+ requires EMA50>EMA200 on 1H ({ema50_1h:.2f} <= {ema200_1h:.2f})")
                    return None
            else:
                allow = False
                if close_1h > ema200_1h and ema200_slope and structure_1h.trend_strength >= 75 and mtf['alignment'] == 3:
                    allow = True
                if not allow and ema50_1h <= ema200_1h:
                    logger.info(f"[REJECT] {symbol} EMA filter (non A+): conditions not met (close>EMA200? {close_1h>ema200_1h}, slope? {ema200_slope}, trend_strength={structure_1h.trend_strength:.1f}, alignment={mtf['alignment']})")
                    return None
        # ===================== نهاية الفلتر =====================

        order_flow = None
        if should_run_order_flow(symbol, mtf['alignment'], pre_qs, int(STATS.get("loop_count", 0))):
            if CONFIG.get("ORDER_FLOW_ENABLED", True):
                allow_low = (mtf['alignment'] == 2 and pre_qs >= 68) or (mtf['alignment'] == 1 and pre_qs >= 85)
                order_flow = await analyze_order_flow(exchange, symbol, mtf['alignment'], allow_low_alignment=allow_low)

        if lg and order_flow and order_flow.signal == "BEARISH":
            logger.info(f"[REJECT] {symbol} Liquidity Grab with Bearish Order Flow")
            return None
        if not lg and order_flow and order_flow.signal != "BULLISH":
            logger.info(f"[REJECT] {symbol} No Liquidity Grab and Order Flow not BULLISH")
            return None

        volume_profile = None
        if CONFIG.get("ENABLE_VOLUME_PROFILE", False) and CONFIG.get("_VOLUME_PROFILE_SAMPLING_OK", True):
            volume_profile = analyze_volume_profile(df_15m, precheck_score=pre_qs)

        quantum_score, confidence, signal_class = calculate_quantum_score(
            structure_15m, order_flow, volume_profile, liquidity_grab, mtf['alignment'], df_15m, gates_passed=gates_list
        )
        if quantum_score < CONFIG["MIN_QUANTUM_SCORE"]:
            return None
        position_size_usdt, position_size_pct = await calculate_position_size(entry, sl, quantum_score=quantum_score)
        if position_size_usdt == 0:
            return None

        STATS["signals_generated"] += 1
        if signal_class == "QUANTUM_A+":
            STATS["signals_a_plus"] += 1
            STATS["daily_a_plus_count"] = STATS.get("daily_a_plus_count", 0) + 1
        elif signal_class == "QUANTUM_A":
            STATS["signals_a"] += 1
        elif signal_class == "QUANTUM_B":
            STATS["signals_b"] += 1

        current_avg = STATS["avg_quantum_score"]
        total_signals = STATS["signals_generated"]
        STATS["avg_quantum_score"] = ((current_avg * (total_signals - 1)) + quantum_score) / total_signals

        risk_reward = (tp2 - entry) / (entry - sl) if entry > sl else 0
        win_probability = min(95, max(40, quantum_score * 0.8))

        return QuantumSignal(
            symbol=symbol,
            mode="SWING" if risk_reward >= 3 else "SCALP",
            entry=entry,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            atr=atr,
            position_size_usdt=position_size_usdt,
            position_size_pct=position_size_pct,
            quantum_score=quantum_score,
            confidence=confidence,
            signal_class=signal_class,
            market_structure=structure_15m,
            order_flow=order_flow,
            volume_profile=volume_profile,
            liquidity_grab=liquidity_grab,
            mtf_alignment=mtf['alignment'],
            trend_1h=structure_1h.structure,
            structure_15m=structure_15m.structure,
            entry_5m=structure_5m.structure,
            risk_reward=risk_reward,
            win_probability=win_probability,
            gates_passed=gates_list
        )
    except Exception as e:
        logger.error(f"[Signal Generator Error] {symbol}: {str(e)[:200]}")
        return None

# ===================== TELEGRAM FORMATTER (مُحدَّث لهدفين) =====================
def format_quantum_signal(signal: QuantumSignal) -> str:
    emoji_map = {
        "QUANTUM_A+": "🟢⭐",
        "QUANTUM_A": "🟢",
        "QUANTUM_B": "🟡"
    }
    emoji = emoji_map.get(signal.signal_class, "⚪")
    clean_symbol = escape_html(signal.symbol.replace('/', ''))
    tv_link = f"https://www.tradingview.com/chart/?symbol=MEXC:{clean_symbol}"
    risk = signal.entry - signal.sl
    tp1_r = (signal.tp1 - signal.entry) / risk if risk > 0 else 0
    tp2_r = (signal.tp2 - signal.entry) / risk if risk > 0 else 0
    live_badge = get_execution_mode_badge()
    account_size = CONFIG["ACCOUNT_SIZE_USDT"]
    risk_amount = signal.position_size_usdt * (risk / signal.entry)
    risk_percent = (risk_amount / account_size) * 100
    message = f"""
{emoji} {escape_html(signal.signal_class)} - {escape_html(signal.mode)} - {escape_html(signal.symbol)} ({live_badge})
🎯 الدخول: {signal.entry:.6f}
🛑 وقف الخسارة: {signal.sl:.6f}
أهداف الربح (خروج جزئي):
✅ الهدف 1 (R:{tp1_r:.1f}) - الخروج {CONFIG['TP1_EXIT_PCT']*100:.0f}%: {signal.tp1:.6f}
✅ الهدف 2 (R:{tp2_r:.1f}) - الخروج {CONFIG['TP2_EXIT_PCT']*100:.0f}%: {signal.tp2:.6f} (يتم تفعيل التريلينج بعد الهدف الأول)
💰 حجم الصفقة
━━━━━━━━━━━━━━━━
• الحجم: ${signal.position_size_usdt:.2f}
• % من الحساب: {signal.position_size_pct:.2f}%
• المخاطرة الحقيقية: {risk_percent:.2f}% (={risk_amount:.2f}$)
📊 التحليل الكمي
━━━━━━━━━━━━━━━━
• النقاط: {signal.quantum_score:.1f}/100
• الثقة: {signal.confidence:.1f}%
• المخاطرة:العائد: 1:{signal.risk_reward:.2f}
• احتمالية الربح: {signal.win_probability:.1f}%
{"✅" if signal.gates_passed else "⚠️"} البوابات الإجبارية
"""
    if signal.gates_passed:
        for gate in signal.gates_passed:
            message += f"• {escape_html(gate)}\n"
    else:
        message += "• ❌ لم يتم اجتياز البوابات الإجبارية\n"
    message += f"""
🏗️ الهيكل
• الاتجاه: {escape_html(signal.trend_1h)} (1H)
• المحاذاة: {signal.mtf_alignment}/3
• قوة الاتجاه: {signal.market_structure.trend_strength:.0f}%
"""
    if signal.market_structure.bos_bullish:
        message += f"• ✅ كسر الهيكل (مؤسسي)\n"
    if signal.market_structure.order_block:
        ob = signal.market_structure.order_block
        low = ob.get('body_low', ob['low'])
        high = ob.get('body_high', ob['high'])
        message += f"• 📦 كتلة الطلبات: {low:.6f} - {high:.6f}"
        if 'freshness' in ob:
            message += f" (حداثة: {ob['freshness']} شمعة)\n"
        else:
            message += "\n"
    if signal.order_flow:
        message += f"""
💹 تدفق الأوامر
• النمط: {escape_html(signal.order_flow.volume_profile)}
• الخلل: {signal.order_flow.imbalance:+.2f}
• الدلتا: {signal.order_flow.delta:+.2f}
• الإشارة: {escape_html(signal.order_flow.signal)}
"""
        if signal.order_flow.divergence:
            message += f"• ⚠️ اختلاف الأموال الذكية!\n"
    if signal.volume_profile:
        message += f"""
📊 توزيع الحجم
• الموقع: {escape_html(signal.volume_profile.current_position)}
• نقطة التحكم: {signal.volume_profile.poc:.6f}
• VWAP: {signal.volume_profile.vwap:.6f}
"""
    if signal.liquidity_grab:
        message += f"""
🎯 اصطياد السيولة
• النوع: {escape_html(signal.liquidity_grab.grab_type)}
• المستوى: {signal.liquidity_grab.grab_level:.6f}
• قوة الظل: {signal.liquidity_grab.wick_strength*100:.0f}%
• الحجم: {signal.liquidity_grab.volume_spike:.1f}x
• الثقة: {signal.liquidity_grab.confidence:.0f}%
"""
        if hasattr(signal.liquidity_grab, 'equal_lows') and signal.liquidity_grab.equal_lows:
            message += f"• ✅ قيعان متساوية: نعم\n"
    message += f"""
🔍 <a href=\"{tv_link}\">فتح في TradingView</a>
⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
    return message

# ===================== SYMBOL FILTER (مع استثناء BTC/USDT) =====================
async def get_filtered_symbols(exchange) -> List[str]:
    try:
        await rate_limiter.wait_if_needed(weight=CONFIG.get("TICKER_WEIGHT", 2))
        markets = await exchange.load_markets()
        tickers = None
        for attempt in range(CONFIG["MAX_RETRIES"]):
            try:
                await rate_limiter.wait_if_needed(weight=CONFIG.get("TICKER_WEIGHT", 2))
                tickers = await exchange.fetch_tickers()
                rate_limiter.reset_errors()
                break
            except Exception as e:
                rate_limiter.record_error()
                if attempt < CONFIG["MAX_RETRIES"] - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"[Tickers Error] {str(e)[:100]}")
                    return []
        if not tickers:
            return []
        filtered = []
        for symbol, ticker in tickers.items():
            if not symbol.endswith(CONFIG["QUOTE"]):
                continue
            if symbol == "BTC/USDT":
                continue
            if symbol in markets:
                market = markets[symbol]
                if market.get('type') != 'spot':
                    continue
                if not market.get('active', True):
                    continue
                volume = safe_float(ticker.get('quoteVolume'), 0)
                price = safe_float(ticker.get('last'), 0)
                if volume < CONFIG["MIN_VOLUME_24H"]:
                    continue
                if price < 0.00001:
                    continue
                score = min(volume / 1_000_000, 50)
                filtered.append((symbol, score))
        filtered.sort(key=lambda x: x[1], reverse=True)
        max_scan = CONFIG.get("MAX_SYMBOL_SCAN", 80)
        return [s for s, _ in filtered[:max_scan]]
    except Exception as e:
        logger.error(f"[Symbol Filter Error] {str(e)}")
        return []

# ===================== ENHANCED EMERGENCY HELPER =====================
async def mark_trade_emergency(symbol: str, reason: str, critical_msg: str = ""):
    try:
        if not await bot.get_trade_lock(symbol):
            logger.error(f"[Emergency] Failed to acquire lock for {symbol}")
            return
        try:
            if symbol in ACTIVE_TRADES:
                ACTIVE_TRADES[symbol].emergency_state = True
                ACTIVE_TRADES[symbol].emergency_reason = reason
                ACTIVE_TRADES[symbol].emergency_last_attempt = now_utc_iso()
                ACTIVE_TRADES[symbol].emergency_attempts += 1
                trade_snapshot = copy.deepcopy(ACTIVE_TRADES[symbol])
                await asyncio.to_thread(db_manager.save_trade, trade_snapshot)
        finally:
            bot.release_trade_lock(symbol)
    except asyncio.TimeoutError:
        logger.error(f"[Emergency] Timeout for {symbol}")
        return
    STATS["live_emergencies"] += 1
    if critical_msg:
        await send_telegram(critical_msg, critical=True)
    else:
        await send_telegram(
            f"🚨 EMERGENCY STATE ACTIVATED\n\n"
            f"الرمز: {escape_html(symbol)}\n"
            f"السبب: {escape_html(reason)}\n\n"
            f"⚠️ سيتم إعادة المحاولة تلقائيًا كل 10 دقائق.",
            critical=True
        )

# ===================== CONSECUTIVE LOSS GUARD =====================
def record_loss(symbol: str):
    now = time.time()
    bot.consecutive_losses[symbol].append(now)
    cutoff = now - 86400
    bot.consecutive_losses[symbol] = [t for t in bot.consecutive_losses[symbol] if t > cutoff]
    if len(bot.consecutive_losses[symbol]) >= 2:
        bot.consecutive_loss_blacklist[symbol] = now + 21600
        logger.warning(f"[Loss Guard] {symbol} has 2+ losses in 24h -> blacklisted for 6h")
        asyncio.create_task(send_telegram(
            f"🛑 {escape_html(symbol)} محظور لمدة 6 ساعات بسبب خسارتين متتاليتين خلال 24 ساعة.",
            critical=False
        ))

def is_symbol_blacklisted_loss(symbol: str) -> bool:
    expiry = bot.consecutive_loss_blacklist.get(symbol, 0)
    if expiry > time.time():
        return True
    if expiry > 0:
        del bot.consecutive_loss_blacklist[symbol]
    return False

# ===================== INSTITUTIONAL LIVE ENTRY =====================
@metrics.record_latency("live_entry")
async def execute_live_entry_if_enabled(exchange, signal: QuantumSignal) -> Tuple[bool, Optional[TradeState]]:
    if not is_live_trading_enabled():
        return True, None
    if STATS.get("global_consecutive_losses", 0) >= CONFIG["MAX_CONSECUTIVE_LOSSES"]:
        await send_telegram(
            f"🛑 Global consecutive loss circuit breaker reached ({CONFIG['MAX_CONSECUTIVE_LOSSES']} losses). Stopping new entries.",
            critical=True
        )
        return False, None
    if is_daily_loss_blocked():
        await send_telegram(
            f"🛑 Daily Loss Circuit Breaker\n"
            f"تم إيقاف فتح صفقات جديدة اليوم.\n"
            f"• Realized R اليوم: {daily_circuit.get_state().get('realized_r', 0.0):.2f}R\n"
            f"• الحد: {CONFIG.get('DAILY_MAX_LOSS_R')}",
            critical=True
        )
        return False, None
    max_trades = int(CONFIG.get("LIVE_MAX_OPEN_TRADES", 5))
    acquired_symbols = await bot.lock_manager.acquire_all_locks()
    if not acquired_symbols:
        return False, None
    try:
        if len(ACTIVE_TRADES) >= max_trades:
            logger.warning("[LIVE] Max open trades reached; skipping entry.")
            return False, None
    finally:
        bot.lock_manager.release_all_locks(acquired_symbols)
    try:
        data_5m = await cache.get_ohlcv(exchange, signal.symbol, CONFIG["TF_ENTRY"], 150)
        if data_5m:
            df_5m = calculate_indicators(pd.DataFrame(data_5m, columns=['t','open','high','low','close','volume']))
            okq, reason = entry_quality_filter_5m(df_5m, signal)
            if not okq:
                await send_telegram(
                    f"⚠️ تم رفض تنفيذ الصفقة (Entry Quality Filter)\n"
                    f"• الرمز: {escape_html(signal.symbol)}\n"
                    f"• السبب: {escape_html(reason)}"
                )
                return False, None
    except Exception:
        pass
    if CONFIG.get("LIVE_REQUIRE_SPREAD_FILTER", True):
        sp = await get_spread_pct(exchange, signal.symbol)
        if sp is not None and sp > CONFIG["MAX_SPREAD_PCT"]:
            logger.warning(f"[LIVE] Spread too high {signal.symbol}: {sp:.3f}% > {CONFIG['MAX_SPREAD_PCT']}%")
            await send_telegram(
                f"⚠️ تم رفض تنفيذ الصفقة بسبب السبريد\n"
                f"• الرمز: {escape_html(signal.symbol)}\n"
                f"• السبريد: {sp:.3f}%\n"
                f"• الحد الأقصى: {CONFIG['MAX_SPREAD_PCT']}%"
            )
            return False, None
    ref_price = signal.entry
    amount_base, err = await compute_order_amount_base(exchange, signal.symbol, signal.position_size_usdt, ref_price)
    if amount_base <= 0:
        await send_telegram(
            f"❌ فشل حساب كمية الدخول (Limits/Notional)\n"
            f"• الرمز: {escape_html(signal.symbol)}\n"
            f"• السبب: {escape_html(err)}"
        )
        return False, None
    entry_order = await place_limit_buy_entry(exchange, signal.symbol, signal.entry, amount_base)
    if not entry_order or not entry_order.get("id"):
        await send_telegram(
            f"❌ فشل إرسال أمر LIMIT BUY\n"
            f"• الرمز: {escape_html(signal.symbol)}",
            critical=True
        )
        return False, None
    order_id = str(entry_order.get("id"))
    await send_telegram(
        f"🧾 تم إرسال أمر دخول (LIMIT BUY)\n"
        f"• الرمز: {escape_html(signal.symbol)}\n"
        f"• السعر: {signal.entry:.6f}\n"
        f"• الكمية (BASE): {amount_base}\n"
        f"• Order ID: {escape_html(order_id)}"
    )
    filled, final_order = await wait_for_order_fill_or_cancel(
        exchange, signal.symbol, order_id, int(CONFIG.get("ENTRY_LIMIT_TIMEOUT_SEC", 120))
    )
    if not filled:
        await send_telegram(
            f"⏳ لم يتم تنفيذ أمر الدخول في الوقت المحدد وتم إلغاؤه\n"
            f"• الرمز: {escape_html(signal.symbol)}\n"
            f"• Order ID: {escape_html(order_id)}"
        )
        return False, None
    fill_price = safe_float(final_order.get("average"), 0.0) or safe_float(final_order.get("price"), signal.entry)
    fill_amount = safe_float(final_order.get("filled"), 0.0) or amount_base
    slippage_pct = abs(fill_price - signal.entry) / signal.entry
    if slippage_pct > CONFIG["MAX_SLIPPAGE_PCT"]:
        logger.warning(f"[Slippage] {signal.symbol} slippage {slippage_pct:.4%} > {CONFIG['MAX_SLIPPAGE_PCT']:.2%}, closing immediately")
        await send_telegram(
            f"⚠️ تم إغلاق الصفقة بسبب انزلاق سعري كبير\n"
            f"• الرمز: {escape_html(signal.symbol)}\n"
            f"• الانزلاق: {slippage_pct:.4%}\n"
            f"• الحد الأقصى: {CONFIG['MAX_SLIPPAGE_PCT']:.2%}"
        )
        await market_sell_safe(exchange, signal.symbol, fill_amount, max_retries=2)
        return False, None
    adj_sl, adj_tp1, adj_tp2, adj_position_size = recalibrate_levels_on_fill(fill_price, signal)
    actual_risk = fill_price - adj_sl
    if actual_risk > 0:
        actual_rr = (adj_tp1 - fill_price) / actual_risk
        min_allowed_rr = CONFIG["TP1_RR"] * 0.75
        if actual_rr < min_allowed_rr:
            logger.warning(f"[Slippage] {signal.symbol} actual RR {actual_rr:.2f} < {min_allowed_rr:.2f}, closing immediately")
            await send_telegram(
                f"⚠️ تم إغلاق الصفقة بسبب انخفاض RR الفعلي بعد التنفيذ\n"
                f"• الرمز: {escape_html(signal.symbol)}\n"
                f"• RR الفعلي: {actual_rr:.2f}\n"
                f"• الحد الأدنى: {min_allowed_rr:.2f}"
            )
            await market_sell_safe(exchange, signal.symbol, fill_amount, max_retries=2)
            return False, None
    new_position_usdt, new_position_pct = await calculate_position_size(
        fill_price, adj_sl, quantum_score=signal.quantum_score
    )
    if new_position_usdt < CONFIG["MIN_POSITION_SIZE_USDT"] * 0.5:
        logger.warning(f"[Slippage] {signal.symbol} new position size {new_position_usdt:.2f} too small after recalc, closing")
        await send_telegram(
            f"⚠️ تم إغلاق الصفقة لأن حجم المركز الجديد صغير جداً بعد إعادة الحساب\n"
            f"• الرمز: {escape_html(signal.symbol)}\n"
            f"• الحجم الجديد: ${new_position_usdt:.2f}"
        )
        await market_sell_safe(exchange, signal.symbol, fill_amount, max_retries=2)
        return False, None
    adjusted_position_usdt = new_position_usdt
    adjusted_amount_base = adjusted_position_usdt / fill_price
    if adjusted_amount_base > fill_amount:
        adjusted_amount_base = fill_amount
        adjusted_position_usdt = adjusted_amount_base * fill_price
    intended_position_base = signal.position_size_usdt / signal.entry
    intended_risk_amount = intended_position_base * (signal.entry - signal.sl)
    actual_risk_amount = fill_amount * (fill_price - adj_sl)
    if actual_risk_amount > intended_risk_amount * 1.2:
        logger.warning(f"[Slippage] {signal.symbol} actual risk {actual_risk_amount:.2f} > intended {intended_risk_amount:.2f} by >20%, closing immediately")
        await send_telegram(
            f"⚠️ تم إغلاق الصفقة بسبب انحراف المخاطرة بعد التنفيذ\n"
            f"• الرمز: {escape_html(signal.symbol)}\n"
            f"• المخاطرة الفعلية: {actual_risk_amount:.2f}\n"
            f"• المخاطرة المستهدفة: {intended_risk_amount:.2f}\n"
            f"• نسبة الانحراف: {actual_risk_amount/intended_risk_amount:.1f}x"
        )
        await market_sell_safe(exchange, signal.symbol, fill_amount, max_retries=2)
        return False, None
    trade_state = TradeState(
        symbol=signal.symbol,
        entry=fill_price if validate_price(fill_price) else signal.entry,
        original_sl=adj_sl,
        current_sl=adj_sl,
        tp1=adj_tp1,
        tp2=adj_tp2,
        atr=signal.atr,
        signal_class=signal.signal_class,
        quantum_score=signal.quantum_score,
        gates_passed=signal.gates_passed,
        entry_order_id=order_id,
        entry_filled=True,
        entry_fill_price=fill_price,
        entry_fill_amount=adjusted_amount_base,
        is_paper=False,
        execution_mode="LIVE",
        entry_assumed=False,
        order_block_low=signal.market_structure.order_block.get('body_low', signal.market_structure.order_block.get('low', 0.0)) if signal.market_structure.order_block else 0.0,
        order_block_high=signal.market_structure.order_block.get('body_high', signal.market_structure.order_block.get('high', 0.0)) if signal.market_structure.order_block else 0.0,
        liquidity_grab_level=signal.liquidity_grab.grab_level if signal.liquidity_grab else 0.0,
    )
    if CONFIG.get("LIVE_PLACE_SL_ORDER"):
        sl_order = await place_stop_loss_order(exchange, signal.symbol, adj_sl, adjusted_amount_base)
        if sl_order:
            trade_state.sl_order_id = str(sl_order.get("id"))
            logger.info(f"[LIVE] SL order placed for {signal.symbol} at {adj_sl:.6f}, ID: {trade_state.sl_order_id}")
        else:
            await send_telegram(
                f"⚠️ فشل وضع أمر Stop Loss على المنصة للرمز {escape_html(signal.symbol)}\n"
                f"• سيتم الاعتماد على المراقبة الداخلية فقط.",
                critical=False
            )
    await send_telegram(
        f"✅ تم تنفيذ الدخول فعليًا (FILLED)\n"
        f"• الرمز: {escape_html(signal.symbol)}\n"
        f"• متوسط التنفيذ: {trade_state.entry:.6f}\n"
        f"• SL (recalibrated): {trade_state.current_sl:.6f}\n"
        f"• TP1: {trade_state.tp1:.6f}\n"
        f"• TP2: {trade_state.tp2:.6f}\n"
        f"• الكمية (BASE): {trade_state.entry_fill_amount}"
    )
    return True, trade_state

# ===================== ENHANCED BATCH PROCESSOR =====================
async def process_symbol_batch(exchange, symbols: List[str]) -> int:
    signals_found = 0
    for symbol in symbols:
        try:
            if is_symbol_blacklisted_loss(symbol):
                logger.warning(f"[Loss Guard] Skipping blacklisted symbol due to consecutive losses: {symbol}")
                continue
            if bot.lock_manager.is_blacklisted(symbol):
                logger.warning(f"[Blacklist] Skipping blacklisted symbol: {symbol}")
                continue
            if not await bot.get_trade_lock(symbol):
                continue
            try:
                if symbol in ACTIVE_TRADES:
                    continue
            finally:
                bot.release_trade_lock(symbol)
            if is_in_cooldown(symbol):
                continue
            if is_paper_trading_enabled():
                acquired_symbols = await bot.lock_manager.acquire_all_locks()
                if not acquired_symbols:
                    continue
                try:
                    if len(ACTIVE_TRADES) >= int(CONFIG.get("PAPER_MAX_OPEN_TRADES", 25)):
                        continue
                finally:
                    bot.lock_manager.release_all_locks(acquired_symbols)
            signal = await generate_quantum_signal(exchange, symbol)
            if signal:
                signals_found += 1
                if not await bot.get_trade_lock(symbol):
                    continue
                try:
                    if symbol in ACTIVE_TRADES:
                        continue
                finally:
                    bot.release_trade_lock(symbol)
                signal_dict = asdict(signal)
                signal_dict["execution_mode"] = "LIVE" if is_live_trading_enabled() else ("PAPER" if is_paper_trading_enabled() else "SIGNAL")
                message = format_quantum_signal(signal)
                await send_telegram(message)
                if is_live_trading_enabled():
                    ok, live_trade_state = await execute_live_entry_if_enabled(exchange, signal)
                    if not ok or not live_trade_state:
                        set_symbol_cooldown(symbol)
                        await asyncio.sleep(0.02)
                        continue
                    set_symbol_cooldown(symbol)
                    if not await bot.get_trade_lock(symbol):
                        continue
                    try:
                        ACTIVE_TRADES[symbol] = live_trade_state
                    finally:
                        bot.release_trade_lock(symbol)
                    await asyncio.to_thread(db_manager.save_trade, live_trade_state, signal_dict)
                    logger.info(f"[LIVE] ✅ Entry filled - {signal.signal_class} - {symbol} - Score: {signal.quantum_score:.1f}")
                    await asyncio.sleep(0.02)
                    continue
                if is_paper_trading_enabled():
                    entry_fill_amount = signal.position_size_usdt / signal.entry if signal.entry > 0 else 1.0
                    trade_state = TradeState(
                        symbol=symbol,
                        entry=signal.entry,
                        original_sl=signal.sl,
                        current_sl=signal.sl,
                        tp1=signal.tp1,
                        tp2=signal.tp2,
                        atr=signal.atr,
                        signal_class=signal.signal_class,
                        quantum_score=signal.quantum_score,
                        gates_passed=signal.gates_passed,
                        is_paper=True,
                        execution_mode="PAPER",
                        entry_assumed=True,
                        entry_filled=True,
                        entry_fill_amount=entry_fill_amount,
                        order_block_low=signal.market_structure.order_block.get('body_low', signal.market_structure.order_block.get('low', 0.0)) if signal.market_structure.order_block else 0.0,
                        order_block_high=signal.market_structure.order_block.get('body_high', signal.market_structure.order_block.get('high', 0.0)) if signal.market_structure.order_block else 0.0,
                        liquidity_grab_level=signal.liquidity_grab.grab_level if signal.liquidity_grab else 0.0,
                    )
                    if not await bot.get_trade_lock(symbol):
                        continue
                    try:
                        ACTIVE_TRADES[symbol] = trade_state
                    finally:
                        bot.release_trade_lock(symbol)
                    set_symbol_cooldown(symbol)
                    signal_dict["is_paper"] = True
                    signal_dict["entry_assumed"] = True
                    await asyncio.to_thread(db_manager.save_trade, trade_state, signal_dict)
                    STATS["paper_trades_opened"] += 1
                    logger.info(f"[PAPER] ✅ {signal.signal_class} - {symbol} - Score: {signal.quantum_score:.1f} - Gates: {len(signal.gates_passed)}")
                    await asyncio.sleep(0.02)
                    continue
                set_symbol_cooldown(symbol)
                await asyncio.sleep(0.02)
                continue
            await asyncio.sleep(0.02)
        except Exception as e:
            logger.error(f"[Batch Error] {symbol}: {str(e)[:100]}")
            continue
    return signals_found

# ===================== TRADE MONITOR =====================
async def monitor_active_trades(exchange):
    if not ACTIVE_TRADES:
        return
    try:
        await asyncio.wait_for(_monitor_active_trades_internal(exchange), timeout=60)
    except asyncio.TimeoutError:
        logger.error("[Monitor] Timeout exceeded - skipping this cycle")
    except Exception as e:
        logger.error(f"[Monitor Main Error] {str(e)}")

async def db_update_trade_with_version_async(symbol: str, updates: Dict[str, Any], expected_version: int) -> bool:
    try:
        return await asyncio.to_thread(db_manager.update_trade_with_version, symbol, updates, expected_version)
    except Exception as e:
        logger.error(f"[DB Async Update Error] {symbol}: {e}")
        return False

async def _monitor_active_trades_internal(exchange):
    symbols = list(ACTIVE_TRADES.keys())
    ticker_tasks = []
    active_symbols = []
    for symbol in symbols:
        if not bot.lock_manager.is_blacklisted(symbol):
            ticker_tasks.append(fetch_ticker_safe(exchange, symbol))
            active_symbols.append(symbol)
    ticker_results = await asyncio.gather(*ticker_tasks, return_exceptions=True)
    for symbol, ticker_result in zip(active_symbols, ticker_results):
        if isinstance(ticker_result, Exception):
            logger.error(f"[Monitor] Error fetching ticker for {symbol}: {ticker_result}")
            continue
        ticker = ticker_result
        if ticker is None:
            continue
        current_price = safe_float(ticker.get('last'))
        if not validate_price(current_price):
            continue
        if not await bot.get_trade_lock(symbol):
            continue
        try:
            if symbol not in ACTIVE_TRADES:
                continue
            trade = ACTIVE_TRADES[symbol]
            entry = trade.entry
            original_sl = trade.original_sl
            current_sl = trade.current_sl
            tp1 = trade.tp1
            tp2 = trade.tp2
            tp1_hit = trade.tp1_hit
            tp2_hit = trade.tp2_hit
            quantum_score = trade.quantum_score
            be_moved = trade.be_moved
            trailing_active = trade.trailing_active
            atr = trade.atr
            order_block_low = trade.order_block_low
            liquidity_grab_level = trade.liquidity_grab_level
            version = trade._version
            is_exiting = trade.is_exiting
            entry_fill_amount = trade.entry_fill_amount
            remaining_position = trade.remaining_position
            is_paper = trade.is_paper
        finally:
            bot.release_trade_lock(symbol)
        if is_exiting:
            continue
        risk = entry - original_sl
        r_multiple = (current_price - entry) / risk if risk > 0 else 0
        await apply_be_and_trail(
            symbol, entry, current_price, r_multiple, quantum_score,
            be_moved, trailing_active, atr, order_block_low, liquidity_grab_level, version
        )
        if is_live_trading_enabled() and exchange is not None:
            if current_price <= current_sl:
                await close_trade_full(symbol, current_price, "SL", exchange=exchange)
                continue
            if current_price >= tp1 and not tp1_hit:
                await partial_exit(symbol, trade, current_price, "TP1", CONFIG["TP1_EXIT_PCT"], exchange=exchange)
            if current_price >= tp2 and not tp2_hit:
                await partial_exit(symbol, trade, current_price, "TP2", CONFIG["TP2_EXIT_PCT"], exchange=exchange)
        else:
            if current_price <= current_sl:
                await close_trade_full(symbol, current_price, "SL")
                continue
            if current_price >= tp1 and not tp1_hit:
                await partial_exit(symbol, trade, current_price, "TP1", CONFIG["TP1_EXIT_PCT"])
            if current_price >= tp2 and not tp2_hit:
                await partial_exit(symbol, trade, current_price, "TP2", CONFIG["TP2_EXIT_PCT"])

async def apply_be_and_trail(symbol, entry, current_price, r_multiple, quantum_score,
                             be_moved, trailing_active, atr, order_block_low, liquidity_grab_level, version):
    """دالة مشتركة لتطبيق وقف الخسارة المتحرك (Break-even و Trailing)"""
    # Break-Even
    if quantum_score < 75:
        be_at_r = 1.3
    elif quantum_score <= 85:
        be_at_r = 1.1
    else:
        be_at_r = 1.0
    if r_multiple >= be_at_r and not be_moved:
        if atr > 0:
            new_sl = entry + (atr * CONFIG["BE_ATR_MULT"])
        else:
            new_sl = entry + (0.001 * entry)
        updates = {
            'current_sl': new_sl,
            'be_moved': True,
            '_version': version + 1
        }
        if await db_update_trade_with_version_async(symbol, updates, version):
            try:
                async with trade_lock(bot, symbol):
                    if symbol in ACTIVE_TRADES:
                        ACTIVE_TRADES[symbol].current_sl = new_sl
                        ACTIVE_TRADES[symbol].be_moved = True
                        ACTIVE_TRADES[symbol]._version += 1
            except RuntimeError:
                logger.warning(f"[BE] Could not reacquire lock for {symbol} to update local state")
            logger.info(f"[BE] {symbol} - internal SL moved to breakeven using ATR")
    # Trailing
    elif r_multiple >= CONFIG["TRAIL_START_R"] and be_moved:
        if atr > 0:
            atr_estimate = atr
        else:
            atr_estimate = entry * 0.02
        new_sl = current_price - (atr_estimate * CONFIG["TRAIL_ATR_MULT"])
        if new_sl > 0:
            buffer_pct = CONFIG.get("TRAILING_BUFFER_PCT", 0.001)
            if order_block_low > 0 and new_sl > order_block_low:
                new_sl = min(new_sl, order_block_low * (1 - buffer_pct))
            if liquidity_grab_level > 0 and new_sl > liquidity_grab_level:
                new_sl = min(new_sl, liquidity_grab_level * (1 - buffer_pct))
            if new_sl > 0:
                updates = {
                    'current_sl': new_sl,
                    'trailing_active': True,
                    '_version': version + 1
                }
                if await db_update_trade_with_version_async(symbol, updates, version):
                    try:
                        async with trade_lock(bot, symbol):
                            if symbol in ACTIVE_TRADES:
                                ACTIVE_TRADES[symbol].current_sl = new_sl
                                ACTIVE_TRADES[symbol].trailing_active = True
                                ACTIVE_TRADES[symbol]._version += 1
                    except RuntimeError:
                        logger.warning(f"[Trail] Could not reacquire lock for {symbol} to update local state")

async def fetch_ticker_safe(exchange, symbol):
    try:
        await rate_limiter.wait_if_needed(weight=CONFIG.get("TICKER_WEIGHT", 2))
        ticker = await exchange.fetch_ticker(symbol)
        rate_limiter.reset_errors()
        return ticker
    except Exception as e:
        rate_limiter.record_error()
        logger.error(f"[Ticker Error] {symbol}: {e}")
        return None

# ===================== ENHANCED PARTIAL EXIT =====================
async def partial_exit(symbol: str, trade: TradeState, exit_price: float, tp_level: str, exit_pct: float, exchange=None):
    # نستخدم try/finally لضمان تحرير القفل
    lock_acquired = await bot.get_trade_lock(symbol)
    if not lock_acquired:
        logger.error(f"[partial_exit] Failed to acquire lock for {symbol}")
        return
    try:
        if symbol not in ACTIVE_TRADES:
            return
        current_trade = ACTIVE_TRADES[symbol]
        if current_trade.is_exiting:
            logger.warning(f"[partial_exit] Exit already in progress for {symbol}, skipping")
            return
        current_trade.is_exiting = True
        if tp_level == "TP1" and current_trade.tp1_hit:
            return
        elif tp_level == "TP2" and current_trade.tp2_hit:
            return
        if tp_level == "TP1":
            current_trade.tp1_order_done = True
        elif tp_level == "TP2":
            current_trade.tp2_order_done = True
        entry_fill_amount = current_trade.entry_fill_amount
        current_version = current_trade._version
        sl_order_id = current_trade.sl_order_id
        remaining_position_before = current_trade.remaining_position
        await asyncio.to_thread(db_manager.save_trade, current_trade)
        sell_amount_base = entry_fill_amount * exit_pct
        live_sell_ok = True
        fill_price = exit_price
        # نحرر القفل مؤقتًا للقيام بعمليات API
        bot.release_trade_lock(symbol)
        lock_acquired = False
        if is_live_trading_enabled() and exchange and sell_amount_base > 0:
            sell_order = await market_sell_safe(exchange, symbol, sell_amount_base, max_retries=2)
            live_sell_ok = bool(sell_order)
            if live_sell_ok and sell_order:
                fill_price = safe_float(sell_order.get('average')) or safe_float(sell_order.get('price')) or exit_price
                if not validate_price(fill_price):
                    fill_price = exit_price
        # نعيد الحصول على القفل لتحديث الحالة
        lock_acquired = await bot.get_trade_lock(symbol)
        if not lock_acquired:
            logger.error(f"[partial_exit] Failed to reacquire lock for {symbol} after API")
            return
        if symbol not in ACTIVE_TRADES:
            return
        current_trade = ACTIVE_TRADES[symbol]
        if not live_sell_ok:
            if tp_level == "TP1":
                current_trade.tp1_order_done = False
            elif tp_level == "TP2":
                current_trade.tp2_order_done = False
            current_trade.is_exiting = False
            await asyncio.to_thread(db_manager.save_trade, current_trade)
            asyncio.create_task(mark_trade_emergency(symbol, f"partial_sell_failed({tp_level})"))
            return
        if current_trade._version != current_version:
            logger.warning(f"[partial_exit] Version mismatch for {symbol}. Expected {current_version}, got {current_trade._version}. Proceeding.")
        risk = current_trade.entry - current_trade.original_sl
        actual_r_multiple = (fill_price - current_trade.entry) / risk if risk > 0 else 0
        exit_r = actual_r_multiple * exit_pct
        current_trade.remaining_position -= exit_pct
        current_trade.total_realized_r += exit_r
        if tp_level == "TP1":
            current_trade.tp1_hit = True
            logger.info(f"[TP1] {symbol} hit at {fill_price}")
            STATS["tp1_hits"] += 1
        elif tp_level == "TP2":
            current_trade.tp2_hit = True
            logger.info(f"[TP2] {symbol} hit at {fill_price}")
            STATS["tp2_hits"] += 1
        current_trade._version += 1
        current_trade.is_exiting = False
        if sl_order_id and exchange:
            remaining_base = entry_fill_amount * current_trade.remaining_position
            if remaining_base > 0:
                await cancel_stop_loss_order(exchange, symbol, sl_order_id)
                new_sl_order = await place_stop_loss_order(exchange, symbol, current_trade.current_sl, remaining_base)
                if new_sl_order:
                    current_trade.sl_order_id = str(new_sl_order.get("id"))
        await asyncio.to_thread(db_manager.save_trade, current_trade)
        STATS["trades_partial"] += 1
        STATS["total_r_multiple"] += exit_r
        record_daily_r(exit_r)
        profit_pct = ((fill_price - current_trade.entry) / current_trade.entry) * 100
        mode_badge = "✅ LIVE" if is_live_trading_enabled() else ("🟨 PAPER" if getattr(current_trade, "is_paper", False) else "🟦 SIGNAL")
        message = f"""
✅ {escape_html(tp_level)} Hit - {escape_html(symbol)} ({mode_badge})
• الدخول: {current_trade.entry:.6f}
• الخروج (سعر): {fill_price:.6f}
• الربح: {profit_pct:+.2f}%
• مضاعف R: {actual_r_multiple:.2f}R
• نسبة الخروج: {exit_pct*100:.0f}%
• R المحقق: {exit_r:.2f}R
• المتبقي: {(current_trade.remaining_position*100):.0f}%
{"• LIVE Sell: ✅" if is_live_trading_enabled() and live_sell_ok else ("• LIVE Sell: ❌" if is_live_trading_enabled() else "")}
"""
        await send_telegram(message)
        logger.info(f"[Partial Exit] {symbol} - {tp_level} - {profit_pct:+.2f}% - {exit_pct*100:.0f}%")
        if current_trade.remaining_position <= 0.01:
            await close_trade_full(symbol, fill_price, "ALL_TPS", exchange=exchange)
    except Exception as e:
        logger.error(f"[partial_exit] Unhandled error for {symbol}: {e}")
    finally:
        if lock_acquired:
            bot.release_trade_lock(symbol)

# ===================== INSTITUTIONAL CLOSE TRADE FULL =====================
async def close_trade_full(symbol: str, exit_price: float, exit_type: str, exchange=None, sell_order_info: Optional[Dict] = None):
    # نحصل على نسخة من بيانات الصفقة داخل القفل ثم نحرره
    trade_snapshot = None
    remaining = 0
    fill_amount = 0
    entry = 0
    original_sl = 0
    total_realized_r = 0
    quantum_score = 0
    signal_class = ""
    gates_passed = []
    tp1_hit = False
    tp2_hit = False
    execution_mode = ""
    version = 0
    sl_order_id = ""
    try:
        lock_acquired = await bot.get_trade_lock(symbol)
        if not lock_acquired:
            logger.error(f"[close_trade_full] Failed to acquire lock for {symbol}")
            return
        try:
            if symbol not in ACTIVE_TRADES:
                return
            trade = ACTIVE_TRADES[symbol]
            # نأخذ نسخة عميقة لاستخدامها خارج القفل
            trade_snapshot = copy.deepcopy(trade)
            remaining = trade.remaining_position
            fill_amount = trade.entry_fill_amount
            entry = trade.entry
            original_sl = trade.original_sl
            total_realized_r = trade.total_realized_r
            quantum_score = trade.quantum_score
            signal_class = trade.signal_class
            gates_passed = trade.gates_passed
            tp1_hit = trade.tp1_hit
            tp2_hit = trade.tp2_hit
            execution_mode = trade.execution_mode
            version = trade._version
            sl_order_id = trade.sl_order_id
        finally:
            bot.release_trade_lock(symbol)
    except Exception as e:
        logger.error(f"[close_trade_full] Error acquiring trade data: {e}")
        return

    if trade_snapshot is None:
        return

    risk = entry - original_sl
    r_multiple = (exit_price - entry) / risk if risk > 0 else 0
    live_sell_ok = True
    fill_price = exit_price

    if is_live_trading_enabled() and exchange is not None and remaining > 0.01 and exit_type in ("SL", "ALL_TPS", "EMERGENCY_RECOVERY"):
        remaining_base = fill_amount * remaining
        sell_order = await market_sell_safe(exchange, symbol, remaining_base, max_retries=3)
        live_sell_ok = bool(sell_order)
        if live_sell_ok and sell_order:
            fill_price = safe_float(sell_order.get('average')) or safe_float(sell_order.get('price')) or exit_price
        if sl_order_id and exchange and exit_type in ("SL", "ALL_TPS", "EMERGENCY_RECOVERY"):
            await cancel_stop_loss_order(exchange, symbol, sl_order_id)

    final_exit_r = 0.0
    if remaining > 0:
        final_exit_r = r_multiple * remaining
    total_r = total_realized_r + final_exit_r

    try:
        edge_engine = await get_edge_engine()
        await edge_engine.record_trade(
            r_multiple=total_r,
            quantum_score=quantum_score,
            exit_type=exit_type
        )
        logger.debug("[EdgeEngine] Recorded trade for %s: R=%.2f", symbol, total_r)
    except Exception as e:
        logger.error("[EdgeEngine] Failed to record trade: %s", str(e))

    profit_pct = ((fill_price - entry) / entry) * 100
    if r_multiple <= 0:
        STATS["global_consecutive_losses"] = STATS.get("global_consecutive_losses", 0) + 1
        STATS["trades_lost"] += 1
        record_loss(symbol)
    else:
        STATS["global_consecutive_losses"] = 0
        STATS["trades_won"] += 1
        STATS["total_r_multiple"] += final_exit_r
        record_daily_r(final_exit_r)

    await asyncio.to_thread(db_manager.record_trade_history, symbol, trade_snapshot, fill_price, exit_type, total_r, execution_mode)

    mode_badge = "✅ LIVE" if is_live_trading_enabled() else ("🟨 PAPER" if getattr(trade_snapshot, "is_paper", False) else "🟦 SIGNAL")
    message = f"""
{escape_html(exit_type)} - تم إغلاق الصفقة - {escape_html(symbol)} ({mode_badge})
• الدخول: {entry:.6f}
• الخروج (سعر): {fill_price:.6f}
• الربح: {profit_pct:+.2f}%
• إجمالي R: {total_r:.2f}R
{"• LIVE Sell: ✅" if is_live_trading_enabled() and live_sell_ok else ("• LIVE Sell: ❌" if is_live_trading_enabled() else "")}
بيانات الإشارة:
• التصنيف: {escape_html(signal_class) if signal_class else "N/A"}
• النقاط: {quantum_score:.1f}
الأهداف المحققة:
• الهدف 1: {'✅' if tp1_hit else '❌'}
• الهدف 2: {'✅' if tp2_hit else '❌'}
"""
    await send_telegram(message)
    set_symbol_cooldown(symbol)

    # محاولة إزالة الصفقة من ACTIVE_TRADES
    lock_acquired = await bot.get_trade_lock(symbol)
    if lock_acquired:
        try:
            if symbol in ACTIVE_TRADES:
                ACTIVE_TRADES[symbol].sl_order_done = True
                ACTIVE_TRADES.pop(symbol, None)
        finally:
            bot.release_trade_lock(symbol)
    else:
        # إذا فشل الحصول على القفل، نضيف الصفقة إلى قائمة انتظار للحذف (يمكن تنفيذها في الدورة القادمة)
        logger.warning(f"[close_trade_full] Could not remove {symbol} from ACTIVE_TRADES due to lock failure")
        # يمكن إعادة المحاولة في الدورة التالية عبر monitor_active_trades

    await asyncio.to_thread(db_manager.remove_trade, symbol)
    logger.info(f"[SL] {symbol} closed - {exit_type} - {profit_pct:+.2f}% - {final_exit_r:.2f}R (total {total_r:.2f}R)")

# ===================== PERFORMANCE REPORT =====================
async def generate_performance_report() -> str:
    try:
        reset_daily_counters()
        try:
            trade_history = db_manager.get_trade_history()
        except Exception as e:
            logger.error(f"[Report] DB error: {e}")
            trade_history = []
        total_trades = len(trade_history)
        hard_gates_total = STATS['hard_gates_passed'] + STATS['hard_gates_failed']
        hard_gates_success_rate = (STATS['hard_gates_passed'] / hard_gates_total * 100) if hard_gates_total > 0 else 0
        if total_trades == 0:
            basic_report = f"""
📊 تقرير الأداء - Quantum Flow v1.8.9 ULTIMATE INSTITUTIONAL EDITION (هدفين + فلتر مرن)
━━━━━━━━━━━━━━━━━━━━━━━━━━
🧾 الوضع
• LIVE: {'ON' if is_live_trading_enabled() else 'OFF'}
• PAPER: {'ON' if is_paper_trading_enabled() else 'OFF'}
• Entry: LIMIT (Zone-based)
• No-Chasing Gate: {'ON' if CONFIG.get('ENABLE_PRICE_ACCEPTANCE_GATE', True) else 'OFF'}
• Exits: Internal (Market Sell Safe)
• SL on Exchange: {'ON' if CONFIG.get('LIVE_PLACE_SL_ORDER') else 'OFF'}

✅ التحسينات المؤسسية المطبقة
• ✅ فلتر EMA50>EMA200 المرن (إلزامي لـ A+، اختياري بشرط للـ A/B)
• ✅ نظام هدفين فقط مع التريلينج كـ runner
• ✅ تعديلات القيم الصارمة (trend strength 70, min score 68, spread 0.10, cooldown 1200)
• ✅ تفعيل أمر SL على المنصة افتراضياً
• ✅ تحسين Order Flow Sampling
• ✅ RSI window (30-75) قابلة للتعديل
• ✅ Micro Structure Protection
• ✅ Slippage Recheck بعد التنفيذ
• ✅ Consecutive Loss Guard
• ✅ Dynamic Break‑Even
• ✅ BTC Correlation Filter (corr < 0.3)
• ✅ مكافأة إضافية لـ BELOW_VALUE + BULLISH_FLOW
• ✅ Edge Intelligence Engine
• ✅ Trailing Stop مع حماية مناطق السيولة
• ✅ Symbol Filter (أعلى 80 سيولة، حجم > 2M, سعر >= 0.00001) مع استثناء BTC/USDT
• ✅ Market Volatility Filter (ATR% 15m > 8% مرفوض)
• ✅ Quantum Score Rebalance
• ✅ Slippage protection (MAX_SLIPPAGE_PCT = 0.35%)
• ✅ Global consecutive loss circuit breaker
• ✅ إصلاح cache stampede
• ✅ إصلاح TOCTOU في monitor_active_trades
• ✅ إصلاح apply_be_and_trail (تم تمرير tp1 وحذف الاعتماد عليه)
• ✅ إصلاح deadlock في SmartCache
• ✅ إصلاح النوم داخل القفل في WeightedRateLimiter
• ✅ إصلاح منطق TALIB_FALLBACK
• ✅ إزالة tp3 نهائياً من المنطق
• ✅ إزالة المعامل غير المستخدم r_multiple من partial_exit
• ✅ دمج وظائف BTC في دالة واحدة
• ✅ إضافة قفل للمتغيرات العامة BTC_TREND
• ✅ إزالة ازدواجية ديكورات القياس
• ✅ تخزين edge_engine مرة واحدة
• ✅ تحسين validate_config مع فحوصات إضافية
• ✅ تغيير المسارات الافتراضية لتعتمد على os.getcwd()
• ✅ إزالة الكود الميت في ExponentialBackoffRetry
• ✅ إصلاح TOCTOU في close_trade_full (تحديث sl_order_done بعد البيع)
• ✅ تعديل cooldown ليكون فقط في وضعي LIVE/PAPER

🧯 Daily Circuit
• Enabled: {'ON' if CONFIG.get('ENABLE_DAILY_MAX_LOSS', True) else 'OFF'}
• Max Loss (R): {CONFIG.get('DAILY_MAX_LOSS_R')}
• Max Daily A+: {CONFIG.get('MAX_DAILY_A_PLUS', 7)}

⏳ Cooldown
• Seconds: {CONFIG.get('SYMBOL_COOLDOWN_SEC')}

🆘 Emergency Monitor
• Enabled: {'ON' if is_live_trading_enabled() else 'OFF'}

📊 Metrics
• Signal Generation: p50={metrics.get_percentiles('signal_generation').get('p50', 0):.3f}s, p95={metrics.get_percentiles('signal_generation').get('p95', 0):.3f}s
• MTF Analysis: p50={metrics.get_percentiles('mtf_analysis').get('p50', 0):.3f}s, p95={metrics.get_percentiles('mtf_analysis').get('p95', 0):.3f}s
• Order Flow: p50={metrics.get_percentiles('order_flow_analysis').get('p50', 0):.3f}s, p95={metrics.get_percentiles('order_flow_analysis').get('p95', 0):.3f}s
{"🔄 " + str(len(ACTIVE_TRADES)) + " صفقة تم استردادها" if ACTIVE_TRADES else ""}
جاهز! 🎯
"""
            cache_stats = cache.get_stats()
            basic_report += f"""
📊 ذاكرة التخزين المؤقت
• معدل النجاح: {cache_stats['hit_rate']:.1f}%
• الحجم: {cache_stats['size']}
• متوسط الوصولات: {cache_stats['avg_access_count']:.1f}
"""
            return basic_report

        winning_trades = [t for t in trade_history if safe_float(t['r_multiple']) > 0]
        losing_trades = [t for t in trade_history if safe_float(t['r_multiple']) <= 0]
        total_won = len(winning_trades)
        total_lost = len(losing_trades)
        win_rate = (total_won / total_trades * 100) if total_trades > 0 else 0
        total_r_history = sum(safe_float(t['r_multiple']) for t in trade_history)
        avg_r = total_r_history / total_trades if total_trades > 0 else 0

        if total_won > 0 and total_lost > 0:
            avg_win_r = sum(safe_float(t['r_multiple']) for t in winning_trades) / total_won
            avg_loss_r = abs(sum(safe_float(t['r_multiple']) for t in losing_trades) / total_lost) if total_lost > 0 else 1.0
            win_pct = total_won / total_trades
            loss_pct = total_lost / total_trades
            expectancy = (win_pct * avg_win_r) - (loss_pct * avg_loss_r)
        elif total_won > 0:
            avg_win_r = sum(safe_float(t['r_multiple']) for t in winning_trades) / total_won
            expectancy = avg_win_r
        else:
            expectancy = avg_r

        total_profit = sum(safe_float(t['profit_pct']) for t in trade_history if safe_float(t['profit_pct']) > 0)
        total_loss = abs(sum(safe_float(t['profit_pct']) for t in trade_history if safe_float(t['profit_pct']) < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        metrics_summary = metrics.get_summary()

        try:
            edge_engine = await get_edge_engine()
            score_r_corr = getattr(edge_engine, 'score_r_corr', 1.0)
            trades_list = getattr(edge_engine, 'trades', [])
        except Exception:
            score_r_corr = 1.0
            trades_list = []

        report = f"""
📊 تقرير الأداء - Quantum Flow v1.8.9 ULTIMATE INSTITUTIONAL EDITION (هدفين + فلتر مرن)
━━━━━━━━━━━━━━━━━━━━━━━━━━
🧾 الوضع
• LIVE: {'ON' if is_live_trading_enabled() else 'OFF'}
• PAPER: {'ON' if is_paper_trading_enabled() else 'OFF'}
• Cooldown (sec): {CONFIG.get('SYMBOL_COOLDOWN_SEC')}

🎯 الإشارات (اليوم/الإجمالي)
• المولدة: {STATS['signals_generated']}
• A+ : {STATS['signals_a_plus']} (اليوم: {STATS.get('daily_a_plus_count', 0)}/{CONFIG.get('MAX_DAILY_A_PLUS', 7)})
• A : {STATS['signals_a']}
• B : {STATS['signals_b']}
• متوسط النقاط: {STATS['avg_quantum_score']:.1f}/100

✅ البوابات الإجبارية
• تم اجتيازها: {STATS['hard_gates_passed']}
• فشلت: {STATS['hard_gates_failed']}
• نسبة النجاح: {hard_gates_success_rate:.1f}%

📈 الصفقات (من السجل)
• الإجمالي: {total_trades}
• الرابحة: {total_won} ✅
• الخاسرة: {total_lost} ❌
• معدل النجاح: {win_rate:.1f}%

🎯 إحصائيات الأهداف الجزئية
• الهدف 1: {STATS['tp1_hits']}
• الهدف 2: {STATS['tp2_hits']}
• الخروج الجزئي: {STATS['trades_partial']}

💰 مقاييس الأداء
• متوسط R: {avg_r:.2f}R
• القيمة المتوقعة: {expectancy:.2f}R
• معامل الربح: {profit_factor:.2f}
• إجمالي R: {total_r_history:.2f}R
• إجمالي الربح: {total_profit:.2f}%

🧾 LIVE
• أوامر تم إرسالها: {STATS['live_orders_placed']}
• أوامر تم تنفيذها: {STATS['live_orders_filled']}
• أوامر تم إلغاؤها: {STATS['live_orders_canceled']}
• عمليات بيع منفذة: {STATS['live_sells_executed']}
• أخطاء أوامر: {STATS['live_order_errors']}
• حالات طوارئ: {STATS.get('live_emergencies', 0)}

🧯 Daily Circuit
• Realized R اليوم: {daily_circuit.get_state().get('realized_r', 0.0):.2f}R
• Blocked: {'YES' if daily_circuit.get_state().get('blocked') else 'NO'}

🔄 النظام
• الصفقات النشطة: {len(ACTIVE_TRADES)}
• أخطاء API: {STATS['api_errors']}
• إعادة المحاولات: {STATS['retries_count']}
• Circuit Breaker: {api_circuit.get_state().get('state')}
• Lock Failures: {sum(v['count'] for v in bot.lock_manager.failed_locks.values())}
• Blacklisted Symbols (lock): {sum(1 for s in bot.lock_manager.failed_locks.keys() if bot.lock_manager.is_blacklisted(s))}
• Consecutive Loss Blacklist: {len(bot.consecutive_loss_blacklist)}
• Global Consecutive Losses: {STATS.get('global_consecutive_losses', 0)} / {CONFIG.get('MAX_CONSECUTIVE_LOSSES', 3)}
"""
        if metrics_summary:
            report += f"""
⏱️ مقاييس الأداء
"""
            for operation, data in metrics_summary.items():
                report += f"• {operation}: p50={data.get('p50', 0):.3f}s, p95={data.get('p95', 0):.3f}s, count={data.get('count', 0)}\n"
        cache_stats = cache.get_stats()
        report += f"""
📊 ذاكرة التخزين المؤقت
• معدل النجاح: {cache_stats['hit_rate']:.1f}%
• الحجم: {cache_stats['size']}
• متوسط الوصولات: {cache_stats['avg_access_count']:.1f}

🧠 Edge Intelligence Engine
• Score-R Correlation: {score_r_corr:.2f}
• Trades Analyzed: {len(trades_list)}
"""
        return report
    except Exception as e:
        logger.error(f"[Report Generation Error] {e}")
        return f"❌ فشل إنشاء التقرير: {str(e)[:100]}"

# ===================== COMPUTATIONAL OPTIMIZATION =====================
async def toggle_computational_features():
    loop_count = STATS.get("loop_count", 0)
    # ملاحظة: أزلنا CONFIG["_ORDER_FLOW_SAMPLING_OK"] لأنه غير مستخدم
    CONFIG["_VOLUME_PROFILE_SAMPLING_OK"] = (loop_count % 5 == 0)
    if loop_count % 7 == 0:
        await cache.smart_cache_cleanup()
    if loop_count % 100 == 0:
        now = time.time()
        async with bot.correlation_cache_lock:
            bot.correlation_cache = {k: v for k, v in bot.correlation_cache.items() if now - v[1] < 3600}

# ===================== HEALTH CHECK ENDPOINT =====================
health_rate_limit = {}
health_rate_limit_lock = asyncio.Lock()
last_health_cleanup = time.time()

async def health_check_handler(request):
    global last_health_cleanup
    client_ip = request.remote if request.remote else 'unknown'
    async with health_rate_limit_lock:
        now = time.time()
        if now - last_health_cleanup > 600:
            for ip in list(health_rate_limit.keys()):
                health_rate_limit[ip] = [t for t in health_rate_limit[ip] if now - t < 60]
                if not health_rate_limit[ip]:
                    del health_rate_limit[ip]
            last_health_cleanup = now
        health_rate_limit[client_ip] = [t for t in health_rate_limit.get(client_ip, []) if now - t < 60]
        if len(health_rate_limit.get(client_ip, [])) >= 100:
            return web.json_response({"error": "Rate limit exceeded"}, status=429)
        health_rate_limit.setdefault(client_ip, []).append(now)
    health_status = "healthy"
    issues = []
    if len(ACTIVE_TRADES) > CONFIG.get("LIVE_MAX_OPEN_TRADES", 5):
        health_status = "degraded"
        issues.append("too_many_active_trades")
    emergency_count = sum(1 for t in ACTIVE_TRADES.values() if t.emergency_state)
    if emergency_count > 0:
        health_status = "warning"
        issues.append(f"{emergency_count}_emergency_states")
    if api_circuit.get_state()["state"] == "OPEN":
        health_status = "degraded"
        issues.append("circuit_breaker_open")
    if daily_circuit.is_blocked():
        health_status = "warning"
        issues.append("daily_loss_limit_reached")
    cache_stats = cache.get_stats()
    if cache_stats["hit_rate"] < 50:
        health_status = "warning"
        issues.append("low_cache_hit_rate")
    lock_failures = sum(v['count'] for v in bot.lock_manager.failed_locks.values())
    if lock_failures > 10:
        health_status = "warning"
        issues.append(f"high_lock_failures({lock_failures})")
    blacklisted_count = sum(1 for s in bot.lock_manager.failed_locks.keys() if bot.lock_manager.is_blacklisted(s))
    if blacklisted_count > 5:
        health_status = "warning"
        issues.append(f"high_blacklisted_symbols({blacklisted_count})")
    response_data = {
        "status": health_status,
        "timestamp": now_utc_iso(),
        "issues": issues,
        "metrics": {
            "active_trades": len(ACTIVE_TRADES),
            "emergency_states": emergency_count,
            "circuit_breaker": api_circuit.get_state(),
            "daily_circuit": daily_circuit.get_state(),
            "cache_stats": cache_stats,
            "lock_failures": lock_failures,
            "blacklisted_symbols": blacklisted_count,
            "consecutive_loss_blacklist": len(bot.consecutive_loss_blacklist),
            "performance_metrics": metrics.get_summary()
        }
    }
    return web.json_response(response_data)

# ===================== MEMORY MONITOR =====================
async def memory_monitor_task():
    if not CONFIG.get("ENABLE_MEMORY_MONITORING", False):
        return
    tracemalloc.start()
    logger.info("🧠 Memory monitoring enabled")
    while not shutdown_manager.should_stop:
        await asyncio.sleep(300)
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:5]
        logger.info("🧠 Top Memory Usage:")
        for stat in top_stats:
            logger.info(f" {stat}")

# ===================== ENHANCED MAIN LOOP =====================
async def main_loop(exchange):
    emergency_monitor_task = None
    checkpoint_task = None
    memory_task = None
    runner = None
    site = None
    try:
        logger.info("="*70)
        logger.info("🚀 QUANTUM FLOW TRADING BOT v1.8.9 - ULTIMATE INSTITUTIONAL EDITION (هدفين + فلتر مرن)")
        logger.info("✅ جميع التحسينات المؤسسية المطلوبة والإصلاحات الحرجة مطبقة")
        logger.info("="*70)
        logger.info(f"البورصة: {CONFIG['EXCHANGE'].upper()}")
        logger.info(f"الأطر الزمنية: {CONFIG['TF_TREND']}, {CONFIG['TF_STRUCTURE']}, {CONFIG['TF_ENTRY']}")
        logger.info(f"الحد الأدنى للنقاط: {CONFIG['MIN_QUANTUM_SCORE']}")
        logger.info(f"نظام البوابات الإجبارية: {'مفعل' if CONFIG['ENABLE_HARD_GATES'] else 'معطل'}")
        logger.info(f"فلتر BTC: {'مفعل' if CONFIG['ENABLE_BTC_FILTER'] else 'معطل'}")
        logger.info(f"LONG ONLY: {'✅' if CONFIG['LONG_ONLY'] else '❌'}")
        logger.info(f"LIVE TRADING: {'✅' if is_live_trading_enabled() else '❌'}")
        logger.info(f"PAPER TRADING: {'✅' if is_paper_trading_enabled() else '❌'}")
        logger.info(f"COOLDOWN: {CONFIG.get('SYMBOL_COOLDOWN_SEC', 0)} sec")
        logger.info(f"ENTRY TYPE: {CONFIG.get('ENTRY_ORDER_TYPE')}")
        logger.info(f"NO-CHASING GATE: {'✅' if CONFIG.get('ENABLE_PRICE_ACCEPTANCE_GATE', True) else '❌'}")
        logger.info(f"CIRCUIT BREAKER: {'✅' if CONFIG.get('CIRCUIT_BREAKER_ENABLED', True) else '❌'}")
        logger.info(f"HEALTH CHECK: {'✅' if CONFIG.get('ENABLE_HEALTH_CHECK', False) else '❌'}")
        logger.info(f"MEMORY MONITORING: {'✅' if CONFIG.get('ENABLE_MEMORY_MONITORING', False) else '❌'}")
        logger.info(f"ENHANCED LOCK MANAGER: ✅ (مع Recovery و Blacklisting و TTL)")
        logger.info(f"METRICS COLLECTOR: ✅")
        logger.info(f"EXPONENTIAL BACKOFF RETRY: ✅ (يشمل ccxt exceptions)")
        logger.info("✅ جميع التحسينات الجديدة مدمجة:")
        logger.info(" 1. فلتر EMA50>EMA200 المرن (إلزامي لـ A+، اختياري بشرط للـ A/B)")
        logger.info(" 2. نظام هدفين فقط مع التريلينج كـ runner")
        logger.info(" 3. تعديلات القيم الصارمة (trend strength 70, min score 68, spread 0.10, cooldown 1200)")
        logger.info(" 4. تفعيل أمر SL على المنصة افتراضياً")
        logger.info(" 5. تحسين Order Flow Sampling")
        logger.info(" 6. RSI window (30-75) قابلة للتعديل")
        logger.info(" 7. Micro Structure Protection")
        logger.info(" 8. Slippage Recheck بعد التنفيذ")
        logger.info(" 9. Consecutive Loss Guard")
        logger.info("10. Dynamic Break‑Even")
        logger.info("11. BTC Correlation Filter (corr < 0.3)")
        logger.info("12. مكافأة إضافية لـ BELOW_VALUE + BULLISH_FLOW")
        logger.info("13. Edge Intelligence Engine")
        logger.info("14. Trailing Stop مع حماية مناطق السيولة")
        logger.info("15. Symbol Filter (أعلى 80 سيولة، حجم > 2M, سعر >= 0.00001) مع استثناء BTC/USDT")
        logger.info("16. Market Volatility Filter (ATR% 15m > 8% مرفوض)")
        logger.info("17. Quantum Score Rebalance")
        logger.info("18. Slippage protection (MAX_SLIPPAGE_PCT = 0.35%)")
        logger.info("19. Global consecutive loss circuit breaker")
        logger.info("20. إصلاح ترتيب فحص الأهداف (if مستقل لكل هدف)")
        logger.info("21. إضافة استثناءات ccxt إلى ExponentialBackoffRetry")
        logger.info("22. حماية cache stampede")
        logger.info("23. إزالة reconcile_balances الوهمية")
        logger.info("24. تعديل مسارات /content/ لتكون قابلة للتكوين")
        logger.info("25. إعادة تسمية fetch_ticker_with_lock إلى fetch_ticker_safe")
        logger.info("26. إصلاح SL الاحتياطي باستخدام ATR/CONFIG")
        logger.info("27. إصلاح TOCTOU في monitor_active_trades")
        logger.info("28. إصلاح apply_be_and_trail (تم تمرير tp1 وحذف الاعتماد عليه)")
        logger.info("29. إصلاح deadlock في SmartCache")
        logger.info("30. إصلاح النوم داخل القفل في WeightedRateLimiter")
        logger.info("31. إصلاح منطق TALIB_FALLBACK")
        logger.info("32. إزالة tp3 نهائياً من المنطق")
        logger.info("33. إزالة المعامل غير المستخدم r_multiple من partial_exit")
        logger.info("34. دمج وظائف BTC في دالة واحدة (check_btc_conditions)")
        logger.info("35. إضافة قفل للمتغيرات العامة BTC_TREND")
        logger.info("36. إزالة ازدواجية ديكورات القياس")
        logger.info("37. تخزين edge_engine مرة واحدة")
        logger.info("38. تحسين validate_config مع فحوصات إضافية")
        logger.info("39. تغيير المسارات الافتراضية لتعتمد على os.getcwd()")
        logger.info("40. إزالة الكود الميت في ExponentialBackoffRetry")
        logger.info("41. إصلاح TOCTOU في close_trade_full (تحديث sl_order_done بعد البيع)")
        logger.info("42. تعديل cooldown ليكون فقط في وضعي LIVE/PAPER")
        logger.info("="*70)

        db_manager.init_database()
        if CONFIG["ENABLE_DB_PERSISTENCE"]:
            loaded_trades = db_manager.load_active_trades()
            if loaded_trades:
                ACTIVE_TRADES.update(loaded_trades)
                logger.info(f"[الاسترداد] تم تحميل {len(loaded_trades)} صفقة نشطة (DB)")

        await exchange.load_markets()
        logger.info(f"متصل! الأسواق: {len(exchange.markets)}")

        if is_live_trading_enabled():
            await validate_stop_loss_capability(exchange)

        if is_live_trading_enabled():
            await ensure_live_trading_ready(exchange)

        # تهيئة edge_engine مرة واحدة
        bot.edge_engine = await get_edge_engine()

        if CONFIG.get("ENABLE_MEMORY_MONITORING", False):
            memory_task = asyncio.create_task(memory_monitor_task())
            shutdown_manager.add_task(memory_task)

        if CONFIG.get("ENABLE_HEALTH_CHECK", False):
            app = web.Application()
            app.router.add_get('/health', health_check_handler)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', CONFIG.get("HEALTH_CHECK_PORT", 8080))
            await site.start()
            logger.info(f"🏥 Health check server started on :{CONFIG.get('HEALTH_CHECK_PORT', 8080)}/health")

        if CONFIG.get("ENABLE_CHECKPOINTS", True):
            checkpoint_task = asyncio.create_task(checkpoint_saver())
            shutdown_manager.add_task(checkpoint_task)

        if is_live_trading_enabled():
            emergency_monitor_task = asyncio.create_task(emergency_state_monitor(exchange))
            shutdown_manager.add_task(emergency_monitor_task)
            logger.info("[Main] Emergency state monitor started")

        await send_telegram(f"""
🚀 تم تشغيل Quantum Flow Bot v1.8.9 - ULTIMATE INSTITUTIONAL EDITION (هدفين + فلتر مرن)
🧾 الوضع
• LIVE TRADING: {'ON' if is_live_trading_enabled() else 'OFF'}
• PAPER TRADING: {'ON' if is_paper_trading_enabled() else 'OFF'}
• Entry: LIMIT (Zone-based)
• No-Chasing Gate: {'ON' if CONFIG.get('ENABLE_PRICE_ACCEPTANCE_GATE', True) else 'OFF'}
• Exits: Internal (Market Sell Safe)
• SL on Exchange: {'ON' if CONFIG.get('LIVE_PLACE_SL_ORDER') else 'OFF'}

✅ التحسينات المؤسسية المطبقة
• ✅ فلتر EMA50>EMA200 المرن (إلزامي لـ A+، اختياري بشرط للـ A/B)
• ✅ نظام هدفين فقط مع التريلينج كـ runner
• ✅ تعديلات القيم الصارمة (trend strength 70, min score 68, spread 0.10, cooldown 1200)
• ✅ تفعيل أمر SL على المنصة افتراضياً
• ✅ تحسين Order Flow Sampling
• ✅ RSI window (30-75) قابلة للتعديل
• ✅ Micro Structure Protection
• ✅ Slippage Recheck بعد التنفيذ
• ✅ Consecutive Loss Guard
• ✅ Dynamic Break‑Even
• ✅ BTC Correlation Filter (corr < 0.3)
• ✅ مكافأة إضافية لـ BELOW_VALUE + BULLISH_FLOW
• ✅ Edge Intelligence Engine
• ✅ Trailing Stop مع حماية مناطق السيولة
• ✅ Symbol Filter (أعلى 80 سيولة، حجم > 2M, سعر >= 0.00001) مع استثناء BTC/USDT
• ✅ Market Volatility Filter (ATR% 15m > 8% مرفوض)
• ✅ Quantum Score Rebalance
• ✅ Slippage protection (MAX_SLIPPAGE_PCT = 0.35%)
• ✅ Global consecutive loss circuit breaker
• ✅ إصلاح ترتيب فحص الأهداف (if مستقل لكل هدف)
• ✅ إضافة استثناءات ccxt إلى ExponentialBackoffRetry
• ✅ حماية cache stampede
• ✅ إزالة reconcile_balances الوهمية
• ✅ تعديل مسارات /content/ لتكون قابلة للتكوين
• ✅ إعادة تسمية fetch_ticker_with_lock إلى fetch_ticker_safe
• ✅ إصلاح SL الاحتياطي باستخدام ATR/CONFIG
• ✅ إصلاح TOCTOU في monitor_active_trades
• ✅ إصلاح apply_be_and_trail (تم تمرير tp1 وحذف الاعتماد عليه)
• ✅ إصلاح deadlock في SmartCache
• ✅ إصلاح النوم داخل القفل في WeightedRateLimiter
• ✅ إصلاح منطق TALIB_FALLBACK
• ✅ إزالة tp3 نهائياً من المنطق
• ✅ إزالة المعامل غير المستخدم r_multiple من partial_exit
• ✅ دمج وظائف BTC في دالة واحدة
• ✅ إضافة قفل للمتغيرات العامة BTC_TREND
• ✅ إزالة ازدواجية ديكورات القياس
• ✅ تخزين edge_engine مرة واحدة
• ✅ تحسين validate_config
• ✅ تغيير المسارات الافتراضية لتعتمد على os.getcwd()
• ✅ إزالة الكود الميت في ExponentialBackoffRetry
• ✅ إصلاح TOCTOU في close_trade_full (تحديث sl_order_done بعد البيع)
• ✅ تعديل cooldown ليكون فقط في وضعي LIVE/PAPER

🧯 Daily Circuit
• Enabled: {'ON' if CONFIG.get('ENABLE_DAILY_MAX_LOSS', True) else 'OFF'}
• Max Loss (R): {CONFIG.get('DAILY_MAX_LOSS_R')}
• Max Daily A+: {CONFIG.get('MAX_DAILY_A_PLUS', 7)}

⏳ Cooldown
• Seconds: {CONFIG.get('SYMBOL_COOLDOWN_SEC')}

🆘 Emergency Monitor
• Enabled: {'ON' if is_live_trading_enabled() else 'OFF'}

📊 Metrics Collection
• Signal Generation Latency
• MTF Analysis Performance
• Order Flow Analysis Timing
• Overall System Health
{"🔄 " + str(len(ACTIVE_TRADES)) + " صفقة تم استردادها" if ACTIVE_TRADES else ""}
جاهز! 🎯
""")

        loop_count = 0
        all_symbols = []
        while not shutdown_manager.should_stop:
            try:
                loop_start = time.time()
                loop_count += 1
                STATS["loop_count"] = loop_count
                await toggle_computational_features()
                try:
                    if not bot.edge_engine.should_trade():
                        logger.info("[EdgeEngine] System state PAUSED - sleeping for 60 seconds")
                        await asyncio.sleep(60)
                        continue
                except Exception as e:
                    logger.error(f"[EdgeEngine] Error checking trade state: {e}")

                btc_status = await check_btc_conditions(exchange)
                if not btc_status['safe_to_trade']:
                    logger.warning(f"[Main] التداول متوقف مؤقتاً - BTC {btc_status['trend']}")
                    await asyncio.sleep(60)
                    continue

                if loop_count % 30 == 0 or not all_symbols:
                    all_symbols = await get_filtered_symbols(exchange)

                await monitor_active_trades(exchange)

                if all_symbols:
                    batch_size = CONFIG["BATCH_SIZE"]
                    for i in range(0, len(all_symbols), batch_size):
                        batch = all_symbols[i:i+batch_size]
                        await process_symbol_batch(exchange, batch)
                        await asyncio.sleep(1)

                if loop_count % 50 == 0:
                    report = await generate_performance_report()
                    await send_telegram(report)
                    try:
                        edge_report = bot.edge_engine.get_telemetry_report()
                        await send_telegram(f"<code>{edge_report}</code>")
                        if getattr(bot.edge_engine, 'score_r_corr', 1.0) < 0.1 and len(getattr(bot.edge_engine, 'trades', [])) >= 20:
                            await send_telegram(
                                f"⚠️ <b>Quantum score losing predictive power</b>\n"
                                f"Correlation: {getattr(bot.edge_engine, 'score_r_corr', 1.0):.2f}\n"
                                f"Trades analyzed: {len(getattr(bot.edge_engine, 'trades', []))}",
                                critical=False
                            )
                    except Exception as e:
                        logger.error(f"[EdgeEngine] Telemetry error: {str(e)}")

                loop_time = time.time() - loop_start
                sleep_time = max(5, 15 - loop_time)
                if CONFIG["DEBUG_MODE"] or loop_count % 10 == 0:
                    logger.info(f"[دورة {loop_count}] الوقت: {loop_time:.1f}ث، الانتظار: {sleep_time:.1f}ث، "
                                f"الإشارات: {STATS['signals_generated']}, النشطة: {len(ACTIVE_TRADES)}, "
                                f"paper_opened={STATS.get('paper_trades_opened',0)}, cooldown={CONFIG.get('SYMBOL_COOLDOWN_SEC')}, "
                                f"البوابات: {STATS['hard_gates_passed']}/{STATS['hard_gates_failed']}, "
                                f"live: placed={STATS['live_orders_placed']}, filled={STATS['live_orders_filled']}, "
                                f"canceled={STATS['live_orders_canceled']}, emergencies={STATS.get('live_emergencies',0)}, "
                                f"dailyR={daily_circuit.get_state().get('realized_r',0.0):.2f}, blocked={daily_circuit.get_state().get('blocked')}, "
                                f"circuit={api_circuit.get_state().get('state')}, "
                                f"lock_failures={sum(v['count'] for v in bot.lock_manager.failed_locks.values())}, "
                                f"blacklisted={sum(1 for s in bot.lock_manager.failed_locks.keys() if bot.lock_manager.is_blacklisted(s))}, "
                                f"loss_blacklist={len(bot.consecutive_loss_blacklist)}, "
                                f"global_losses={STATS.get('global_consecutive_losses',0)}, "
                                f"edge_state={bot.edge_engine.system_state}, edge_mult={bot.edge_engine.risk_multiplier():.2f}"
                    )
                await asyncio.sleep(sleep_time)
            except KeyboardInterrupt:
                logger.info("\n[النظام] تم استلام إشارة الإيقاف")
                break
            except Exception as e:
                logger.error(f"[خطأ الدورة] {str(e)}")
                if CONFIG["DEBUG_MODE"]:
                    traceback.print_exc()
                await send_telegram(
                    f"🚨 خطأ حرج\n\n{escape_html(str(e)[:500])}",
                    critical=True
                )
                await asyncio.sleep(30)
    except Exception as e:
        logger.error(f"[خطأ فادح] {str(e)}")
        traceback.print_exc()
        await send_telegram(
            f"💥 خطأ فادح - توقف البوت\n\n{escape_html(str(e)[:500])}",
            critical=True
        )
    finally:
        await shutdown_manager.shutdown()
        if runner:
            try:
                await runner.cleanup()
                logger.info("✅ Health check server cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up health check: {e}")

# ===================== EMERGENCY STATE MONITOR =====================
async def emergency_state_monitor(exchange):
    MAX_TOTAL_ATTEMPTS = 10
    while not shutdown_manager.should_stop:
        try:
            await asyncio.sleep(300)
            emergency_trades = []
            acquired_symbols = await bot.lock_manager.acquire_all_locks()
            if not acquired_symbols:
                continue
            try:
                for symbol, trade in list(ACTIVE_TRADES.items()):
                    if trade.emergency_state:
                        emergency_trades.append((symbol, copy.deepcopy(trade)))
            finally:
                bot.lock_manager.release_all_locks(acquired_symbols)
            if not emergency_trades:
                continue
            for symbol, trade_snapshot in emergency_trades:
                try:
                    if trade_snapshot.emergency_attempts >= MAX_TOTAL_ATTEMPTS:
                        await send_telegram(
                            f"🛑 EMERGENCY ABANDONED: {symbol}\n"
                            f"Max attempts ({MAX_TOTAL_ATTEMPTS}) exceeded",
                            critical=True
                        )
                        if await bot.get_trade_lock(symbol):
                            try:
                                if symbol in ACTIVE_TRADES:
                                    ACTIVE_TRADES[symbol].emergency_state = False
                                    await asyncio.to_thread(db_manager.save_trade, ACTIVE_TRADES[symbol])
                            finally:
                                bot.release_trade_lock(symbol)
                        continue
                    last_attempt = datetime.fromisoformat(
                        trade_snapshot.emergency_last_attempt.replace('Z', '+00:00')
                    )
                    if (datetime.now(timezone.utc) - last_attempt).total_seconds() < 600:
                        continue
                    await rate_limiter.wait_if_needed(weight=2)
                    ticker = await exchange.fetch_ticker(symbol)
                    rate_limiter.reset_errors()
                    current_price = safe_float(ticker.get('last'))
                    if not validate_price(current_price):
                        continue
                    remaining_base = trade_snapshot.entry_fill_amount * trade_snapshot.remaining_position
                    sell_order = await market_sell_safe(exchange, symbol, remaining_base, max_retries=2)
                    if sell_order:
                        fill_price = safe_float(sell_order.get('average')) or safe_float(sell_order.get('price')) or current_price
                        risk = trade_snapshot.entry - trade_snapshot.original_sl
                        r_multiple = (fill_price - trade_snapshot.entry) / risk if risk > 0 else 0
                        await close_trade_full(symbol, fill_price, "EMERGENCY_RECOVERY", exchange=exchange, sell_order_info=sell_order)
                        await send_telegram(
                            f"🟢 EMERGENCY RECOVERY SUCCESS\n\n"
                            f"الرمز: {escape_html(symbol)}\n"
                            f"تم البيع الناجح بعد {trade_snapshot.emergency_attempts} محاولات فاشلة.\n"
                            f"تم إغلاق الصفقة تلقائيًا."
                        )
                    else:
                        if await bot.get_trade_lock(symbol):
                            try:
                                if symbol in ACTIVE_TRADES:
                                    ACTIVE_TRADES[symbol].emergency_attempts += 1
                                    ACTIVE_TRADES[symbol].emergency_last_attempt = now_utc_iso()
                                    await asyncio.to_thread(db_manager.save_trade, ACTIVE_TRADES[symbol])
                            finally:
                                bot.release_trade_lock(symbol)
                        if trade_snapshot.emergency_attempts % 3 == 0:
                            await send_telegram(
                                f"🔄 EMERGENCY REMINDER (Attempt {trade_snapshot.emergency_attempts})\n\n"
                                f"الرمز: {escape_html(symbol)}\n"
                                f"السبب: {escape_html(trade_snapshot.emergency_reason[:100])}\n"
                                f"الكمية المتبقية (BASE): {remaining_base}\n\n"
                                f"⚠️ الصفقة لا تزال في حالة الطوارئ. يرجى البيع يدويًا أو الانتظار لمحاولة تلقائية أخرى."
                            )
                except Exception as e:
                    logger.error(f"[Emergency Monitor Error] {symbol}: {str(e)[:100]}")
                    continue
        except Exception as e:
            logger.error(f"[Emergency Monitor Main Error] {str(e)}")
            await asyncio.sleep(60)

# ===================== ENTRY POINT =====================
async def async_main():
    exchange = None
    try:
        logger.info("\n" + "="*70)
        logger.info("QUANTUM FLOW v1.8.9 - ULTIMATE INSTITUTIONAL EDITION (هدفين + فلتر مرن)")
        logger.info("✅ جميع التحسينات المؤسسية المطلوبة والإصلاحات الحرجة مطبقة")
        logger.info("="*70)

        if not SCIPY_AVAILABLE:
            logger.warning("⚠️ scipy غير متوفر - استخدام الطرق البديلة")
        if not ML_AVAILABLE:
            logger.warning("⚠️ sklearn غير متوفر - ميزات ML معطلة")
        if not TALIB_AVAILABLE:
            logger.info("ℹ️ TA-Lib غير متوفر - سيتم استخدام ta (عادي)")

        load_telegram_from_env()
        load_mexc_from_env()
        _ensure_runtime_paths()

        global db_manager
        db_manager = DatabaseManager(CONFIG["DB_PATH"])
        try:
            daily_circuit.attach_db(db_manager)
        except Exception as e:
            logger.error(f"[Daily Circuit] attach_db failed: {e}")

        if CONFIG["ENABLE_DB_PERSISTENCE"]:
            logger.info("✅ ثبات قاعدة البيانات مفعّل")

        try:
            validate_config()
        except Exception as e:
            logger.error(f"❌ فشل التحقق من الإعدادات: {e}")
            return

        try:
            load_checkpoint()
        except Exception:
            pass

        if is_live_trading_enabled():
            if not CONFIG.get("MEXC_API_KEY") or not CONFIG.get("MEXC_API_SECRET"):
                logger.warning("⚠️ LIVE TRADING مفعّل لكن مفاتيح MEXC غير موجودة في ENV - سيتم تعطيل التنفيذ الحقيقي")
                CONFIG["ENABLE_LIVE_TRADING"] = False

        exchange = getattr(ccxt, CONFIG['EXCHANGE'])({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            'timeout': 30000,
        })
        if is_live_trading_enabled():
            exchange.apiKey = CONFIG.get("MEXC_API_KEY", "")
            exchange.secret = CONFIG.get("MEXC_API_SECRET", "")

        def _handle_signal(sig, frame):
            logger.warning(f"[Signal] Received {sig}, initiating graceful shutdown...")
            shutdown_manager.should_stop = True

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

        await main_loop(exchange)

    except KeyboardInterrupt:
        logger.info("\n[النظام] تم استلام إشارة الإيقاف في async_main")
    except Exception as e:
        logger.error(f"[خطأ فادح في async_main] {str(e)}")
        traceback.print_exc()
        await save_emergency_checkpoint_async(e)
    finally:
        logger.info("🔧 جاري التنظيف النهائي...")
        cleanup_errors = []
        if exchange is not None:
            try:
                await exchange.close()
                logger.info("✅ تم إغلاق اتصال البورصة")
            except Exception as e:
                cleanup_errors.append(f"Exchange: {e}")
        try:
            await close_session()
            logger.info("✅ تم إغلاق جلسة HTTP")
        except Exception as e:
            cleanup_errors.append(f"HTTP Session: {e}")
        if cleanup_errors:
            logger.error(f"Cleanup errors: {', '.join(cleanup_errors)}")

def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("\n👋 توقف البوت بواسطة المستخدم")
    except Exception as e:
        logger.error(f"💥 خطأ فادح في التشغيل: {str(e)}")
        traceback.print_exc()
    logger.info("✅ التنظيف مكتمل")

if __name__ == "__main__":
    main()
