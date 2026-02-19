#!/usr/bin/env python3
"""
QUANTUM FLOW TRADING BOT v1.8.4 - ULTIMATE INSTITUTIONAL EDITION
تم تعديله للتشغيل المستمر على Railway بدون أي تبعيات Colab.
يقرأ متغيرات البيئة TELEGRAM_BOT_TOKEN و TELEGRAM_CHAT_ID.
جميع التحسينات والإصلاحات الحرجة المطلوبة مدمجة بدقة واحترافية.

⚠️ تم إصلاح الخطأ النحوي في السطر 5186: تم تغيير 'main() if not task.done():' إلى 'if not task.done(): main()'
"""

import asyncio
import aiohttp
from aiohttp import web
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
import sys
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
from contextlib import contextmanager
import random
import math

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
        logging.FileHandler('quantum_flow_institutional.log', encoding='utf-8')
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
            if not task.done():  # ✅ تم التأكد من الصياغة الصحيحة
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("✅ All tasks cancelled")

shutdown_manager = GracefulShutdown()

# ===================== AUDIT LOGGING =====================
def log_order_audit(order_type: str, symbol: str, price: float, amount: float, status: str = ""):
    try:
        with open("audit.log", "a", encoding='utf-8') as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()},{order_type},{symbol},{price},{amount},{status}\n")
    except Exception as e:
        logger.error(f"[Audit Log Error] {str(e)}")

# ===================== METRICS COLLECTOR (مُحسّن) =====================
class MetricsCollector:
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
            return wrapper
        return decorator
    
    async def record_error(self, operation: str, error_type: str):
        async with self.lock:
            self.metrics[f"{operation}_errors"].append({
                "timestamp": time.time(),
                "error_type": error_type
            })
    
    def get_percentiles(self, metric: str) -> Dict[str, float]:
        if metric not in self.metrics or not self.metrics[metric]:
            return {}
        
        durations = [m["duration"] for m in self.metrics[metric] if "duration" in m]
        if not durations:
            return {}
        
        durations.sort()
        
        return {
            "p50": durations[len(durations) // 2],
            "p95": durations[int(len(durations) * 0.95)] if len(durations) > 1 else durations[0],
            "p99": durations[int(len(durations) * 0.99)] if len(durations) > 1 else durations[0],
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
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_exception = e
                if attempt == self.max_retries - 1:
                    raise
                delay = min(
                    self.base_delay * (2 ** attempt) + random.uniform(0, 1),
                    self.max_delay
                )
                logger.warning(f"Retry {attempt+1}/{self.max_retries} after {delay:.1f}s: {e}")
                await asyncio.sleep(delay)
        
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
                self.failed_locks.pop(symbol, None)
                return True
            except asyncio.TimeoutError:
                if attempt == self.MAX_LOCK_RETRIES - 1:
                    now = time.time()
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
        if symbol not in self.failed_locks:
            return False
        # TTL check
        if time.time() - self.failed_locks[symbol]['last_failure'] > self.BLACKLIST_TTL:
            # Auto-expire
            self.failed_locks.pop(symbol, None)
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
            "tp3_hits": 0,
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
        }
        self.lock_manager = EnhancedLockManager()
        self.symbol_cooldown: Dict[str, float] = {}
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        self.btc_trend: Optional[Dict] = None
        self.btc_last_check: float = 0
    
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

# ===================== PERFORMANCE LOGGING DECORATOR =====================
def log_execution_time(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if elapsed > 5.0:
            logger.warning(f"⏱️ SLOW: {func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

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
    if not isinstance(text, str):
        text = str(text)
    
    code_pattern = r'<code>(.*?)</code>'
    code_matches = list(re.finditer(code_pattern, text, re.DOTALL))
    
    temp_placeholders = []
    for i, match in enumerate(code_matches):
        placeholder = f"__CODE_{i}__"
        temp_placeholders.append((placeholder, match.group(1)))
        text = text.replace(match.group(0), placeholder, 1)
    
    escape_chars = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
    }
    for char, escape in escape_chars.items():
        text = text.replace(char, escape)
    
    for placeholder, code_content in temp_placeholders:
        text = text.replace(placeholder, f'<code>{code_content}</code>')
    
    return text

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
    # Whitelist approach: فقط الرموز المسموح بها
    pattern = r'^[A-Z0-9]{2,20}/[A-Z]{2,10}$'  # مثال: BTC/USDT, ETH/USDT
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

        runtime_dir = os.path.join(os.getcwd(), "quantum_runtime")
        os.makedirs(runtime_dir, exist_ok=True)

        def _fix_path(key: str, default_filename: str):
            p = CONFIG.get(key, "")
            if isinstance(p, str) and p.startswith("/content/"):
                newp = os.path.join(runtime_dir, default_filename)
                CONFIG[key] = newp

        _fix_path("CHECKPOINT_PATH", "quantum_checkpoint.pkl")
        _fix_path("EMERGENCY_CHECKPOINT_PATH", "quantum_emergency_checkpoint.pkl")
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
    """مؤشر تجزئة ثابت عبر عمليات التشغيل"""
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
    tp3: float
    atr: float = 0.0
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
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
    tp3_order_done: bool = False
    sl_order_done: bool = False
    sl_order_id: str = ""  # معرف أمر وقف الخسارة على المنصة
    emergency_state: bool = False
    emergency_reason: str = ""
    emergency_last_attempt: str = ""
    emergency_attempts: int = 0
    is_paper: bool = False
    execution_mode: str = "SIGNAL"
    entry_assumed: bool = False
    is_exiting: bool = False  # علم لمنع الدخول المتزامن في partial_exit
    _version: int = 0

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
    tp3: float
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

# ===================== FIXED CONFIGURATION =====================
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
    "DAILY_RESET_UTC_HOUR": 0,
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
    "TP2_RR": 2.5,
    "TP3_RR": 4.0,
    "TP1_EXIT_PCT": 0.5,
    "TP2_EXIT_PCT": 0.3,
    "TP3_EXIT_PCT": 0.2,
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
    "MIN_QUANTUM_SCORE": 60,
    "QUANTUM_A_SCORE": 75,
    "QUANTUM_A_PLUS_SCORE": 85,
    "MAX_DAILY_A_PLUS": 5,
    
    # Hard Gates
    "ENABLE_HARD_GATES": True,
    "HARD_GATE_1_MIN_TREND_STRENGTH": 60,
    "HARD_GATE_1_MIN_MTF_ALIGNMENT": 2,
    "HARD_GATE_2_REQUIRE_ZONE": True,
    "HARD_GATE_2_MIN_LG_CONFIDENCE": 75,
    "HARD_GATE_2_OB_FRESHNESS": 10,
    "HARD_GATE_3_REQUIRE_BOOSTER": False,
    
    # Volume Profile
    "ENABLE_VOLUME_PROFILE": True,
    "VOLUME_PROFILE_BINS": 50,
    "VALUE_AREA_PCT": 0.7,
    "HVN_THRESHOLD": 1.0,
    "LVN_THRESHOLD": 0.5,
    
    # BASELINE
    "ENABLE_VOLUME_PROFILE_BASELINE": False,
    "ENABLE_ORDER_FLOW_BASELINE": False,
    
    # Liquidity Grab
    "LG_WICK_MIN_RATIO": 0.3,
    "LG_RECOVERY_MIN": 0.5,
    "LG_VOLUME_MULTIPLIER": 1.5,
    "LG_EQUAL_LOWS_REQUIRED": 3,
    "LG_EQUAL_LOWS_RANGE_ATR_MULT": 0.5,
    
    # Multi-Timeframe
    "MIN_MTF_ALIGNMENT": 2,
    
    # Market Regime Filter
    "ENABLE_MARKET_REGIME_FILTER": True,
    "MIN_ADX_FOR_TREND": 20,
    "MAX_CHASE_MOVE_PCT": 0.03,
    
    # BTC Filter
    "ENABLE_BTC_FILTER": True,
    "BTC_CRASH_THRESHOLD": -3.0,
    "BTC_WARNING_THRESHOLD": -1.5,
    
    # Live Trading
    "ENABLE_LIVE_TRADING": False,
    "MEXC_API_KEY": "",
    "MEXC_API_SECRET": "",
    "LIVE_MAX_OPEN_TRADES": 5,
    "MAX_SPREAD_PCT": 0.1,
    "ENTRY_ORDER_TYPE": "limit",
    "ENTRY_LIMIT_TIMEOUT_SEC": 120,
    "ENTRY_LIMIT_POLL_SEC": 3,
    "LIVE_REQUIRE_SPREAD_FILTER": True,
    "LIVE_RECALIBRATE_LEVELS_ON_FILL": True,
    "LIVE_RECALIBRATION_MODE": "rr",
    "LIVE_REQUIRE_BALANCE_RECONCILIATION": True,
    "MIN_DUST_THRESHOLD": 0.000001,
    "LIVE_PLACE_SL_ORDER": False,          # NEW: وضع أمر وقف خسارة على المنصة
    "LIVE_SL_ORDER_TYPE": "stop-loss",    # أو "stop-limit"
    
    # Paper Trading
    "PAPER_TRADING_MODE": False,
    "PAPER_MAX_OPEN_TRADES": 25,
    
    # Entry Quality Filter
    "ENABLE_ENTRY_QUALITY_FILTER": True,
    "ENTRY_QUALITY_MAX_ATR_PCT_5M": 6.5,
    "ENTRY_QUALITY_MAX_BB_WIDTH_5M": 0.08,
    "ENTRY_QUALITY_MAX_DISTANCE_FROM_ZONE_ATR": 1.2,
    
    # Cooldown
    "SYMBOL_COOLDOWN_SEC": 1800,
    
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
    "CHECKPOINT_PATH": "/content/quantum_checkpoint.pkl",
    "EMERGENCY_CHECKPOINT_PATH": "/content/quantum_emergency_checkpoint.pkl",
    "EMERGENCY_SELL_LOG": "/content/emergency_sells.json",
    
    # Debug
    "DEBUG_MODE": False,
    
    # Batch Processing
    "BATCH_SIZE": 15,
    
    # Advanced
    "ORDER_FLOW_ENABLE_FOR_ALIGNMENT_2_IF_STRONG_SCORE": True,
    "ORDER_FLOW_PRECHECK_MIN_SCORE": 75.0,
    
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
    "RECONCILIATION_INTERVAL_SEC": 300,  # كل 5 دقائق
}

# ===================== TELEGRAM ENV AUTO-LOAD =====================
def load_telegram_from_env():
    token = (
        os.getenv("TELEGRAM_BOT_TOKEN")
        or os.getenv("TG_TOKEN")
        or os.getenv("TELEGRAM_TOKEN")
        or os.getenv("BOT_TOKEN")
    )
    chat = (
        os.getenv("TELEGRAM_CHAT_ID")
        or os.getenv("TG_CHAT")
        or os.getenv("TELEGRAM_CHAT")
        or os.getenv("CHAT_ID")
    )

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

# ===================== GLOBAL STATE =====================
HTTP_SESSION: Optional[aiohttp.ClientSession] = None
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

def price_acceptance_gate_5m(
    df_5m: pd.DataFrame,
    ob: Optional[Dict[str, Any]],
    lg: Optional['LiquidityGrab']
) -> Tuple[bool, str]:
    
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

        # التحقق من صحة slice الحجم
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
    errors = []

    if CONFIG.get("ENABLE_LIVE_TRADING"):
        if not CONFIG.get("MEXC_API_KEY"):
            errors.append("❌ MEXC_API_KEY مفقود (LIVE TRADING ON)")
        if not CONFIG.get("MEXC_API_SECRET"):
            errors.append("❌ MEXC_API_SECRET مفقود (LIVE TRADING ON)")

    if CONFIG.get("TP1_RR", 0) >= CONFIG.get("TP2_RR", 0):
        errors.append("❌ TP2_RR يجب أن يكون أكبر من TP1_RR")
    if CONFIG.get("TP2_RR", 0) >= CONFIG.get("TP3_RR", 0):
        errors.append("❌ TP3_RR يجب أن يكون أكبر من TP2_RR")

    if CONFIG.get("RISK_PER_TRADE_PCT", 0) > 5:
        errors.append("⚠️ المخاطرة عالية جداً (>5%)")

    if not CONFIG.get("TG_TOKEN") or not CONFIG.get("TG_CHAT"):
        CONFIG.setdefault("_TG_WARNED_IN_VALIDATE", False)
        if not CONFIG["_TG_WARNED_IN_VALIDATE"]:
            logger.warning("⚠️ Telegram غير مكوّن بالكامل - سيتم تشغيل الوضع الصامت تلقائياً إذا لزم")
            CONFIG["_TG_WARNED_IN_VALIDATE"] = True

    if CONFIG.get("LIVE_RECALIBRATION_MODE") not in ("rr",):
        errors.append("❌ LIVE_RECALIBRATION_MODE غير صحيح (اختر: rr)")

    seen_keys = set()
    duplicate_keys = []
    for key in CONFIG.keys():
        if key in seen_keys:
            duplicate_keys.append(key)
        else:
            seen_keys.add(key)
    
    if duplicate_keys:
        errors.append(f"❌ مفاتيح مكررة في CONFIG: {duplicate_keys}")

    if errors:
        for err in errors:
            logger.error(err)
        raise ValueError("Configuration validation failed")

    logger.info("✅ Configuration validated")

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
        async with self.lock:
            now = time.time()

            while self.request_times and now - self.request_times[0] > 60:
                self.request_times.popleft()
            
            while self.weights and now - self.weights[0][0] > 60:
                _, w = self.weights.popleft()
                self.total_weight -= w
            
            while self.sec_times and now - self.sec_times[0] > 1:
                self.sec_times.popleft()

            if self.consecutive_errors > 0 and CONFIG.get("EXPONENTIAL_BACKOFF", True):
                backoff_time = min(2 ** self.consecutive_errors, 32)
                await asyncio.sleep(backoff_time)
                now = time.time()
                while self.request_times and now - self.request_times[0] > 60:
                    self.request_times.popleft()
                while self.weights and now - self.weights[0][0] > 60:
                    _, w = self.weights.popleft()
                    self.total_weight -= w
                while self.sec_times and now - self.sec_times[0] > 1:
                    self.sec_times.popleft()

            if len(self.sec_times) >= self.max_per_second:
                wait_s = 1 - (now - self.sec_times[0]) + 0.05
                if wait_s > 0:
                    await asyncio.sleep(wait_s)
                    now = time.time()
                while self.sec_times and now - self.sec_times[0] > 1:
                    self.sec_times.popleft()

            if len(self.request_times) >= self.max_per_minute:
                oldest = self.request_times[0]
                wait_time = 60 - (now - oldest) + 0.1
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    now = time.time()
                while self.request_times and now - self.request_times[0] > 60:
                    self.request_times.popleft()

            if self.total_weight + weight > self.max_weight_per_minute:
                if self.weights:
                    oldest_time = self.weights[0][0]
                    wait_time = 60 - (now - oldest_time) + 0.5
                    if wait_time > 0:
                        logger.warning(f"⏸️ Rate limit weight: waiting {wait_time:.1f}s")
                        await asyncio.sleep(wait_time)
                        now = time.time()
                    while self.weights and now - self.weights[0][0] > 60:
                        _, w = self.weights.popleft()
                        self.total_weight -= w

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

# ===================== ENHANCED SMART CACHE =====================
class SmartCache:
    def __init__(self):
        self.cache: Dict[str, Tuple[float, Any, int]] = {}
        self.lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
        self.max_size = 1000
    
    def _get_ttl(self, timeframe: str) -> int:
        ttl_map = {"1m": 15, "5m": 30, "15m": 60, "1h": 180, "4h": 600}
        return ttl_map.get(timeframe, 30)
    
    async def smart_cache_cleanup(self):
        async with self.lock:
            now = time.time()
            self.cache = {
                k: v for k, v in self.cache.items()
                if now - v[0] <= 600
            }
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
        cache_key = f"{symbol}{timeframe}{limit}"
        ttl = self._get_ttl(timeframe)
        
        async with self.lock:
            if cache_key in self.cache:
                cache_time, data, access_count = self.cache[cache_key]
                if time.time() - cache_time < ttl:
                    self.cache[cache_key] = (cache_time, data, access_count + 1)
                    self.hits += 1
                    return data
                else:
                    del self.cache[cache_key]
        
        self.misses += 1
        
        if CONFIG.get("CIRCUIT_BREAKER_ENABLED", True) and not api_circuit.can_attempt():
            logger.warning(f"[Circuit Breaker] Skipping API call for {symbol} {timeframe}")
            return None
        
        retry = ExponentialBackoffRetry(
            max_retries=CONFIG.get("RETRY_MAX_RETRIES", 3),
            base_delay=CONFIG.get("RETRY_BASE_DELAY", 1),
            max_delay=CONFIG.get("RETRY_MAX_DELAY", 60)
        )
        
        try:
            data = await retry.execute(
                self._fetch_ohlcv_with_rate_limit,
                exchange, symbol, timeframe, limit
            )
            
            async with self.lock:
                if len(self.cache) >= self.max_size * 0.9:
                    logger.warning(f"🧹 Cache near limit ({len(self.cache)}/{self.max_size}) - aggressive cleanup")
                    sorted_items = sorted(
                        self.cache.items(),
                        key=lambda x: (x[1][2], x[1][0])
                    )
                    keep_count = int(self.max_size * 0.7)
                    self.cache = dict(sorted_items[-keep_count:])
                    logger.info(f"✅ Cache reduced to {len(self.cache)} items")
                
                self.cache[cache_key] = (time.time(), data, 1)
            
            rate_limiter.reset_errors()
            api_circuit.record_success()
            return data
            
        except Exception as e:
            STATS["retries_count"] += 1
            rate_limiter.record_error()
            api_circuit.record_failure()
            logger.error(f"[Cache] Failed to fetch {symbol} {timeframe}: {str(e)[:100]}")
            return None
    
    async def _fetch_ohlcv_with_rate_limit(self, exchange, symbol: str, timeframe: str, limit: int):
        await rate_limiter.wait_if_needed(weight=2)
        return await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    async def clear_old_entries(self):
        async with self.lock:
            now = time.time()
            to_delete = [k for k, (t, _, _) in self.cache.items() 
                         if now - t > 600]
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
        url = f"https://api.telegram.org/bot{CONFIG['TG_TOKEN']}/sendMessage"
        
        if len(msg) > 4000:
            msg = msg[:3900] + "\n\n...[تم الاختصار]"
        
        async with session.post(url, json={
            "chat_id": CONFIG["TG_CHAT"],
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
        logger.error(f"[TG Exception] {str(e)[:100]}")
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
        return ((best_ask - best_bid) / best_bid) * 100
    except Exception:
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
        return order
    except Exception as e:
        STATS["live_order_errors"] += 1
        logger.error(f"[LIVE Entry Error] {symbol}: {str(e)[:150]}")
        log_order_audit("LIMIT_BUY", symbol, price, amount, f"ERROR: {str(e)[:50]}")
        return None

async def place_stop_loss_order(exchange, symbol: str, stop_price: float, amount: float) -> Optional[Dict[str, Any]]:
    """
    وضع أمر وقف خسارة على المنصة (stop-loss أو stop-limit حسب الإعدادات).
    يدعم MEXC spot: نوع الأمر 'stop-loss' مع stopPrice.
    """
    if not CONFIG.get("LIVE_PLACE_SL_ORDER"):
        return None
    
    order_type = CONFIG.get("LIVE_SL_ORDER_TYPE", "stop-loss")
    try:
        amt = _round_amount_to_precision(exchange, symbol, amount)
        # سعر التنفيذ لوقف الخسارة: عادةً يكون سعر السوق (لstop-loss) أو يمكن تحديد سعر حد (لstop-limit)
        # هنا نستخدم stop-loss بسيط مع stopPrice
        params = {'stopPrice': _round_price_to_precision(exchange, symbol, stop_price)}
        await rate_limiter.wait_if_needed(weight=1)
        order = await exchange.create_order(symbol, order_type, "sell", amt, None, params)
        log_order_audit("STOP_LOSS", symbol, stop_price, amt, "PLACED")
        return order
    except Exception as e:
        logger.error(f"[LIVE SL Order Error] {symbol}: {str(e)[:150]}")
        log_order_audit("STOP_LOSS", symbol, stop_price, amount, f"ERROR: {str(e)[:50]}")
        return None

async def cancel_stop_loss_order(exchange, symbol: str, order_id: str) -> bool:
    try:
        await rate_limiter.wait_if_needed(weight=1)
        await exchange.cancel_order(order_id, symbol)
        log_order_audit("CANCEL_SL", symbol, 0, 0, "CANCELLED")
        return True
    except Exception as e:
        logger.error(f"[LIVE Cancel SL Error] {symbol}: {str(e)[:150]}")
        return False

async def wait_for_order_fill_or_cancel(exchange, symbol: str, order_id: str, timeout_sec: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
    start = time.time()
    last = None
    poll = max(1, int(CONFIG.get("ENTRY_LIMIT_POLL_SEC", 3)))
    
    while time.time() - start < timeout_sec:
        try:
            await rate_limiter.wait_if_needed(weight=1)
            last = await exchange.fetch_order(order_id, symbol)
            
            status = last.get("status")
            filled = safe_float(last.get("filled"), 0.0)
            amount = safe_float(last.get("amount"), 0.0)
            
            if status == "closed" and amount > 0 and filled >= amount * 0.999:
                STATS["live_orders_filled"] += 1
                log_order_audit("LIMIT_BUY", symbol, last.get('price', 0), filled, "FILLED")
                return True, last
            
            if status in ("canceled", "rejected", "expired"):
                log_order_audit("LIMIT_BUY", symbol, last.get('price', 0), filled, status)
                return False, last
        
        except Exception:
            pass
        
        await asyncio.sleep(poll)
    
    try:
        await rate_limiter.wait_if_needed(weight=1)
        await exchange.cancel_order(order_id, symbol)
        STATS["live_orders_canceled"] += 1
        try:
            await rate_limiter.wait_if_needed(weight=1)
            last = await exchange.fetch_order(order_id, symbol)
        except Exception:
            pass
        log_order_audit("LIMIT_BUY", symbol, 0, 0, "TIMEOUT_CANCELED")
    except Exception as e:
        logger.warning(f"[LIVE Cancel Warn] {symbol}: {str(e)[:120]}")
    
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
        return order
    except Exception as e:
        STATS["live_order_errors"] += 1
        logger.error(f"[LIVE Sell Error] {symbol}: {str(e)[:150]}")
        log_order_audit("MARKET_SELL", symbol, 0, amount, f"ERROR: {str(e)[:50]}")
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
        if atr_pct > safe_float(CONFIG.get("ENTRY_QUALITY_MAX_ATR_PCT_5M", 6.5), 6.5):
            return False, f"atr_pct_too_high({atr_pct:.2f}%)"
        
        bb_width = safe_float(last.get("bb_width", 0.0), 0.0)
        if bb_width > safe_float(CONFIG.get("ENTRY_QUALITY_MAX_BB_WIDTH_5M", 0.08), 0.08):
            return False, f"bb_width_too_high({bb_width:.3f})"
        
        price_change_2_candles = (last['close'] - prev['close']) / prev['close']
        if abs(price_change_2_candles) > 0.02:
            return False, "momentum_too_fast"
        
        if 'rsi' in df_5m.columns:
            rsi = safe_float(last['rsi'])
            if rsi > 70:
                return False, "rsi_overbought"
        
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
def recalibrate_levels_on_fill(
    fill_price: float,
    signal: 'QuantumSignal'
) -> Tuple[float, float, float, float, float]:
    if not validate_price(fill_price):
        return signal.sl, signal.tp1, signal.tp2, signal.tp3, signal.position_size_usdt
    
    mode = CONFIG.get("LIVE_RECALIBRATION_MODE", "rr")
    
    orig_entry = safe_float(signal.entry, 0.0)
    orig_sl = safe_float(signal.sl, 0.0)
    orig_risk = orig_entry - orig_sl
    
    if orig_risk <= 0:
        return signal.sl, signal.tp1, signal.tp2, signal.tp3, signal.position_size_usdt
    
    if mode == "rr":
        tp1_rr = (safe_float(signal.tp1) - orig_entry) / orig_risk if orig_risk > 0 else CONFIG["TP1_RR"]
        tp2_rr = (safe_float(signal.tp2) - orig_entry) / orig_risk if orig_risk > 0 else CONFIG["TP2_RR"]
        tp3_rr = (safe_float(signal.tp3) - orig_entry) / orig_risk if orig_risk > 0 else CONFIG["TP3_RR"]
        
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
        new_tp3 = fill_price + (orig_risk * tp3_rr)
        
        new_position_value, _ = calculate_position_size(fill_price, new_sl)
        if new_position_value < signal.position_size_usdt:
            logger.info(f"[Recalibrate] Position size reduced from {signal.position_size_usdt:.2f} to {new_position_value:.2f}")
            return new_sl, new_tp1, new_tp2, new_tp3, new_position_value
        
        return new_sl, new_tp1, new_tp2, new_tp3, signal.position_size_usdt
    
    return signal.sl, signal.tp1, signal.tp2, signal.tp3, signal.position_size_usdt

# ===================== ENHANCED DATABASE MANAGER =====================
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
                    sl_order_id TEXT
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
                    INSERT OR REPLACE INTO daily_state 
                    (id, date, realized_r, blocked, last_updated)
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
                    INSERT OR REPLACE INTO active_trades 
                    (symbol, entry, original_sl, current_sl, tp1, tp2, tp3, atr,
                     tp1_hit, tp2_hit, tp3_hit, remaining_position, be_moved,
                     trailing_active, entry_time, last_update, total_realized_r, signal_data,
                     emergency_state, emergency_reason, emergency_last_attempt, emergency_attempts, version,
                     sl_order_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_state.symbol,
                    trade_state.entry,
                    trade_state.original_sl,
                    trade_state.current_sl,
                    trade_state.tp1,
                    trade_state.tp2,
                    trade_state.tp3,
                    trade_state.atr,
                    int(trade_state.tp1_hit),
                    int(trade_state.tp2_hit),
                    int(trade_state.tp3_hit),
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
                    trade_state.sl_order_id
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
    
    def record_trade_history(self, symbol: str, trade_state: TradeState, exit_price: float, 
                           exit_type: str, r_multiple: float, execution_mode: str = ""):
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
                    INSERT INTO trade_history 
                    (symbol, entry, exit, exit_type, profit_pct, r_multiple, 
                     quantum_score, signal_class, gates_passed, entry_time, exit_time, execution_mode)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    trade_state.entry,
                    exit_price,
                    exit_type,
                    profit_pct,
                    r_multiple,  # هنا ندخل total_r (الإجمالي للصفقة) وليس فقط الجزء الأخير
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
                        tp3=row['tp3'],
                        atr=row['atr'],
                        tp1_hit=bool(row['tp1_hit']),
                        tp2_hit=bool(row['tp2_hit']),
                        tp3_hit=bool(row['tp3_hit']),
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
                        sl_order_id=row['sl_order_id'] if 'sl_order_id' in row.keys() else ""
                    )
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
                    SELECT symbol, entry, exit, exit_type, profit_pct, r_multiple, 
                           quantum_score, signal_class, gates_passed, entry_time, exit_time, execution_mode
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

db_manager = DatabaseManager(CONFIG["DB_PATH"])

# ===================== DAILY CIRCUIT BREAKER =====================
class DailyCircuitBreaker:
    def __init__(self):
        self.date = None
        self.realized_r = 0.0
        self.blocked = False
        self.load_state()
    
    def load_state(self):
        if not CONFIG["ENABLE_DB_PERSISTENCE"]:
            return
        
        db_date, db_realized_r, db_blocked = db_manager.load_daily_state()
        if db_date:
            self.date = db_date
            self.realized_r = db_realized_r
            self.blocked = db_blocked
            logger.info(f"[Daily Circuit] Loaded state: date={self.date}, R={self.realized_r:.2f}, blocked={self.blocked}")
    
    def save_state(self):
        if not CONFIG["ENABLE_DB_PERSISTENCE"]:
            return
        
        db_manager.save_daily_state(self.date or "", self.realized_r, self.blocked)
    
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

daily_circuit = DailyCircuitBreaker()

def record_daily_r(r_value: float):
    daily_circuit.record_daily_r(r_value)

def is_daily_loss_blocked() -> bool:
    return daily_circuit.is_blocked()

# ===================== CHECKPOINT SYSTEM =====================
async def checkpoint_saver():
    if not CONFIG.get("ENABLE_CHECKPOINTS", True):
        return
    
    interval = int(CONFIG.get("CHECKPOINT_INTERVAL_SEC", 300))
    path = CONFIG.get("CHECKPOINT_PATH", "/content/quantum_checkpoint.pkl")
    
    import pickle
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
            
            with open(path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            logger.info(f"💾 Checkpoint saved - {len(active_trades_copy)} trades")
        
        except Exception as e:
            logger.error(f"Checkpoint save error: {e}")
            await asyncio.sleep(60)

def load_checkpoint() -> bool:
    if not CONFIG.get("ENABLE_CHECKPOINTS", True):
        return False
    
    path = CONFIG.get("CHECKPOINT_PATH", "/content/quantum_checkpoint.pkl")
    
    import pickle
    try:
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        logger.info(f"♻️ Checkpoint loaded from {checkpoint.get('timestamp')}")
        
        STATS.update(checkpoint.get('stats', {}))
        
        recovered = 0
        at = checkpoint.get('active_trades', {}) or {}
        for symbol, trade_data in at.items():
            try:
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

def save_emergency_checkpoint(err: Exception):
    try:
        active_trades_copy = {k: asdict(v) for k, v in ACTIVE_TRADES.items()}
        
        checkpoint = {
            'timestamp': now_utc_iso(),
            'active_trades': active_trades_copy,
            'stats': STATS.copy(),
            'error': str(err)
        }
        
        path = CONFIG.get("EMERGENCY_CHECKPOINT_PATH", "/content/quantum_emergency_checkpoint.pkl")
        with open(path, 'wb') as f:
            import pickle
            pickle.dump(checkpoint, f)
        
        logger.info("💾 Emergency checkpoint saved (sync)")
    except Exception as e:
        logger.error(f"Emergency checkpoint failed: {e}")

# ===================== INDICATORS =====================
def calculate_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) < 20:
        return None
    
    try:
        df = df.copy()
        df = df[df['close'] > 0].copy()
        if len(df) < 20:
            return None
        
        if TALIB_AVAILABLE:
            TALIB_FALLBACK = False
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
            else:
                TALIB_FALLBACK = False
        else:
            TALIB_FALLBACK = True
        
        if TALIB_FALLBACK:
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
        recent_high = safe_float(df['high'].iloc[-CONFIG["BOS_CONFIRMATION_CANDLES"]:].max())
        recent_low = safe_float(df['low'].iloc[-CONFIG["BOS_CONFIRMATION_CANDLES"]:].min())
        recent_close_avg = safe_float(df['close'].iloc[-2:].mean())
        mult = CONFIG["BOS_CONFIRMATION_MULTIPLIER"]
        
        bos_bullish = (
            recent_high > prev_swing_high and
            recent_close_avg > prev_swing_high * mult
        )
        
        bos_bearish = (
            recent_low < prev_swing_low and
            recent_close_avg < prev_swing_low * mult
        )
        
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
            for i in range(len(df) - lookback, len(df)):
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
                
                is_fresh = (len(df) - 1 - i) < 20
                
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
            for i in range(1, len(df) - 1):
                prev_candle = df.iloc[i-1]
                current_candle = df.iloc[i]
                # next_candle غير مستخدم حالياً
                if prev_candle['high'] < current_candle['low']:  # فجوة صاعدة
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
                elif prev_candle['low'] > current_candle['high']:  # فجوة هابطة
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
        
        # استبعاد آخر شمعة من حساب السيولة
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
@log_execution_time
async def analyze_order_flow(exchange, symbol: str, mtf_alignment: int = 0, 
                           allow_low_alignment: bool = False) -> Optional[OrderFlowData]:
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
        await metrics.record_error("order_flow_analysis", type(e).__name__)
        logger.error(f"[Order Flow Error] {symbol}: {str(e)[:100]}")
        return None

# ===================== VOLUME PROFILE ANALYSIS (مُحسّن) =====================
def analyze_volume_profile(df: pd.DataFrame, precheck_score: float = 0.0) -> Optional[VolumeProfileData]:
    if not CONFIG.get("ENABLE_VOLUME_PROFILE", False) or df is None or len(df) < 50:
        return None
    
    # التحقق من أخذ عينات الحجم
    if not CONFIG.get("_VOLUME_PROFILE_SAMPLING_OK", True):
        return None
    
    try:
        if len(df) > 100:
            df = df.iloc[-100:]
        
        if precheck_score < 50:
            return None
        
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
        
        hvn_levels = [safe_float(bin_edges[i]) for i in range(len(hist))
                     if hist[i] > hvn_threshold]
        lvn_levels = [safe_float(bin_edges[i]) for i in range(len(hist))
                     if hist[i] < lvn_threshold]
        
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

# ===================== LIQUIDITY GRAB DETECTION (مُحسّن) =====================
def detect_liquidity_grab(df: pd.DataFrame) -> Optional[LiquidityGrab]:
    if not CONFIG.get("ENABLE_LIQUIDITY_GRAB", True) or df is None or len(df) < 30:
        return None
    
    try:
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2] if len(df) > 1 else last_candle
        
        # استبعاد آخر شمعة من حساب الدعم والمقاومة
        window = df.iloc[-21:-1] if len(df) >= 21 else df.iloc[:-1]
        if len(window) == 0:
            return None
        
        support = safe_float(window['low'].min())
        resistance = safe_float(window['high'].max())
        current_atr = safe_float(df['atr'].iloc[-1]) if 'atr' in df.columns else 0
        
        equal_lows = False
        equal_lows_range = 0.0
        if 'low' in df.columns:
            recent_lows = df['low'].iloc[-10:].tolist()
            if len(recent_lows) >= 5:
                sorted_lows = sorted(recent_lows[-5:])
                max_low = max(sorted_lows)
                min_low = min(sorted_lows)
                
                atr_range_threshold = current_atr * CONFIG["LG_EQUAL_LOWS_RANGE_ATR_MULT"]
                range_condition = (max_low - min_low) < atr_range_threshold
                equal_lows = range_condition and len(recent_lows) >= CONFIG["LG_EQUAL_LOWS_REQUIRED"]
                equal_lows_range = max_low - min_low
        
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
        
        sweep_candle_close_in_range = (
            last_candle['close'] > support and
            last_candle['close'] < resistance
        )
        
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
                wick_ratio * 100 +
                recovery * 50 +
                (last_candle['volume'] / avg_volume / CONFIG["LG_VOLUME_MULTIPLIER"]) * 20,
                100
            )
            
            if equal_lows and len(df) >= CONFIG["LG_EQUAL_LOWS_REQUIRED"] and range_condition:
                confidence = min(confidence * 1.2, 100)
            
            if confidence < 60:
                return None
            
            sweep_idx_in_recent = len(df) - 1
            sweep_timestamp = int(df['t'].iloc[-1]) if 't' in df.columns else None
            
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
                upper_wick_ratio * 100 +
                (last_candle['volume'] / avg_volume / CONFIG["LG_VOLUME_MULTIPLIER"]) * 20,
                100
            )
            
            if confidence < 60:
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

# ===================== MULTI TIMEFRAME ANALYSIS =====================
@metrics.record_latency("mtf_analysis")
@log_execution_time
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
        
        # نؤجل تحليل Volume Profile لحين توفر precheck_score
        liquidity_grab = detect_liquidity_grab(df_5m)
        
        return {
            'structure_1h': structure_1h,
            'structure_15m': structure_15m,
            'structure_5m': structure_5m,
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
    
    gate1_passed = (
        market_structure.trend_strength >= CONFIG["HARD_GATE_1_MIN_TREND_STRENGTH"] and
        mtf_alignment >= CONFIG["HARD_GATE_1_MIN_MTF_ALIGNMENT"] and
        market_structure.structure == "BULLISH"
    )
    
    if gate1_passed:
        gates_passed.append("GATE_1_TREND")
    else:
        all_gates_passed = False
        if CONFIG["DEBUG_MODE"]:
            logger.info(f"[Hard Gates] GATE_1 failed: Trend={market_structure.trend_strength:.1f}, "
                      f"MTF={mtf_alignment}, Structure={market_structure.structure}")
    
    if CONFIG["ENABLE_HARD_GATES"] and CONFIG["HARD_GATE_2_REQUIRE_ZONE"]:
        has_strong_lg = (
            liquidity_grab and 
            liquidity_grab.detected and 
            liquidity_grab.confidence >= CONFIG.get("HARD_GATE_2_MIN_LG_CONFIDENCE", 75) and 
            liquidity_grab.grab_type == "BULLISH"
        )
        
        has_fresh_ob = (
            market_structure.order_block and
            market_structure.order_block.get("freshness", 999) <= CONFIG.get("HARD_GATE_2_OB_FRESHNESS", 10) and
            market_structure.trend_strength >= 65
        )
        
        gate2_passed = has_strong_lg or has_fresh_ob
        
        if gate2_passed:
            if has_strong_lg:
                gates_passed.append("GATE_2_STRONG_LIQUIDITY_GRAB")
            if has_fresh_ob:
                gates_passed.append("GATE_2_FRESH_ORDER_BLOCK")
        else:
            all_gates_passed = False
            if CONFIG["DEBUG_MODE"]:
                logger.info(f"[Hard Gates] GATE_2 failed: No strong Liquidity Grab or fresh Order Block")
    
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
    df: pd.DataFrame
) -> Tuple[float, float, str]:
    
    gates_ok, gates_list = evaluate_hard_gates(
        market_structure, order_flow, volume_profile,
        liquidity_grab, mtf_alignment, df
    )
    
    if CONFIG["ENABLE_HARD_GATES"] and not gates_ok:
        return 0.0, 0.0, "REJECT"
    
    score = 0.0
    confidence_factors = []
    
    if market_structure.structure == "BULLISH":
        score += 15
        confidence_factors.append(60)
    
    if market_structure.bos_bullish:
        score += 15
        confidence_factors.append(80)
    
    if market_structure.trend_strength > 70:
        score += 10
        confidence_factors.append(market_structure.trend_strength)
    elif market_structure.trend_strength > 55:
        score += 5
        confidence_factors.append(market_structure.trend_strength)
    
    if market_structure.order_block:
        score += 10
        confidence_factors.append(70)
        freshness = market_structure.order_block.get('freshness', 100)
        if freshness < 5:
            score += 5
            confidence_factors.append(90)
    
    if liquidity_grab and liquidity_grab.detected and liquidity_grab.grab_type == "BULLISH":
        score += 20
        confidence_factors.append(liquidity_grab.confidence)
        
        if liquidity_grab.wick_strength > 0.75:
            score += 5
            confidence_factors.append(85)
        
        if liquidity_grab.volume_spike > 3.0:
            score += 5
            confidence_factors.append(80)
        
        if hasattr(liquidity_grab, 'equal_lows') and liquidity_grab.equal_lows:
            score += 5
            confidence_factors.append(90)
    
    if order_flow:
        if order_flow.signal == "BULLISH":
            score += 4
            confidence_factors.append(order_flow.confidence)
        elif order_flow.signal == "NEUTRAL":
            score += 2
            confidence_factors.append(50)
        
        if order_flow.volume_profile == "AGGRESSIVE_BUYING":
            score += 4
            confidence_factors.append(75)
        elif order_flow.volume_profile == "DISTRIBUTION":
            score -= 3
            confidence_factors.append(30)
        
        if order_flow.imbalance > 0.3:
            score += 3
            confidence_factors.append(65)
    
    if volume_profile:
        if volume_profile.current_position == "BELOW_VALUE":
            score += 8
            confidence_factors.append(75)
        elif volume_profile.current_position == "IN_VALUE":
            score += 5
            confidence_factors.append(60)
    
    score += mtf_alignment * 5
    if mtf_alignment == 3:
        confidence_factors.append(95)
    elif mtf_alignment == 2:
        confidence_factors.append(75)
    
    if 'rsi' in df.columns:
        rsi = safe_float(df['rsi'].iloc[-1])
        if 30 < rsi < 70:
            score += 5
            confidence_factors.append(65)
    
    quantum_score = max(0.0, min(100.0, score))
    quantum_score = min(100, quantum_score * 0.85)
    
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

# ===================== دالة حساب مستويات وقف الخسارة والأهداف =====================
def compute_sl_and_tp_from_structure(entry: float, market_structure: MarketStructure, atr: float, liquidity_grab: Optional[LiquidityGrab]) -> Tuple[float, float, float, float]:
    """
    حساب وقف الخسارة النهائي بناءً على كتلة الطلبات أو ATR، مع تطبيق الحد الأقصى لنسبة الخسارة.
    """
    if not validate_price(entry) or entry <= 0:
        return 0.0, 0.0, 0.0, 0.0
    
    if not validate_price(atr) or atr <= 0:
        atr = entry * 0.02
    
    # محاولة استخدام كتلة الطلبات أولاً
    sl = 0.0
    if market_structure.order_block:
        ob_sl = market_structure.order_block['low'] * 0.995
        if 0 < ob_sl < entry:
            sl = ob_sl
    
    # إذا لم نجد كتلة طلبات صالحة، نستخدم ATR
    if sl == 0:
        sl = entry - (atr * CONFIG["ATR_SL_MULT"])
        if sl >= entry or sl <= 0:
            sl = entry * 0.95  # fallback
    
    # تطبيق الحد الأقصى لنسبة الخسارة
    max_sl_distance = entry * (CONFIG["MAX_SL_PCT"] / 100)
    hard_sl = entry - max_sl_distance
    if sl < hard_sl:
        logger.warning(f"[SL] SL {sl:.6f} exceeds MAX_SL_PCT, clamping to {hard_sl:.6f}")
        sl = hard_sl
    
    # حساب الأهداف بناءً على R:R
    risk = entry - sl
    if risk <= 0:
        logger.error(f"[SL] Invalid risk: {risk:.6f}")
        return 0.0, 0.0, 0.0, 0.0
    
    tp1 = entry + (risk * CONFIG["TP1_RR"])
    tp2 = entry + (risk * CONFIG["TP2_RR"])
    tp3 = entry + (risk * CONFIG["TP3_RR"])
    
    return sl, tp1, tp2, tp3

# ===================== POSITION SIZING =====================
def calculate_position_size(entry: float, sl: float, account_size: Optional[float] = None) -> Tuple[float, float]:
    if account_size is None:
        account_size = CONFIG["ACCOUNT_SIZE_USDT"]
    
    if not validate_price(entry) or not validate_price(sl) or sl >= entry:
        return 0.0, 0.0
    
    risk_amount = account_size * (CONFIG["RISK_PER_TRADE_PCT"] / 100)
    
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

# ===================== BTC TREND CHECK =====================
async def check_btc_trend(exchange) -> Dict:
    global BTC_TREND, BTC_LAST_CHECK
    
    if not CONFIG["ENABLE_BTC_FILTER"]:
        return {"trend": "NEUTRAL", "change_1h": 0, "safe_to_trade": True}
    
    if BTC_TREND and (time.time() - BTC_LAST_CHECK) < 300:
        return BTC_TREND
    
    try:
        await rate_limiter.wait_if_needed(weight=2)
        data = await cache.get_ohlcv(exchange, "BTC/USDT", "1h", 100)
        
        if not data or len(data) < 20:
            return {"trend": "NEUTRAL", "change_1h": 0, "safe_to_trade": True}
        
        df = pd.DataFrame(data, columns=['t','open','high','low','close','volume'])
        current_price = safe_float(df['close'].iloc[-1])
        
        price_1h_ago = safe_float(df['close'].iloc[-2]) if len(df) >= 2 else current_price
        price_4h_ago = safe_float(df['close'].iloc[-5]) if len(df) >= 5 else price_1h_ago
        
        change_1h = ((current_price - price_1h_ago) / price_1h_ago) * 100 if price_1h_ago > 0 else 0
        change_4h = ((current_price - price_4h_ago) / price_4h_ago) * 100 if price_4h_ago > 0 else 0
        
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
        
        BTC_TREND = {
            "trend": trend,
            "change_1h": round(change_1h, 2),
            "change_4h": round(change_4h, 2),
            "safe_to_trade": safe_to_trade,
            "price": current_price
        }
        
        BTC_LAST_CHECK = time.time()
        
        if trend == "CRASH":
            logger.warning(f"[BTC] 🚨 تحذير: انهيار! 1H: {change_1h:.2f}%, 4H: {change_4h:.2f}%")
            await send_telegram(
                f"⚠️ تحذير انهيار BTC\n\n"
                f"📉 التغير خلال ساعة: {change_1h:.2f}%\n"
                f"📉 التغير خلال 4 ساعات: {change_4h:.2f}%\n\n"
                f"🛑 تم إيقاف التداول مؤقتاً!",
                critical=True
            )
        
        return BTC_TREND
    
    except Exception as e:
        logger.error(f"[BTC Check Error] {str(e)[:100]}")
        return {"trend": "NEUTRAL", "change_1h": 0, "safe_to_trade": True}

# ===================== FIXED ORDER FLOW SAMPLING LOGIC (مع staggered per symbol) =====================
def should_run_order_flow(symbol: str, mtf_alignment: int, precheck_score: float, loop_count: int) -> bool:
    if not CONFIG.get("ORDER_FLOW_ENABLED", True):
        return False
    
    if mtf_alignment >= 3:
        return True
    
    if mtf_alignment == 2 and CONFIG.get("ORDER_FLOW_ENABLE_FOR_ALIGNMENT_2_IF_STRONG_SCORE", True):
        if precheck_score >= safe_float(CONFIG.get("ORDER_FLOW_PRECHECK_MIN_SCORE", 75.0), 75.0):
            return True
    
    if mtf_alignment >= 2 and CONFIG.get("ORDER_FLOW_SAMPLING_ENABLED", True):
        # توزيع عشوائي حسب الرمز باستخدام دالة تجزئة ثابتة
        sample_every = CONFIG.get("ORDER_FLOW_SAMPLE_EVERY_N_LOOPS", 3)
        symbol_hash = stable_hash(symbol) % sample_every
        if symbol_hash == loop_count % sample_every:
            return True
    
    return False

# ===================== ENHANCED EMERGENCY STATE MONITOR =====================
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
                                    db_manager.save_trade(ACTIVE_TRADES[symbol])
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
                    current_price = safe_float(ticker.get('last'))
                    
                    if not validate_price(current_price):
                        continue
                    
                    remaining_base = trade_snapshot.entry_fill_amount * trade_snapshot.remaining_position
                    
                    sell_order = await market_sell_safe(exchange, symbol, remaining_base, max_retries=2)
                    
                    if sell_order:
                        # استخدام سعر التنفيذ الفعلي من الأمر
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
                                    db_manager.save_trade(ACTIVE_TRADES[symbol])
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

# ===================== BALANCE RECONCILIATION TASK =====================
async def reconcile_balances(exchange):
    """
    مهمة دورية لمقارنة الأرصدة الفعلية على المنصة مع المراكز المتوقعة.
    """
    while not shutdown_manager.should_stop:
        await asyncio.sleep(CONFIG.get("RECONCILIATION_INTERVAL_SEC", 300))
        
        if not is_live_trading_enabled():
            continue
        
        if not ACTIVE_TRADES:
            continue
        
        try:
            # جلب الرصيد الكامل
            balance = await exchange.fetch_balance()
            if not balance:
                continue
            
            for symbol, trade in list(ACTIVE_TRADES.items()):
                try:
                    base_asset = symbol.split('/')[0]
                    real_balance = balance['free'].get(base_asset, 0.0)
                    
                    expected = trade.entry_fill_amount * trade.remaining_position
                    
                    if abs(real_balance - expected) > CONFIG["MIN_DUST_THRESHOLD"]:
                        logger.warning(f"[Reconciliation] Mismatch for {symbol}: expected {expected:.8f}, real {real_balance:.8f}")
                        await send_telegram(
                            f"⚠️ تنبيه المطابقة\n\n"
                            f"الرمز: {escape_html(symbol)}\n"
                            f"الرصيد المتوقع: {expected:.8f} {base_asset}\n"
                            f"الرصيد الفعلي: {real_balance:.8f} {base_asset}\n"
                            f"الفرق: {abs(real_balance - expected):.8f}\n\n"
                            f"قد يكون هناك تنفيذ غير متوقع أو خطأ في الحسابات الداخلية.",
                            critical=False
                        )
                        
                        # يمكن إضافة منطق إيقاف التداول لهذا الرمز إذا كان الفرق كبيراً
                        if abs(real_balance - expected) > expected * 0.1:  # فرق 10%
                            await mark_trade_emergency(
                                symbol,
                                reason=f"Reconciliation mismatch: expected {expected}, real {real_balance}",
                                critical_msg=f"🚨 Reconciliation Critical: {symbol}\nفرق كبير بين الرصيد الفعلي والمتوقع. يرجى المراجعة الفورية."
                            )
                except Exception as e:
                    logger.error(f"[Reconciliation] Error processing {symbol}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"[Reconciliation] Main error: {e}")

# ===================== INSTITUTIONAL SIGNAL GENERATOR (مُعاد هيكلته) =====================
@metrics.record_latency("signal_generation")
@log_execution_time
async def generate_quantum_signal(exchange, symbol: str) -> Optional[QuantumSignal]:
    try:
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
        
        mtf = await analyze_multi_timeframe(exchange, symbol)
        
        if not mtf or mtf['signal'] != "ENTER":
            return None
        
        structure_1h = mtf['structure_1h']
        structure_15m = mtf['structure_15m']
        structure_5m = mtf['structure_5m']
        liquidity_grab = mtf['liquidity_grab']
        df_15m = mtf['df_15m']
        df_5m = mtf['df_5m']
        
        if CONFIG["LONG_ONLY"] and structure_1h.structure != "BULLISH":
            return None
        
        ob = structure_15m.order_block if (structure_15m and structure_15m.order_block) else None
        lg = liquidity_grab if (liquidity_grab and liquidity_grab.detected and liquidity_grab.grab_type == "BULLISH") else None
        
        ok_accept, reason = price_acceptance_gate_5m(df_5m, ob, lg)
        if not ok_accept:
            if CONFIG.get("DEBUG_MODE"):
                logger.info(f"[EntryGate] {symbol} rejected: {reason}")
            return None
        
        # تحديد سعر الدخول
        if ob:
            entry = _ob_entry_price(ob)
        elif lg:
            entry = safe_float(lg.grab_level) * 1.0005
        else:
            return None
        
        if not validate_price(entry):
            return None
        
        atr = safe_float(df_15m['atr'].iloc[-1])
        
        # حساب وقف الخسارة والأهداف أولاً
        sl, tp1, tp2, tp3 = compute_sl_and_tp_from_structure(entry, structure_15m, atr, liquidity_grab)
        if sl == 0:
            return None
        
        # حساب حجم الصفقة بناءً على وقف الخسارة
        position_size_usdt, position_size_pct = calculate_position_size(entry, sl)
        if position_size_usdt == 0:
            return None
        
        # حساب النقاط الأولية (دون order flow) لاستخدامها في sampling و volume profile
        pre_qs, pre_conf, pre_class = calculate_quantum_score(
            structure_15m,
            None,
            None,  # volume profile غير محسوب بعد
            liquidity_grab,
            mtf['alignment'],
            df_15m
        )
        
        if pre_class == "QUANTUM_A+":
            reset_daily_counters()
            max_daily_a_plus = CONFIG.get("MAX_DAILY_A_PLUS", 3)
            if STATS.get("daily_a_plus_count", 0) >= max_daily_a_plus:
                logger.info(f"[Daily Cap] A+ cap reached ({STATS['daily_a_plus_count']}/{max_daily_a_plus}) - rejecting")
                return None
        
        # تشغيل order flow إذا لزم الأمر
        order_flow = None
        if should_run_order_flow(symbol, mtf['alignment'], pre_qs, int(STATS.get("loop_count", 0))):
            if CONFIG.get("ORDER_FLOW_ENABLED", True):
                allow_low = (mtf['alignment'] == 2 and pre_qs >= 75) or (mtf['alignment'] == 1 and pre_qs >= 85)
                order_flow = await analyze_order_flow(exchange, symbol, mtf['alignment'], allow_low_alignment=allow_low)
        
        # تشغيل volume profile الآن باستخدام pre_qs
        volume_profile = None
        if CONFIG.get("ENABLE_VOLUME_PROFILE", False) and CONFIG.get("_VOLUME_PROFILE_SAMPLING_OK", True):
            volume_profile = analyze_volume_profile(df_15m, precheck_score=pre_qs)
        
        # حساب النقاط النهائية مع order flow و volume profile
        quantum_score, confidence, signal_class = calculate_quantum_score(
            structure_15m,
            order_flow,
            volume_profile,
            liquidity_grab,
            mtf['alignment'],
            df_15m
        )
        
        if quantum_score < CONFIG["MIN_QUANTUM_SCORE"]:
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
        
        risk_reward = (tp3 - entry) / (entry - sl) if entry > sl else 0
        win_probability = min(95, max(40, quantum_score * 0.8))
        
        return QuantumSignal(
            symbol=symbol,
            mode="SWING" if risk_reward >= 3 else "SCALP",
            entry=entry,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
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
            gates_passed=evaluate_hard_gates(
                structure_15m, order_flow, volume_profile,
                liquidity_grab, mtf['alignment'], df_15m
            )[1]
        )
    
    except Exception as e:
        logger.error(f"[Signal Generator Error] {symbol}: {str(e)[:200]}")
        return None

# ===================== TELEGRAM FORMATTER =====================
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
    tp3_r = (signal.tp3 - signal.entry) / risk if risk > 0 else 0
    
    live_badge = get_execution_mode_badge()
    
    message = f"""
{emoji} {escape_html(signal.signal_class)} - {escape_html(signal.mode)} - {escape_html(signal.symbol)}  ({live_badge})

🎯 الدخول: {signal.entry:.6f}
🛑 وقف الخسارة: {signal.sl:.6f}

أهداف الربح (خروج جزئي):
✅ الهدف 1 (R:{tp1_r:.1f}) - الخروج 50%: {signal.tp1:.6f}
✅ الهدف 2 (R:{tp2_r:.1f}) - الخروج 30%: {signal.tp2:.6f}
✅ الهدف 3 (R:{tp3_r:.1f}) - الخروج 20%: {signal.tp3:.6f}

💰 حجم الصفقة
━━━━━━━━━━━━━━━━
• الحجم: ${signal.position_size_usdt:.2f}
• % من الحساب: {signal.position_size_pct:.2f}%
• المخاطرة: {CONFIG['RISK_PER_TRADE_PCT']}%

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
        message += f"\n💹 تدفق الأوامر\n"
        message += f"• النمط: {escape_html(signal.order_flow.volume_profile)}\n"
        message += f"• الخلل: {signal.order_flow.imbalance:+.2f}\n"
        message += f"• الدلتا: {signal.order_flow.delta:+.2f}\n"
        message += f"• الإشارة: {escape_html(signal.order_flow.signal)}\n"
        if signal.order_flow.divergence:
            message += f"• ⚠️ اختلاف الأموال الذكية!\n"
    
    if signal.volume_profile:
        message += f"\n📊 توزيع الحجم\n"
        message += f"• الموقع: {escape_html(signal.volume_profile.current_position)}\n"
        message += f"• نقطة التحكم: {signal.volume_profile.poc:.6f}\n"
        message += f"• VWAP: {signal.volume_profile.vwap:.6f}\n"
    
    if signal.liquidity_grab:
        message += f"\n🎯 اصطياد السيولة\n"
        message += f"• النوع: {escape_html(signal.liquidity_grab.grab_type)}\n"
        message += f"• المستوى: {signal.liquidity_grab.grab_level:.6f}\n"
        message += f"• قوة الظل: {signal.liquidity_grab.wick_strength*100:.0f}%\n"
        message += f"• الحجم: {signal.liquidity_grab.volume_spike:.1f}x\n"
        message += f"• الثقة: {signal.liquidity_grab.confidence:.0f}%\n"
        if hasattr(signal.liquidity_grab, 'equal_lows') and signal.liquidity_grab.equal_lows:
            message += f"• ✅ قيعان متساوية: نعم\n"
    
    message += f"\n🔍 <a href=\"{tv_link}\">فتح في TradingView</a>\n"
    message += f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
    
    return message

# ===================== SYMBOL FILTER =====================
async def get_filtered_symbols(exchange) -> List[str]:
    try:
        await rate_limiter.wait_if_needed(weight=CONFIG.get("TICKER_WEIGHT", 2))
        markets = await exchange.load_markets()
        
        tickers = None
        for attempt in range(CONFIG["MAX_RETRIES"]):
            try:
                await rate_limiter.wait_if_needed(weight=CONFIG.get("TICKER_WEIGHT", 2))
                tickers = await exchange.fetch_tickers()
                break
            except Exception as e:
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
            
            if symbol in markets:
                market = markets[symbol]
                if market.get('type') != 'spot':
                    continue
                if not market.get('active', True):
                    continue
            
            volume = safe_float(ticker.get('quoteVolume'), 0)
            price = safe_float(ticker.get('last'), 0)
            
            if (volume < 50000 or  # MIN_VOLUME_USDT
                price > 1000 or  # MAX_PRICE
                not validate_price(price)):
                continue
            
            score = min(volume / 1_000_000, 50)
            if 0 < price < 0.0001:
                score *= 0.5
            
            filtered.append((symbol, score))
        
        filtered.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in filtered[:200]]
    
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
                db_manager.save_trade(trade_snapshot)
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

# ===================== INSTITUTIONAL LIVE ENTRY (مع إدارة SL على المنصة) =====================
@metrics.record_latency("live_entry")
@log_execution_time
async def execute_live_entry_if_enabled(exchange, signal: QuantumSignal) -> Tuple[bool, Optional[TradeState]]:
    if not is_live_trading_enabled():
        return True, None
    
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
    
    adj_sl, adj_tp1, adj_tp2, adj_tp3, adj_position_size = recalibrate_levels_on_fill(fill_price, signal)
    
    trade_state = TradeState(
        symbol=signal.symbol,
        entry=fill_price if validate_price(fill_price) else signal.entry,
        original_sl=adj_sl,
        current_sl=adj_sl,
        tp1=adj_tp1,
        tp2=adj_tp2,
        tp3=adj_tp3,
        atr=signal.atr,
        signal_class=signal.signal_class,
        quantum_score=signal.quantum_score,
        gates_passed=signal.gates_passed,
        entry_order_id=order_id,
        entry_filled=True,
        entry_fill_price=fill_price,
        entry_fill_amount=fill_amount,
        is_paper=False,
        execution_mode="LIVE",
        entry_assumed=False,
    )
    
    # وضع أمر وقف خسارة على المنصة إذا كان مفعلاً
    if CONFIG.get("LIVE_PLACE_SL_ORDER"):
        sl_order = await place_stop_loss_order(exchange, signal.symbol, adj_sl, fill_amount)
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
        f"• TP3: {trade_state.tp3:.6f}\n"
        f"• الكمية (BASE): {trade_state.entry_fill_amount}"
    )
    
    return True, trade_state

# ===================== ENHANCED BATCH PROCESSOR =====================
async def process_symbol_batch(exchange, symbols: List[str]) -> int:
    signals_found = 0
    
    for symbol in symbols:
        try:
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
                
                set_symbol_cooldown(symbol)
                
                signal_dict = asdict(signal)
                signal_dict["execution_mode"] = "LIVE" if is_live_trading_enabled() else ("PAPER" if is_paper_trading_enabled() else "SIGNAL")
                
                message = format_quantum_signal(signal)
                await send_telegram(message)
                
                if is_live_trading_enabled():
                    ok, live_trade_state = await execute_live_entry_if_enabled(exchange, signal)
                    if not ok or not live_trade_state:
                        await asyncio.sleep(0.1)
                        continue
                    
                    if not await bot.get_trade_lock(symbol):
                        continue
                    
                    try:
                        ACTIVE_TRADES[symbol] = live_trade_state
                    finally:
                        bot.release_trade_lock(symbol)
                    
                    db_manager.save_trade(live_trade_state, signal_dict)
                    
                    logger.info(f"[LIVE] ✅ Entry filled - {signal.signal_class} - {symbol} - Score: {signal.quantum_score:.1f}")
                    await asyncio.sleep(0.1)
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
                        tp3=signal.tp3,
                        atr=signal.atr,
                        signal_class=signal.signal_class,
                        quantum_score=signal.quantum_score,
                        gates_passed=signal.gates_passed,
                        is_paper=True,
                        execution_mode="PAPER",
                        entry_assumed=True,
                        entry_filled=True,
                        entry_fill_amount=entry_fill_amount,
                    )
                    
                    if not await bot.get_trade_lock(symbol):
                        continue
                    
                    try:
                        ACTIVE_TRADES[symbol] = trade_state
                    finally:
                        bot.release_trade_lock(symbol)
                    
                    signal_dict["is_paper"] = True
                    signal_dict["entry_assumed"] = True
                    db_manager.save_trade(trade_state, signal_dict)
                    
                    STATS["paper_trades_opened"] += 1
                    logger.info(f"[PAPER] ✅ {signal.signal_class} - {symbol} - Score: {signal.quantum_score:.1f} - Gates: {len(signal.gates_passed)}")
                    
                    await asyncio.sleep(0.1)
                    continue
                
                await asyncio.sleep(0.1)
                continue
            
            await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"[Batch Error] {symbol}: {str(e)[:100]}")
            continue
    
    return signals_found

# ===================== TRADE MONITOR (مع parallel ticker fetch) =====================
async def monitor_active_trades(exchange):
    if not ACTIVE_TRADES:
        return
    
    try:
        # استخدام wait_for لفرض مهلة قصوى على الدورة بأكملها
        await asyncio.wait_for(_monitor_active_trades_internal(exchange), timeout=30)
    except asyncio.TimeoutError:
        logger.error("[Monitor] Timeout exceeded - skipping this cycle")
    except Exception as e:
        logger.error(f"[Monitor Main Error] {str(e)}")

async def _monitor_active_trades_internal(exchange):
    symbols = list(ACTIVE_TRADES.keys())
    
    # جلب جميع tickers بالتوازي
    ticker_tasks = []
    for symbol in symbols:
        if not bot.lock_manager.is_blacklisted(symbol):
            ticker_tasks.append(fetch_ticker_with_lock(exchange, symbol))
    
    ticker_results = await asyncio.gather(*ticker_tasks, return_exceptions=True)
    
    # معالجة كل رمز على حدة
    for symbol, ticker_result in zip(symbols, ticker_results):
        if isinstance(ticker_result, Exception):
            logger.error(f"[Monitor] Error fetching ticker for {symbol}: {ticker_result}")
            continue
        
        ticker = ticker_result
        if ticker is None:
            continue
        
        current_price = safe_float(ticker.get('last'))
        if not validate_price(current_price):
            continue
        
        # الحصول على القفل للمعالجة
        if not await bot.get_trade_lock(symbol):
            continue
        
        try:
            if symbol not in ACTIVE_TRADES:
                continue
            trade = ACTIVE_TRADES[symbol]
        finally:
            bot.release_trade_lock(symbol)
        
        risk = trade.entry - trade.original_sl
        r_multiple = (current_price - trade.entry) / risk if risk > 0 else 0
        
        if is_live_trading_enabled() and exchange is not None:
            if current_price <= trade.current_sl:
                await close_trade_full(symbol, current_price, "SL", exchange=exchange)
                continue
            elif current_price >= trade.tp3 and not trade.tp3_hit:
                await partial_exit(symbol, trade, current_price, "TP3",
                                CONFIG["TP3_EXIT_PCT"], r_multiple, exchange=exchange)
            elif current_price >= trade.tp2 and not trade.tp2_hit:
                await partial_exit(symbol, trade, current_price, "TP2",
                                CONFIG["TP2_EXIT_PCT"], r_multiple, exchange=exchange)
            elif current_price >= trade.tp1 and not trade.tp1_hit:
                await partial_exit(symbol, trade, current_price, "TP1",
                                CONFIG["TP1_EXIT_PCT"], r_multiple, exchange=exchange)
            
            if r_multiple >= CONFIG["BE_AT_R"] and not trade.be_moved:
                if trade.atr > 0:
                    new_sl = trade.entry + (trade.atr * CONFIG["BE_ATR_MULT"])
                else:
                    new_sl = trade.entry + (0.001 * trade.entry)
                
                updates = {
                    'current_sl': new_sl,
                    'be_moved': True,
                    '_version': trade._version + 1
                }
                
                if db_manager.update_trade_with_version(symbol, updates, trade._version):
                    if await bot.get_trade_lock(symbol):
                        try:
                            if symbol in ACTIVE_TRADES:
                                ACTIVE_TRADES[symbol].current_sl = new_sl
                                ACTIVE_TRADES[symbol].be_moved = True
                                ACTIVE_TRADES[symbol]._version += 1
                        finally:
                            bot.release_trade_lock(symbol)
                    
                    logger.info(f"[BE] {symbol} - internal SL moved to breakeven using ATR")
            
            elif r_multiple >= CONFIG["TRAIL_START_R"] and trade.be_moved:
                atr_estimate = trade.atr if trade.atr > 0 else (trade.tp1 - trade.entry) / CONFIG["TP1_RR"]
                new_sl = current_price - (atr_estimate * CONFIG["TRAIL_ATR_MULT"])
                
                if new_sl > trade.current_sl:
                    updates = {
                        'current_sl': new_sl,
                        'trailing_active': True,
                        '_version': trade._version + 1
                    }
                    
                    if db_manager.update_trade_with_version(symbol, updates, trade._version):
                        if await bot.get_trade_lock(symbol):
                            try:
                                if symbol in ACTIVE_TRADES:
                                    ACTIVE_TRADES[symbol].current_sl = new_sl
                                    ACTIVE_TRADES[symbol].trailing_active = True
                                    ACTIVE_TRADES[symbol]._version += 1
                            finally:
                                bot.release_trade_lock(symbol)
                
                continue
        
        if current_price <= trade.current_sl:
            await close_trade_full(symbol, current_price, "SL")
            continue
        elif current_price >= trade.tp3 and not trade.tp3_hit:
            await partial_exit(symbol, trade, current_price, "TP3",
                            CONFIG["TP3_EXIT_PCT"], r_multiple)
        elif current_price >= trade.tp2 and not trade.tp2_hit:
            await partial_exit(symbol, trade, current_price, "TP2",
                            CONFIG["TP2_EXIT_PCT"], r_multiple)
        elif current_price >= trade.tp1 and not trade.tp1_hit:
            await partial_exit(symbol, trade, current_price, "TP1",
                            CONFIG["TP1_EXIT_PCT"], r_multiple)
        
        if r_multiple >= CONFIG["BE_AT_R"] and not trade.be_moved:
            if trade.atr > 0:
                new_sl = trade.entry + (trade.atr * CONFIG["BE_ATR_MULT"])
            else:
                new_sl = trade.entry + (0.001 * trade.entry)
            
            updates = {
                'current_sl': new_sl,
                'be_moved': True,
                '_version': trade._version + 1
            }
            
            if db_manager.update_trade_with_version(symbol, updates, trade._version):
                if await bot.get_trade_lock(symbol):
                    try:
                        if symbol in ACTIVE_TRADES:
                            ACTIVE_TRADES[symbol].current_sl = new_sl
                            ACTIVE_TRADES[symbol].be_moved = True
                            ACTIVE_TRADES[symbol]._version += 1
                    finally:
                        bot.release_trade_lock(symbol)
                
                logger.info(f"[BE] {symbol} - SL moved to breakeven using ATR")
        
        elif r_multiple >= CONFIG["TRAIL_START_R"] and trade.be_moved:
            atr_estimate = trade.atr if trade.atr > 0 else (trade.tp1 - trade.entry) / CONFIG["TP1_RR"]
            new_sl = current_price - (atr_estimate * CONFIG["TRAIL_ATR_MULT"])
            
            if new_sl > trade.current_sl:
                updates = {
                    'current_sl': new_sl,
                    'trailing_active': True,
                    '_version': trade._version + 1
                }
                
                if db_manager.update_trade_with_version(symbol, updates, trade._version):
                    if await bot.get_trade_lock(symbol):
                        try:
                            if symbol in ACTIVE_TRADES:
                                ACTIVE_TRADES[symbol].current_sl = new_sl
                                ACTIVE_TRADES[symbol].trailing_active = True
                                ACTIVE_TRADES[symbol]._version += 1
                        finally:
                            bot.release_trade_lock(symbol)

async def fetch_ticker_with_lock(exchange, symbol):
    try:
        await rate_limiter.wait_if_needed(weight=CONFIG.get("TICKER_WEIGHT", 2))
        ticker = await exchange.fetch_ticker(symbol)
        return ticker
    except Exception as e:
        logger.error(f"[Ticker Error] {symbol}: {e}")
        return None

# ===================== ENHANCED PARTIAL EXIT (مع إدارة SL على المنصة ومع race condition fix) =====================
async def partial_exit(symbol: str, trade: TradeState, exit_price: float,
                      tp_level: str, exit_pct: float, r_multiple: float, exchange=None):
    # الاحتفاظ بالقفل طوال العملية
    if not await bot.get_trade_lock(symbol):
        logger.error(f"[partial_exit] Failed to acquire lock for {symbol}")
        return
    
    try:
        if symbol not in ACTIVE_TRADES:
            return
        
        current_trade = ACTIVE_TRADES[symbol]
        
        # التحقق من is_exiting
        if current_trade.is_exiting:
            logger.warning(f"[partial_exit] Exit already in progress for {symbol}, skipping")
            return
        
        # تعيين علامة الخروج
        current_trade.is_exiting = True
        
        # قراءة البيانات المطلوبة
        if tp_level == "TP1" and current_trade.tp1_hit:
            return
        elif tp_level == "TP2" and current_trade.tp2_hit:
            return
        elif tp_level == "TP3" and current_trade.tp3_hit:
            return
        
        if tp_level == "TP1":
            current_trade.tp1_order_done = True
        elif tp_level == "TP2":
            current_trade.tp2_order_done = True
        elif tp_level == "TP3":
            current_trade.tp3_order_done = True
        
        entry_fill_amount = current_trade.entry_fill_amount
        current_version = current_trade._version
        sl_order_id = current_trade.sl_order_id
        
        # حفظ حالة مؤقتة (يمكن تحديثها لاحقاً)
        db_manager.save_trade(current_trade)
        
        # تنفيذ البيع (داخل القفل)
        sell_amount_base = entry_fill_amount * exit_pct
        live_sell_ok = True
        fill_price = exit_price  # افتراضي
        
        if is_live_trading_enabled() and exchange and sell_amount_base > 0:
            sell_order = await market_sell_safe(exchange, symbol, sell_amount_base, max_retries=2)
            live_sell_ok = bool(sell_order)
            
            if live_sell_ok and sell_order:
                # استخراج سعر التنفيذ الفعلي
                fill_price = safe_float(sell_order.get('average')) or safe_float(sell_order.get('price')) or exit_price
                if not validate_price(fill_price):
                    fill_price = exit_price
            
            if not live_sell_ok:
                # إعادة تعيين العلامات
                if tp_level == "TP1":
                    current_trade.tp1_order_done = False
                elif tp_level == "TP2":
                    current_trade.tp2_order_done = False
                elif tp_level == "TP3":
                    current_trade.tp3_order_done = False
                
                # إلغاء علامة الخروج
                current_trade.is_exiting = False
                
                # حفظ الحالة المعدلة
                db_manager.save_trade(current_trade)
                
                # تفعيل حالة الطوارئ
                asyncio.create_task(mark_trade_emergency(symbol, f"partial_sell_failed({tp_level})"))
                return
            
            # إذا كان هناك أمر SL على المنصة، يجب إلغاؤه وتحديثه بالكمية المتبقية
            if sl_order_id and exchange:
                remaining_base = entry_fill_amount * (current_trade.remaining_position - exit_pct)
                if remaining_base > 0:
                    # إلغاء الأمر القديم
                    await cancel_stop_loss_order(exchange, symbol, sl_order_id)
                    # وضع أمر جديد للكمية المتبقية
                    new_sl_order = await place_stop_loss_order(exchange, symbol, current_trade.current_sl, remaining_base)
                    if new_sl_order:
                        current_trade.sl_order_id = str(new_sl_order.get("id"))
                    else:
                        logger.warning(f"[partial_exit] Failed to place new SL for {symbol}")
        
        # تحديث الحالة بعد البيع
        risk = trade.entry - trade.original_sl
        actual_r_multiple = (fill_price - trade.entry) / risk if risk > 0 else 0
        exit_r = actual_r_multiple * exit_pct
        
        # التحقق من الإصدار (دمج بسيط)
        if current_trade._version != current_version:
            logger.warning(f"[partial_exit] Version mismatch for {symbol}. Expected {current_version}, got {current_trade._version}. Proceeding.")
        
        current_trade.remaining_position -= exit_pct
        current_trade.total_realized_r += exit_r
        
        if tp_level == "TP1":
            current_trade.tp1_hit = True
        elif tp_level == "TP2":
            current_trade.tp2_hit = True
        elif tp_level == "TP3":
            current_trade.tp3_hit = True
        
        current_trade._version += 1
        current_trade.is_exiting = False  # إعادة تعيين علامة الخروج
        
        db_manager.save_trade(current_trade)
        
    finally:
        bot.release_trade_lock(symbol)
    
    if tp_level == "TP1":
        STATS["tp1_hits"] += 1
    elif tp_level == "TP2":
        STATS["tp2_hits"] += 1
    elif tp_level == "TP3":
        STATS["tp3_hits"] += 1
    
    STATS["trades_partial"] += 1
    STATS["total_r_multiple"] += exit_r
    record_daily_r(exit_r)
    
    profit_pct = ((fill_price - trade.entry) / trade.entry) * 100
    mode_badge = "✅ LIVE" if is_live_trading_enabled() else ("🟨 PAPER" if getattr(trade, "is_paper", False) else "🟦 SIGNAL")
    
    if await bot.get_trade_lock(symbol):
        try:
            current_remaining = ACTIVE_TRADES[symbol].remaining_position if symbol in ACTIVE_TRADES else 0
        finally:
            bot.release_trade_lock(symbol)
    else:
        current_remaining = trade.remaining_position - exit_pct
    
    message = f"""
✅ {escape_html(tp_level)} Hit - {escape_html(symbol)}  ({mode_badge})

• الدخول: {trade.entry:.6f}
• الخروج (سعر): {fill_price:.6f}
• الربح: {profit_pct:+.2f}%
• مضاعف R: {actual_r_multiple:.2f}R
• نسبة الخروج: {exit_pct*100:.0f}%
• R المحقق: {exit_r:.2f}R
• المتبقي: {(current_remaining*100):.0f}%
{"• LIVE Sell: ✅" if is_live_trading_enabled() and live_sell_ok else ("• LIVE Sell: ❌" if is_live_trading_enabled() else "")}
"""
    await send_telegram(message)
    
    logger.info(f"[Partial Exit] {symbol} - {tp_level} - {profit_pct:+.2f}% - {exit_pct*100:.0f}%")
    
    if current_remaining <= 0.01:
        await close_trade_full(symbol, fill_price, "ALL_TPS", exchange=exchange)

# ===================== INSTITUTIONAL CLOSE TRADE FULL (مُعدل لإصلاح double counting) =====================
async def close_trade_full(symbol: str, exit_price: float, exit_type: str, exchange=None, sell_order_info: Optional[Dict] = None):
    trade = None
    remaining = 0
    fill_amount = 0
    try:
        if not await bot.get_trade_lock(symbol):
            logger.error(f"[close_trade_full] Failed to acquire lock for {symbol}")
            return
        
        try:
            if symbol not in ACTIVE_TRADES:
                return
            
            trade = ACTIVE_TRADES[symbol]
            remaining = trade.remaining_position
            fill_amount = trade.entry_fill_amount
            entry = trade.entry
            original_sl = trade.original_sl
            total_realized_r = trade.total_realized_r
            quantum_score = getattr(trade, "quantum_score", 0.0)
            signal_class = getattr(trade, "signal_class", "")
            gates_passed = getattr(trade, "gates_passed", [])
            tp1_hit = trade.tp1_hit
            tp2_hit = trade.tp2_hit
            tp3_hit = trade.tp3_hit
            execution_mode = trade.execution_mode
            version = trade._version
            sl_order_id = trade.sl_order_id
            
            ACTIVE_TRADES[symbol].sl_order_done = True
        finally:
            bot.release_trade_lock(symbol)
    except Exception:
        return
    
    risk = entry - original_sl
    r_multiple = (exit_price - entry) / risk if risk > 0 else 0
    
    live_sell_ok = True
    fill_price = exit_price
    if is_live_trading_enabled() and exchange is not None and remaining > 0.01 and exit_type in ("SL", "ALL_TPS"):
        remaining_base = fill_amount * remaining
        sell_order = await market_sell_safe(exchange, symbol, remaining_base, max_retries=3)
        live_sell_ok = bool(sell_order)
        
        if live_sell_ok and sell_order:
            fill_price = safe_float(sell_order.get('average')) or safe_float(sell_order.get('price')) or exit_price
        
        if not live_sell_ok:
            if await bot.get_trade_lock(symbol):
                try:
                    if symbol in ACTIVE_TRADES:
                        ACTIVE_TRADES[symbol].sl_order_done = False
                finally:
                    bot.release_trade_lock(symbol)
            
            await mark_trade_emergency(
                symbol,
                reason=f"full_close_sell_failed({exit_type})",
                critical_msg=(
                    f"🚨 EMERGENCY: فشل إغلاق الصفقة (Sell Failed)\n\n"
                    f"الرمز: {escape_html(symbol)}\n"
                    f"النوع: {escape_html(exit_type)}\n"
                    f"الكمية المتبقية (BASE): {remaining_base}\n"
                    f"السعر الحالي (تقريبي): {exit_price:.6f}\n\n"
                    f"⚠️ تحذير: الصفقة لم تُغلق فعليًا. يرجى البيع يدويًا فورًا!"
                )
            )
            return
        
        # إلغاء أمر SL على المنصة إذا كان موجوداً
        if sl_order_id and exchange:
            await cancel_stop_loss_order(exchange, symbol, sl_order_id)
    
    # تعريف final_exit_r قبل if
    final_exit_r = 0.0
    if remaining > 0:
        final_exit_r = r_multiple * remaining
        # نضيف فقط الجزء النهائي إلى الإحصائيات
        total_r = total_realized_r + final_exit_r
    else:
        total_r = total_realized_r
    
    profit_pct = ((fill_price - entry) / entry) * 100
    
    if exit_type == "SL":
        STATS["trades_lost"] += 1
        STATS["total_r_multiple"] += final_exit_r  # إضافة الجزء النهائي فقط
    else:
        STATS["trades_won"] += 1
        STATS["total_r_multiple"] += final_exit_r  # إضافة الجزء النهائي فقط
    
    record_daily_r(final_exit_r)  # تسجيل الجزء النهائي فقط
    
    # تخزين total_r في سجل التاريخ (r_multiple = total_r)
    db_manager.record_trade_history(symbol, trade, fill_price, exit_type, total_r, execution_mode)
    
    mode_badge = "✅ LIVE" if is_live_trading_enabled() else ("🟨 PAPER" if getattr(trade, "is_paper", False) else "🟦 SIGNAL")
    
    message = f"""
{escape_html(exit_type)} - تم إغلاق الصفقة - {escape_html(symbol)}  ({mode_badge})

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
• الهدف 3: {'✅' if tp3_hit else '❌'}
"""
    
    await send_telegram(message)
    
    set_symbol_cooldown(symbol)
    
    if await bot.get_trade_lock(symbol):
        try:
            ACTIVE_TRADES.pop(symbol, None)
        finally:
            bot.release_trade_lock(symbol)
    
    db_manager.remove_trade(symbol)
    
    logger.info(f"[Trade Closed] {symbol} - {exit_type} - {profit_pct:+.2f}% - {final_exit_r:.2f}R (total {total_r:.2f}R)")

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
📊 تقرير الأداء - Quantum Flow v1.8.4 ULTIMATE INSTITUTIONAL EDITION
━━━━━━━━━━━━━━━━━━━━━━━━━━

🧾 الوضع
• LIVE: {'ON' if is_live_trading_enabled() else 'OFF'}
• PAPER: {'ON' if is_paper_trading_enabled() else 'OFF'}
• Entry: LIMIT (Zone-based)
• No-Chasing Gate: {'ON' if CONFIG.get('ENABLE_PRICE_ACCEPTANCE_GATE', True) else 'OFF'}
• Exits: Internal (Market Sell Safe)
• SL on Exchange: {'ON' if CONFIG.get('LIVE_PLACE_SL_ORDER') else 'OFF'}

✅ التحسينات المؤسسية المطبقة
• ✅ Atomic Partial Exit with Optimistic Locking + is_exiting flag
• ✅ Enhanced Lock Manager with Recovery & Blacklisting (TTL)
• ✅ Database Connection Leaks Fixed
• ✅ Smart Cache with Memory Management
• ✅ Exponential Backoff Retry Strategy
• ✅ Enhanced Health Check with Diagnostics
• ✅ Metrics Collector for Performance Monitoring
• ✅ FVG Logic Corrected
• ✅ Telegram Date Format Fixed
• ✅ TA-Lib Import Fixed
• ✅ Liquidity Grab Support/Resistance Fix
• ✅ Daily Circuit Double Counting Fix
• ✅ Order Flow Sampling Logic Fixed (staggered per symbol + stable hash)
• ✅ Position Sizing Order Corrected
• ✅ Volume Gate Slice Safety Added
• ✅ Volume Profile precheck_score fix
• ✅ close_trade_full final_exit_r defined
• ✅ Trade history stores total R, stats incremental
• ✅ SL order management on partial exits
• ✅ Balance Reconciliation Task Added
• ✅ asyncio.timeout fallback removed, replaced with asyncio.wait_for

🧯 Daily Circuit
• Enabled: {'ON' if CONFIG.get('ENABLE_DAILY_MAX_LOSS', True) else 'OFF'}
• Max Loss (R): {CONFIG.get('DAILY_MAX_LOSS_R')}
• Max Daily A+: {CONFIG.get('MAX_DAILY_A_PLUS', 3)}

⏳ Cooldown
• Seconds: {CONFIG.get('SYMBOL_COOLDOWN_SEC')}

🆘 Emergency Monitor
• Enabled: {'ON' if is_live_trading_enabled() else 'OFF'}

📊 Metrics
• Signal Generation: {metrics.get_percentiles('signal_generation').get('p50', 0):.3f}s (p95)
• MTF Analysis: {metrics.get_percentiles('mtf_analysis').get('p50', 0):.3f}s (p95)
• Order Flow: {metrics.get_percentiles('order_flow_analysis').get('p50', 0):.3f}s (p95)

{"🔄 " + str(len(ACTIVE_TRADES)) + " صفقة تم استردادها" if ACTIVE_TRADES else ""}

جاهز! 🎯
"""
            cache_stats = cache.get_stats()
            basic_report += f"\n📊 ذاكرة التخزين المؤقت\n"
            basic_report += f"• معدل النجاح: {cache_stats['hit_rate']:.1f}%\n"
            basic_report += f"• الحجم: {cache_stats['size']}\n"
            basic_report += f"• متوسط الوصولات: {cache_stats['avg_access_count']:.1f}\n"
            
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
        
        report = f"""
📊 تقرير الأداء - Quantum Flow v1.8.4 ULTIMATE INSTITUTIONAL EDITION
━━━━━━━━━━━━━━━━━━━━━━━━━━

🧾 الوضع
• LIVE: {'ON' if is_live_trading_enabled() else 'OFF'}
• PAPER: {'ON' if is_paper_trading_enabled() else 'OFF'}
• Cooldown (sec): {CONFIG.get('SYMBOL_COOLDOWN_SEC')}

🎯 الإشارات (اليوم/الإجمالي)
• المولدة: {STATS['signals_generated']}
• A+ : {STATS['signals_a_plus']} (اليوم: {STATS.get('daily_a_plus_count', 0)}/{CONFIG.get('MAX_DAILY_A_PLUS', 3)})
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
• الهدف 3: {STATS['tp3_hits']}
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
• Blacklisted Symbols: {sum(1 for s in bot.lock_manager.failed_locks.keys() if bot.lock_manager.is_blacklisted(s))}
"""
        
        if metrics_summary:
            report += f"\n⏱️ مقاييس الأداء\n"
            for operation, data in metrics_summary.items():
                report += f"• {operation}: p50={data.get('p50', 0):.3f}s, p95={data.get('p95', 0):.3f}s, count={data.get('count', 0)}\n"
        
        cache_stats = cache.get_stats()
        report += f"\n📊 ذاكرة التخزين المؤقت\n"
        report += f"• معدل النجاح: {cache_stats['hit_rate']:.1f}%\n"
        report += f"• الحجم: {cache_stats['size']}\n"
        report += f"• متوسط الوصولات: {cache_stats['avg_access_count']:.1f}\n"
        
        return report
    
    except Exception as e:
        logger.error(f"[Report Generation Error] {e}")
        return f"❌ فشل إنشاء التقرير: {str(e)[:100]}"

# ===================== COMPUTATIONAL OPTIMIZATION (مُعدل) =====================
async def toggle_computational_features():
    loop_count = STATS.get("loop_count", 0)
    
    # تعيين علامات أخذ العينات دون تغيير الإعدادات الأساسية
    CONFIG["_ORDER_FLOW_SAMPLING_OK"] = (loop_count % 3 == 0)
    CONFIG["_VOLUME_PROFILE_SAMPLING_OK"] = (loop_count % 5 == 0)
    
    if loop_count % 7 == 0:
        await cache.smart_cache_cleanup()

# ===================== HEALTH CHECK ENDPOINT =====================
async def health_check_handler(request):
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
    
    blacklisted_count = sum(1 for s in bot.lock_manager.failed_locks.keys() 
                           if bot.lock_manager.is_blacklisted(s))
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
            logger.info(f"  {stat}")

# ===================== ENHANCED MAIN LOOP =====================
async def main_loop(exchange):
    emergency_monitor_task = None
    checkpoint_task = None
    memory_task = None
    reconciliation_task = None
    runner = None
    site = None
    
    try:
        logger.info("="*70)
        logger.info("🚀 QUANTUM FLOW TRADING BOT v1.8.4 - ULTIMATE INSTITUTIONAL EDITION")
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
        logger.info(f"METRICS COLLECTOR: ✅ (مع تصحيح success)")
        logger.info(f"EXPONENTIAL BACKOFF RETRY: ✅")
        logger.info("✅ جميع الإصلاحات الحرجة مطبقة:")
        logger.info("  1. إصلاح استيراد talib (إزالة الاستيراد العلوي)")
        logger.info("  2. إصلاح toggle_computational_features (استخدام علامات بدلاً من تعديل الإعدادات)")
        logger.info("  3. إصلاح Liquidity Grab (استبعاد آخر شمعة)")
        logger.info("  4. إصلاح double counting في close_trade_full (final_exit_r معرّف، واستخدام total_r في التاريخ)")
        logger.info("  5. إضافة أمر وقف خسارة على المنصة مع الإلغاء والتحديث بعد الخروج الجزئي")
        logger.info("  6. استخدام سعر التنفيذ الفعلي من أوامر البيع")
        logger.info("  7. إعادة ترتيب حساب حجم الصفقة (SL أولاً)")
        logger.info("  8. إضافة التحقق من صحة slice الحجم في price_acceptance_gate")
        logger.info("  9. إزالة المتغيرات غير المستخدمة (next_candle)")
        logger.info(" 10. تحسين إيقاف تشغيل خادم health check")
        logger.info(" 11. تحسين أخذ عينات order flow (staggered per symbol) + stable hash")
        logger.info(" 12. إصلاح Volume Profile بتمرير precheck_score")
        logger.info(" 13. تحسين المراقبة بجلب tickers بالتوازي")
        logger.info(" 14. تحسين التحقق من صحة الرموز (whitelist)")
        logger.info(" 15. إصلاح Race Condition في partial_exit (قفل طوال العملية + is_exiting)")
        logger.info(" 16. إزالة fallback asyncio.timeout واستخدام asyncio.wait_for مباشرة")
        logger.info(" 17. إضافة مهمة Reconciliation للمطابقة بين الرصيد والمراكز")
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
            await ensure_live_trading_ready(exchange)
        
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
            
            reconciliation_task = asyncio.create_task(reconcile_balances(exchange))
            shutdown_manager.add_task(reconciliation_task)
            logger.info("[Main] Balance reconciliation task started")
        
        await send_telegram(f"""
🚀 تم تشغيل Quantum Flow Bot v1.8.4 - ULTIMATE INSTITUTIONAL EDITION

🧾 الوضع
• LIVE TRADING: {'ON' if is_live_trading_enabled() else 'OFF'}
• PAPER TRADING: {'ON' if is_paper_trading_enabled() else 'OFF'}
• Entry: LIMIT (Zone-based)
• No-Chasing Gate: {'ON' if CONFIG.get('ENABLE_PRICE_ACCEPTANCE_GATE', True) else 'OFF'}
• Exits: Internal (Market Sell Safe)
• SL on Exchange: {'ON' if CONFIG.get('LIVE_PLACE_SL_ORDER') else 'OFF'}

✅ التحسينات المؤسسية المطبقة
• ✅ Atomic Partial Exit with Optimistic Locking + is_exiting flag
• ✅ Enhanced Lock Manager with Recovery & Blacklisting (TTL)
• ✅ Database Connection Leaks Fixed
• ✅ Smart Cache with Memory Management
• ✅ Exponential Backoff Retry Strategy
• ✅ Enhanced Health Check with Diagnostics
• ✅ Metrics Collector for Performance Monitoring
• ✅ FVG Logic Corrected
• ✅ Telegram Date Format Fixed
• ✅ TA-Lib Import Fixed
• ✅ Liquidity Grab Support/Resistance Fix
• ✅ Daily Circuit Double Counting Fix
• ✅ Order Flow Sampling Logic Fixed (staggered per symbol + stable hash)
• ✅ Position Sizing Order Corrected
• ✅ Volume Gate Slice Safety Added
• ✅ Volume Profile precheck_score fix
• ✅ close_trade_full final_exit_r defined
• ✅ Trade history stores total R, stats incremental
• ✅ SL order management on partial exits
• ✅ Balance Reconciliation Task Added
• ✅ asyncio.timeout fallback removed, replaced with asyncio.wait_for

🧯 Daily Circuit
• Enabled: {'ON' if CONFIG.get('ENABLE_DAILY_MAX_LOSS', True) else 'OFF'}
• Max Loss (R): {CONFIG.get('DAILY_MAX_LOSS_R')}
• Max Daily A+: {CONFIG.get('MAX_DAILY_A_PLUS', 3)}

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
                
                btc_status = await check_btc_trend(exchange)
                if not btc_status['safe_to_trade']:
                    logger.warning(f"[Main] التداول متوقف مؤقتاً - BTC {btc_status['trend']}")
                    await asyncio.sleep(60)
                    continue
                
                if loop_count % 5 == 0 or not all_symbols:
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
                              f"blacklisted={sum(1 for s in bot.lock_manager.failed_locks.keys() if bot.lock_manager.is_blacklisted(s))}"
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

# ===================== ENTRY POINT =====================
async def async_main():
    exchange = None
    
    try:
        logger.info("\n" + "="*70)
        logger.info("QUANTUM FLOW v1.8.4 - ULTIMATE INSTITUTIONAL EDITION")
        logger.info("✅ جميع التحسينات المؤسسية المطلوبة والإصلاحات الحرجة مطبقة")
        logger.info("="*70)
        
        if not SCIPY_AVAILABLE:
            logger.warning("⚠️ scipy غير متوفر - استخدام الطرق البديلة")
        if not ML_AVAILABLE:
            logger.warning("⚠️ sklearn غير متوفر - ميزات ML معطلة")
        if not TALIB_AVAILABLE:
            logger.info("ℹ️ TA-Lib غير متوفر - سيتم استخدام ta (عادي)")
        
        load_telegram_from_env()
        
        if CONFIG["ENABLE_DB_PERSISTENCE"]:
            logger.info("✅ ثبات قاعدة البيانات مفعّل")
        
        _ensure_runtime_paths()
        
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
        
        await main_loop(exchange)
        
    except KeyboardInterrupt:
        logger.info("\n[النظام] تم استلام إشارة الإيقاف في async_main")
    except Exception as e:
        logger.error(f"[خطأ فادح في async_main] {str(e)}")
        traceback.print_exc()
        save_emergency_checkpoint(e)
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
    """نقطة الدخول الرئيسية مع حلقة إعادة تشغيل تلقائية"""
    while True:
        try:
            shutdown_manager.should_stop = False
            asyncio.run(async_main())
        except KeyboardInterrupt:
            logger.info("\n👋 توقف البوت بواسطة المستخدم")
            break
        except Exception as e:
            logger.error(f"💥 خطأ فادح في التشغيل: {str(e)}")
            traceback.print_exc()
            logger.info("سيتم إعادة التشغيل بعد 60 ثانية...")
            time.sleep(60)
    logger.info("✅ التنظيف مكتمل")

if __name__ == "__main__":
    main()            # Not supported in notebook/Windows; ignore
            pass

# Prevent duplicate bot tasks in same kernel
_MAIN_LOOP_TASK: Optional[asyncio.Task] = None

# ===================== ENUMS =====================
class EntryType(Enum):
    RETEST_ORDER_BLOCK = "RETEST_ORDER_BLOCK"
    RETEST_EMA = "RETEST_EMA"
    LIQUIDITY_GRAB = "LIQUIDITY_GRAB"

class ZoneType(Enum):
    ORDER_BLOCK = "ORDER_BLOCK"
    EMA = "EMA"
    PREVIOUS_HIGH = "PREVIOUS_HIGH"

class ExecutionMode(Enum):
    SIGNAL = "SIGNAL"
    PAPER = "PAPER"
    LIVE = "LIVE"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class TimeInForce(Enum):
    GTC = "GTC"   # Good till cancelled
    IOC = "IOC"   # Immediate or cancel
    FOK = "FOK"   # Fill or kill

# ===================== DATA CLASSES =====================
@dataclass
class MicroBOS:
    detected: bool
    break_price: float
    previous_high: float
    break_candle_index: int
    strength: float
    volume_spike: bool

@dataclass
class RetestZone:
    zone_type: str
    low: float
    high: float
    mid: float
    freshness: int

@dataclass
class ScalpEntry:
    detected: bool
    entry_price: float
    entry_type: str
    retest_zone: Optional[RetestZone]
    confirmation_candle: Dict[str, Any]
    volume_ok: bool
    rejection_wick: bool
    liquidity_sweep: bool
    pullback_ok: bool
    bos_index: int
    retest_index: int

# ===================== TYPE DEFINITIONS =====================
class SignalData(TypedDict):
    symbol: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    atr: float
    position_size_usdt: float
    position_size_pct: float
    quantum_score: float
    micro_bos: Dict[str, Any]
    entry_signal: Dict[str, Any]
    spread_pct: float
    timestamp: str

@dataclass
class Order:
    """تمثل أمر دخول حقيقي أو ورقي - مع حفظ الإشارة الأصلية"""
    symbol: str
    side: str  # 'buy'
    order_type: str  # 'limit'
    price: float
    amount: float  # الكمية (عملة الأساس)
    cost: float  # القيمة بالـ USDT
    status: OrderStatus = OrderStatus.PENDING
    filled_amount: float = 0.0
    filled_cost: float = 0.0
    fill_price_avg: float = 0.0
    order_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    timeout_sec: int = 60
    signal_snapshot: Optional[SignalData] = None

@dataclass
class TradeState:
    """حالة الصفقة النشطة (بعد التنفيذ)"""
    symbol: str
    entry_price: float
    original_sl: float
    current_sl: float
    tp1: float
    tp2: float
    tp3: float
    atr: float
    position_size_usdt: float
    position_size_asset: float
    original_position_size_asset: float  # 🟢 NEW: الكمية الأصلية للحساب الدقيق
    quantum_score: float
    is_paper: bool
    execution_mode: str
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    remaining_pct: float = 1.0
    be_moved: bool = False
    trailing_active: bool = False
    entry_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_update: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_realized_r: float = 0.0
    order_id: Optional[str] = None
    fill_log: List[Dict] = field(default_factory=list)
    closed: bool = False

# ===================== LOGGING WITH ROTATION =====================
def setup_logging():
    logger_instance = logging.getLogger(__name__)
    logger_instance.setLevel(logging.INFO)

    # 🟢 CRITICAL (Colab): منع تكرار handlers عند إعادة تشغيل الخلية
    if logger_instance.handlers:
        return logger_instance

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        'micro_bos_scalping.log',
        maxBytes=10_000_000,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger_instance.addHandler(console_handler)
    logger_instance.addHandler(file_handler)

    # avoid double logging to root
    logger_instance.propagate = False

    return logger_instance

logger = setup_logging()

# ===================== CONFIGURATION WITH ENV VARS =====================
def load_config() -> Dict[str, Any]:
    config = {
        # Exchange
        "EXCHANGE": "mexc",
        "QUOTE": "/USDT",

        # Timeframes
        "TF_ANALYSIS": "15m",
        "TF_ENTRY": "5m",

        # Market Structure Settings
        "SWING_WINDOW": 3,
        "BOS_CONFIRMATION_CANDLES": 1,
        "BOS_CONFIRMATION_MULTIPLIER": 1.0005,
        "ORDER_BLOCK_LOOKBACK": 10,

        # Scalping Filters
        "MIN_EMA200_FILTER": True,
        "MIN_EMA50_ABOVE_200": False,           # 🟢 NEW: أصبح اختيارياً
        "MIN_RSI": 45,                           # 🟢 NEW: مخفف
        "MIN_ADX": 15,                            # 🟢 NEW: مخفف

        # Micro BOS Detection
        "MICRO_BOS_LOOKBACK": 5,
        "MICRO_BOS_MIN_BREAK": 0.001,  # 0.10%
        "MICRO_BOS_CLOSE_CONFIRMATION": True,  # 🟢 NEW: تأكيد الإغلاق فوق القمة

        # Liquidity Grab
        "LG_WICK_MIN_RATIO": 0.4,
        "LG_VOLUME_MULTIPLIER": 1.3,
        "LG_REJECTION_REQUIRED": True,

        # Institutional filters
        "ENABLE_LIQUIDITY_SWEEP_FILTER": True,
        "LIQUIDITY_SWEEP_LOOKBACK": 5,
        "MIN_PULLBACK_ATR": 0.3,
        "LIQUIDITY_SWEEP_REQUIRED": False,       # 🟢 NEW: لم يعد إلزامياً

        # Post-BOS lookahead
        "POST_BOS_LOOKAHEAD_CANDLES": 6,

        # Risk Management
        "RISK_PER_TRADE_PCT": 0.75,
        "MAX_SL_PCT": 0.6,
        "ATR_SL_MULT": 1.5,

        # Take Profit
        "TP1_RR": 1.0,
        "TP2_RR": 2.0,
        "TP3_RR": 3.0,
        "TP1_EXIT_PCT": 0.5,
        "TP2_EXIT_PCT": 0.3,
        "TP3_EXIT_PCT": 0.2,

        # Breakeven & Trailing
        "BE_AT_R": 1.0,
        "BE_ATR_MULT": 0.5,
        "TRAIL_START_R": 2.0,
        "TRAIL_ATR_MULT": 1.0,

        # Position Sizing
        "ACCOUNT_SIZE_USDT": 1000,
        "MIN_POSITION_SIZE_USDT": 10,
        "MAX_POSITION_SIZE_USDT": 100,

        # Trading Settings
        "LONG_ONLY": True,
        "MIN_QUANTUM_SCORE": 50,                  # 🟢 NEW: مخفف

        # Entry Quality Filter
        "ENABLE_ENTRY_QUALITY_FILTER": True,
        "ENTRY_QUALITY_MAX_ATR_PCT_5M": 4.5,      # 🟢 NEW: أوسع للسكالبينج
        "ENTRY_QUALITY_MAX_BB_WIDTH_5M": 0.08,    # 🟢 NEW: أوسع
        "ENTRY_QUALITY_MAX_DISTANCE_FROM_ZONE_ATR": 1.2,  # 🟢 NEW: أوسع
        "MAX_MOMENTUM_CANDLE_PCT": 0.025,         # 🟢 NEW: لن يُستخدم (تم حذف الفلتر)

        # Volume Confirmation
        "MIN_VOLUME_RATIO": 0.8,
        "CONFIRM_CANDLE_VOLUME_RATIO": 0.7,

        # Liquidity filters
        "MIN_24H_VOLUME": 300_000,                 # 🟢 NEW: مخفف
        "MAX_SPREAD_PCT": 0.15,
        "MAX_SYMBOLS_TO_TRADE": 200,

        # Cooldown per symbol
        "MIN_SIGNAL_INTERVAL_SEC": 1800,  # 30 min

        # Slippage buffer
        "ENTRY_SLIPPAGE_BUFFER": 0.0002,  # 0.02%

        # Live Trading
        "ENABLE_LIVE_TRADING": os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true",
        "MEXC_API_KEY": os.getenv("MEXC_API_KEY", ""),
        "MEXC_API_SECRET": os.getenv("MEXC_API_SECRET", ""),
        "LIVE_MAX_OPEN_TRADES": 3,
        "ENTRY_ORDER_TYPE": "limit",
        "ENTRY_LIMIT_TIMEOUT_SEC": 60,
        "ENTRY_ORDER_TIME_IN_FORCE": "IOC",

        # Paper Trading
        "PAPER_TRADING_MODE": os.getenv("PAPER_TRADING_MODE", "false").lower() == "true",

        # Market Hours Filter
        "ENABLE_MARKET_HOURS_FILTER": True,
        # تم التعديل: ساعات التداول أصبحت 24 ساعة (كل الساعات)
        "BEST_HOURS_UTC": list(range(24)),

        # Daily Circuit Breaker
        "ENABLE_DAILY_MAX_LOSS": True,
        "DAILY_MAX_LOSS_R": -3.0,
        "DAILY_MAX_TRADES": 15,

        # Telegram
        "TG_TOKEN": os.getenv("TG_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN", ""),
        "TG_CHAT": os.getenv("TG_CHAT") or os.getenv("TELEGRAM_CHAT_ID", ""),
        "SILENT_MODE": False,

        # Database
        "ENABLE_DB_PERSISTENCE": True,
        "DB_PATH": "micro_bos_scalping.db",

        # Rate Limiting
        "REQUESTS_PER_MINUTE": 1200,
        "MAX_WEIGHT_PER_MINUTE": 6000,
        "REQUESTS_PER_SECOND": 10,

        # Checkpoints
        "ENABLE_CHECKPOINTS": True,
        "CHECKPOINT_INTERVAL_SEC": 180,
        "CHECKPOINT_PATH": "/content/micro_bos_checkpoint.pkl",

        # Debug
        "DEBUG_MODE": os.getenv("DEBUG_MODE", "false").lower() == "true",

        # Batch Processing
        "BATCH_SIZE": 5,

        # Institutional Features
        "CIRCUIT_BREAKER_ENABLED": True,
        "ENABLE_HEALTH_CHECK": True,
        "HEALTH_CHECK_PORT": 8080,

        # Production settings
        "SYMBOL_REFRESH_INTERVAL_HOURS": 6,
        "STATE_FILE_PATH": "state.json",
        "STATE_SAVE_INTERVAL_SEC": 300,

        # Cleanup settings
        "STALE_TRADE_CLEANUP_DAYS": 7,
        "MEMORY_CLEANUP_INTERVAL_SEC": 3600,
        "CACHE_CLEANUP_INTERVAL_SEC": 900,  # 🟢 NEW: تنظيف cache كل 15 دقيقة

        # Fees
        "FEE_RATE": 0.001,  # 0.1%

        # Fill simulation
        "FILL_PROBABILITY_LIMIT": 0.8,
        "PARTIAL_FILL_PROBABILITY": 0.3,
        "PARTIAL_FILL_MIN_PCT": 0.3,
        "PARTIAL_FILL_MAX_PCT": 0.9,
        "ORDER_TIMEOUT_CANCEL": True,

        # 🟢 NEW: Validation settings
        "ENABLE_SIGNAL_VALIDATION": True,
        "MAX_PRICE_DEVIATION_PCT": 0.5,  # السعر لا يتحرك أكثر من 0.5% قبل التنفيذ
        "MAX_SIGNAL_AGE_SEC": 300,  # الإشارة صالحة لـ 5 دقائق فقط

        # 🟢 NEW: Error tracking
        "ERROR_TRACK_MAX_ERRORS": 5,  # عدد الأخطاء قبل cooldown
        "ERROR_TRACK_COOLDOWN_SEC": 3600,  # ساعة واحدة cooldown

        # 🟢 NEW: Price feed cache
        "PRICE_FEED_CACHE_DURATION_SEC": 0.7,      # 🟢 NEW: مخفض جداً للسكالبينج

        # ========== ميزات مؤسسية جديدة ==========
        # Market Regime Filter
        "ENABLE_MARKET_REGIME_FILTER": True,
        "MIN_ADX_FOR_TREND": 20,            # ADX أعلى من هذا يعني اتجاه
        "MAX_CHASE_MOVE_PCT": 0.03,          # لتجنب الدخول بعد حركة كبيرة

        # BTC Filter
        "ENABLE_BTC_FILTER": True,
        "BTC_CRASH_THRESHOLD": -3.0,         # انهيار خلال ساعة
        "BTC_WARNING_THRESHOLD": -1.5,
        "BTC_SYMBOL": "BTC/USDT",

        # Volume Regime Filter
        "ENABLE_VOLUME_REGIME": True,
        "VOLUME_REGIME_MIN_RATIO": 1.2,      # الحجم الحالي / متوسط الأيام > هذا
        "VOLUME_REGIME_DAYS": 5,

        # Order Flow Light
        "ENABLE_ORDERFLOW_LIGHT": True,
        "ORDERFLOW_IMBALANCE_THRESHOLD": 0.25,

        # Loss Cooldown
        "LOSS_COOLDOWN_SEC": 3600,            # ساعة بعد خسارة

        # TradingView exchange name
        "EXCHANGE_NAME_FOR_TV": "MEXC",
    }

    return config

CONFIG = load_config()

# ===================== CONFIG VALIDATION =====================
def validate_config() -> Tuple[bool, str]:
    try:
        if not (0 < CONFIG["RISK_PER_TRADE_PCT"] <= 2):
            return False, "RISK_PER_TRADE_PCT must be between 0 and 2"

        if CONFIG["MAX_SL_PCT"] >= 2:
            return False, "MAX_SL_PCT too wide (>= 2%)"

        if not (CONFIG["TP1_RR"] < CONFIG["TP2_RR"] < CONFIG["TP3_RR"]):
            return False, "TP ratios must be ascending"

        total_exit = CONFIG["TP1_EXIT_PCT"] + CONFIG["TP2_EXIT_PCT"] + CONFIG["TP3_EXIT_PCT"]
        if abs(total_exit - 1.0) > 0.01:
            return False, f"Total exit percentages must equal 1.0 (current: {total_exit})"

        if CONFIG["MIN_POSITION_SIZE_USDT"] > CONFIG["MAX_POSITION_SIZE_USDT"]:
            return False, "MIN_POSITION_SIZE > MAX_POSITION_SIZE"

        if CONFIG["ENABLE_LIVE_TRADING"]:
            if not CONFIG["MEXC_API_KEY"] or not CONFIG["MEXC_API_SECRET"]:
                return False, "Live trading enabled but API credentials missing"

        if not CONFIG["SILENT_MODE"]:
            if not CONFIG["TG_TOKEN"] or not CONFIG["TG_CHAT"]:
                logger.warning("Telegram credentials missing - notifications disabled")

        if not (0 < CONFIG["MIN_QUANTUM_SCORE"] <= 100):
            return False, "MIN_QUANTUM_SCORE must be between 0 and 100"

        valid_tif = ["GTC", "IOC", "FOK"]
        if CONFIG["ENTRY_ORDER_TIME_IN_FORCE"] not in valid_tif:
            return False, f"ENTRY_ORDER_TIME_IN_FORCE must be one of {valid_tif}"

        logger.info("✅ Configuration validation passed")
        return True, "ok"

    except Exception as e:
        return False, f"Validation error: {str(e)}"

config_valid, config_reason = validate_config()
if not config_valid:
    logger.error(f"❌ CONFIGURATION ERROR: {config_reason}")
    sys.exit(1)

# ===================== PERFORMANCE MONITORING =====================
class PerformanceMonitor:
    def __init__(self, window: int = 100):
        self.timings: Deque[float] = deque(maxlen=window)
        self.errors: Deque[str] = deque(maxlen=50)

    def record(self, duration: float):
        self.timings.append(duration)

    def record_error(self, error: str):
        self.errors.append(f"{datetime.now(timezone.utc).isoformat()}: {error}")

    def avg(self) -> float:
        return sum(self.timings) / len(self.timings) if self.timings else 0.0

    def p95(self) -> float:
        if not self.timings:
            return 0.0
        sorted_times = sorted(self.timings)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx] if sorted_times else 0.0

    def p99(self) -> float:
        if not self.timings:
            return 0.0
        sorted_times = sorted(self.timings)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[idx] if sorted_times else 0.0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "count": len(self.timings),
            "avg_ms": self.avg() * 1000,
            "p95_ms": self.p95() * 1000,
            "p99_ms": self.p99() * 1000,
            "error_count": len(self.errors),
        }

# ===================== 🟢 NEW: PRICE FEED CLASS =====================
class PriceFeed:
    """مصدر أسعار موحد للورقي والحي مع cache ذكي"""
    def __init__(self, exchange):
        self.exchange = exchange
        self.cache: Dict[str, Tuple[float, float]] = {}  # {symbol: (price, timestamp)}
        self.cache_duration = CONFIG.get("PRICE_FEED_CACHE_DURATION_SEC", 0.7)

    async def get_price(self, symbol: str) -> Optional[float]:
        """جلب السعر الحالي مع استخدام cache"""
        now = time.time()

        # التحقق من الـ cache
        if symbol in self.cache:
            price, timestamp = self.cache[symbol]
            if now - timestamp < self.cache_duration:
                return price

        # جلب سعر جديد
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            price = ticker.get('last')
            if price and price > 0:
                self.cache[symbol] = (float(price), now)
                return float(price)
        except Exception as e:
            logger.error(f"[PriceFeed] Error fetching {symbol}: {e}")

        return None

    async def get_prices_batch(self, symbols: List[str]) -> Dict[str, float]:
        """جلب أسعار متعددة دفعة واحدة"""
        now = time.time()
        result = {}
        symbols_to_fetch = []

        # استخدام cache للرموز المتاحة
        for symbol in symbols:
            if symbol in self.cache:
                price, timestamp = self.cache[symbol]
                if now - timestamp < self.cache_duration:
                    result[symbol] = price
                    continue
            symbols_to_fetch.append(symbol)

        # جلب الرموز المتبقية
        if symbols_to_fetch:
            try:
                tickers = await self.exchange.fetch_tickers(symbols_to_fetch)
                for symbol, ticker in tickers.items():
                    price = ticker.get('last')
                    if price and price > 0:
                        price = float(price)
                        self.cache[symbol] = (price, now)
                        result[symbol] = price
            except Exception as e:
                logger.error(f"[PriceFeed] Batch fetch error: {e}")

        return result

    def clear_cache(self):
        """مسح الـ cache"""
        self.cache.clear()

# ===================== 🟢 NEW: ERROR TRACKER CLASS =====================
class ErrorTracker:
    """تتبع الأخطاء ووضع cooldown تلقائي للرموز المتكررة الأخطاء"""
    def __init__(self, max_errors: int = 5, cooldown_sec: int = 3600):
        self.max_errors = max_errors
        self.cooldown_sec = cooldown_sec
        self.error_count: Dict[str, int] = defaultdict(int)
        self.cooldown: Dict[str, float] = {}  # {symbol: cooldown_until_timestamp}

    def record_error(self, symbol: str):
        """تسجيل خطأ لرمز معين"""
        self.error_count[symbol] += 1
        if self.error_count[symbol] >= self.max_errors:
            self.cooldown[symbol] = time.time() + self.cooldown_sec
            logger.warning(f"⚠️ {symbol} in cooldown after {self.max_errors} errors (duration: {self.cooldown_sec/60:.0f} min)")

    def record_success(self, symbol: str):
        """تسجيل نجاح - إعادة تعيين العداد"""
        self.error_count[symbol] = 0
        if symbol in self.cooldown:
            del self.cooldown[symbol]

    def is_allowed(self, symbol: str) -> bool:
        """التحقق من إمكانية التداول على الرمز"""
        if symbol in self.cooldown:
            if time.time() < self.cooldown[symbol]:
                return False
            else:
                # انتهى cooldown
                del self.cooldown[symbol]
                self.error_count[symbol] = 0
        return True

    def get_stats(self) -> Dict[str, Any]:
        """إحصائيات الأخطاء"""
        return {
            "symbols_in_cooldown": len(self.cooldown),
            "total_errors": sum(self.error_count.values()),
            "high_error_symbols": [s for s, c in self.error_count.items() if c >= self.max_errors // 2]
        }

# ===================== 🟢 NEW: TRADE JOURNAL CLASS =====================
class TradeJournal:
    """سجل تفصيلي لتحليل أداء الصفقات"""
    def __init__(self):
        self.trade_history: List[Dict[str, Any]] = []

    def record_trade(self, trade: TradeState, exit_price: float, exit_reason: str):
        """تسجيل صفقة مكتملة"""
        entry_time = datetime.fromisoformat(trade.entry_time)
        exit_time = datetime.now(timezone.utc)
        duration_minutes = (exit_time - entry_time).total_seconds() / 60

        self.trade_history.append({
            "symbol": trade.symbol,
            "entry_price": trade.entry_price,
            "exit_price": exit_price,
            "entry_time": trade.entry_time,
            "exit_time": exit_time.isoformat(),
            "duration_minutes": duration_minutes,
            "r_multiple": trade.total_realized_r,
            "quantum_score": trade.quantum_score,
            "exit_reason": exit_reason,
            "tp1_hit": trade.tp1_hit,
            "tp2_hit": trade.tp2_hit,
            "tp3_hit": trade.tp3_hit,
            "is_paper": trade.is_paper,
        })

    def get_performance_stats(self) -> Dict[str, Any]:
        """إحصائيات الأداء الشاملة"""
        if not self.trade_history:
            return {}

        r_multiples = [t['r_multiple'] for t in self.trade_history]
        wins = [r for r in r_multiples if r > 0]
        losses = [r for r in r_multiples if r <= 0]

        win_rate = len(wins) / len(r_multiples) if r_multiples else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 0

        return {
            "total_trades": len(self.trade_history),
            "win_rate": win_rate,
            "avg_win_r": avg_win,
            "avg_loss_r": avg_loss,
            "profit_factor": profit_factor,
            "total_r": sum(r_multiples),
            "expectancy_r": np.mean(r_multiples) if r_multiples else 0,
            "best_trade_r": max(r_multiples) if r_multiples else 0,
            "worst_trade_r": min(r_multiples) if r_multiples else 0,
        }

# ===================== 🟢 NEW: VALIDATION LAYER =====================
def validate_trade_before_execution(signal: SignalData, current_price: float) -> Tuple[bool, str]:
    """التحقق من أن الإشارة ما زالت صالحة قبل التنفيذ"""
    if not CONFIG.get("ENABLE_SIGNAL_VALIDATION", True):
        return True, "ok"

    try:
        # 1. التحقق من أن السعر لم يتحرك كثيراً
        price_deviation = abs(current_price - signal['entry']) / signal['entry']
        max_deviation = CONFIG.get("MAX_PRICE_DEVIATION_PCT", 0.5) / 100
        if price_deviation > max_deviation:
            return False, f"price_moved_too_much({price_deviation*100:.2f}%)"

        # 2. التحقق من أن SL ما زال منطقياً
        if current_price < signal['sl']:
            return False, "price_below_sl"
        if current_price > signal['tp1']:
            return False, "price_above_tp1"

        # 3. التحقق من عمر الإشارة
        signal_time = datetime.fromisoformat(signal['timestamp'])
        signal_age = (datetime.now(timezone.utc) - signal_time).total_seconds()
        max_age = CONFIG.get("MAX_SIGNAL_AGE_SEC", 300)
        if signal_age > max_age:
            return False, f"signal_too_old({signal_age:.0f}s)"

        return True, "ok"

    except Exception as e:
        logger.error(f"[Validation Error] {e}")
        return False, f"validation_error: {e}"

# ===================== GLOBAL STATE =====================
HTTP_SESSION: Optional[aiohttp.ClientSession] = None
ACTIVE_TRADES: Dict[str, TradeState] = {}
PENDING_ORDERS: Dict[str, Order] = {}
API_SEMAPHORE: Optional[asyncio.Semaphore] = None
TRADES_LOCK: Optional[asyncio.Lock] = None  # 🟢 NEW: Lock للحماية من race conditions
PRICE_FEED: Optional[PriceFeed] = None  # 🟢 NEW: مصدر أسعار موحد
ERROR_TRACKER: Optional[ErrorTracker] = None  # 🟢 NEW: تتبع الأخطاء
TRADE_JOURNAL: Optional[TradeJournal] = None  # 🟢 NEW: سجل الصفقات

# ========== ميزات مؤسسية جديدة ==========
BTC_TREND: Optional[Dict] = None
BTC_LAST_CHECK: float = 0
LOSS_COOLDOWN: Dict[str, float] = {}  # {symbol: cooldown_until} بعد صفقة خاسرة

PERF_SIGNAL_GEN = PerformanceMonitor(window=200)
PERF_API_CALL = PerformanceMonitor(window=500)
PERF_ORDER_EXEC = PerformanceMonitor(window=100)

STATS: Dict[str, Any] = {
    "signals_generated": 0,
    "scalp_signals": 0,
    "trades_won": 0,
    "trades_lost": 0,
    "trades_partial": 0,
    "total_r_multiple": 0.0,
    "tp1_hits": 0,
    "tp2_hits": 0,
    "tp3_hits": 0,
    "daily_trades_count": 0,
    "api_errors": 0,
    "avg_score": 0.0,
    "live_orders_placed": 0,
    "live_orders_filled": 0,
    "loop_count": 0,
    "last_reset_date": None,
    "daily_r": 0.0,
    "paper_trades_attempted": 0,
    "paper_trades_filled": 0,
    "paper_trades_partial": 0,
    "paper_trades_cancelled": 0,
}

LAST_SIGNAL_TIME: Dict[str, float] = {}
LAST_BOS_INDEX: Dict[str, int] = {}
_15M_CACHE: Dict[str, Tuple[float, pd.DataFrame]] = {}  # 🟢 سيتم تنظيفه دورياً

SHUTDOWN_REQUESTED = False

# ===================== HELPER FUNCTIONS =====================
def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and not np.isfinite(value):
            return default
        result = float(value)
        if not np.isfinite(result):
            return default
        return result
    except (TypeError, ValueError):
        return default

def validate_price(price: float) -> bool:
    if not isinstance(price, (int, float)):
        return False
    if not (np.isfinite(price) and price > 0):
        return False
    if price > 1e8:
        return False
    return True

def escape_html(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text)
    escape_chars = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
    }
    for char, escape in escape_chars.items():
        text = text.replace(char, escape)
    return text

def is_live_trading_enabled() -> bool:
    return bool(
        CONFIG.get("ENABLE_LIVE_TRADING", False) and
        CONFIG.get("MEXC_API_KEY") and
        CONFIG.get("MEXC_API_SECRET")
    )

def is_paper_trading_enabled() -> bool:
    if is_live_trading_enabled():
        return False
    return bool(CONFIG.get("PAPER_TRADING_MODE", False))

def get_execution_mode() -> ExecutionMode:
    if is_live_trading_enabled():
        return ExecutionMode.LIVE
    elif is_paper_trading_enabled():
        return ExecutionMode.PAPER
    else:
        return ExecutionMode.SIGNAL

def utc_date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def reset_daily_counters_if_needed():
    today = utc_date_str()
    if STATS.get("last_reset_date") != today:
        STATS["last_reset_date"] = today
        STATS["daily_trades_count"] = 0
        STATS["daily_r"] = 0.0
        logger.info(f"✅ Daily counters reset for {today}")

def daily_circuit_breaker_ok() -> Tuple[bool, str]:
    reset_daily_counters_if_needed()

    if CONFIG.get("ENABLE_DAILY_MAX_LOSS", True):
        max_loss_r = safe_float(CONFIG.get("DAILY_MAX_LOSS_R", -3.0), -3.0)
        if STATS.get("daily_r", 0.0) <= max_loss_r:
            return False, f"daily_max_loss_reached(R={STATS.get('daily_r', 0.0):.2f} <= {max_loss_r})"

    max_trades = int(CONFIG.get("DAILY_MAX_TRADES", 15))
    if STATS.get("daily_trades_count", 0) >= max_trades:
        return False, "daily_trade_limit_reached"

    return True, "ok"

def update_daily_stats(r_multiple: float):
    """تحديث إحصائيات اليوم بعد إغلاق صفقة"""
    STATS["daily_r"] = STATS.get("daily_r", 0.0) + r_multiple
    STATS["daily_trades_count"] = STATS.get("daily_trades_count", 0) + 1
    STATS["total_r_multiple"] = STATS.get("total_r_multiple", 0.0) + r_multiple
    if r_multiple > 0:
        STATS["trades_won"] = STATS.get("trades_won", 0) + 1
    else:
        STATS["trades_lost"] = STATS.get("trades_lost", 0) + 1
    logger.debug(f"Daily stats updated: daily_r={STATS['daily_r']:.2f}, trades={STATS['daily_trades_count']}")

def mark_shutdown():
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    logger.info("🛑 Shutdown requested")

@contextmanager
def log_exceptions(context: str = ""):
    try:
        yield
    except Exception as e:
        logger.error(f"[{context}] {str(e)}")
        logger.debug(traceback.format_exc())
        PERF_API_CALL.record_error(f"[{context}] {str(e)}")

# 🟢 FIXED: safe_api_call now accepts a callable that returns a coroutine (to avoid reusing the same coroutine object)
async def safe_api_call(coro_func: Callable, *, label: str = "", retries: int = 2, base_delay: float = 0.3):
    start_time = time.time()
    last_exc = None

    for attempt in range(retries + 1):
        try:
            # call the function to get a new coroutine each time
            coro = coro_func()
            result = await coro
            duration = time.time() - start_time
            PERF_API_CALL.record(duration)
            return result

        except asyncio.CancelledError:
            # 🟢 Colab stability: allow cancellations to propagate cleanly
            raise
        except Exception as e:
            last_exc = e
            STATS["api_errors"] = STATS.get("api_errors", 0) + 1
            PERF_API_CALL.record_error(f"{label}: {str(e)}")

            if attempt >= retries:
                logger.error(f"[API Error] {label}: {str(e)}")
                raise

            delay = base_delay * (2 ** attempt) + random.random() * 0.1
            if CONFIG.get("DEBUG_MODE"):
                logger.debug(f"[API Retry] {label} attempt={attempt+1}/{retries} delay={delay:.2f}s")
            await asyncio.sleep(delay)

    raise last_exc

# ===================== PERSISTENCE =====================
def load_state():
    global LAST_SIGNAL_TIME, LAST_BOS_INDEX, LOSS_COOLDOWN
    state_file = CONFIG["STATE_FILE_PATH"]

    if os.path.exists(state_file):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            LAST_SIGNAL_TIME = {k: float(v) for k, v in data.get("last_signal_time", {}).items()}
            LAST_BOS_INDEX = {k: int(v) for k, v in data.get("last_bos_index", {}).items()}
            LOSS_COOLDOWN = {k: float(v) for k, v in data.get("loss_cooldown", {}).items()}
            logger.info(f"✅ State loaded from {state_file}: {len(LAST_SIGNAL_TIME)} signals, {len(LAST_BOS_INDEX)} BOS indices, {len(LOSS_COOLDOWN)} loss cooldowns")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    else:
        logger.info("No state file found, starting fresh.")

def save_state():
    state_file = CONFIG["STATE_FILE_PATH"]
    try:
        data = {
            "last_signal_time": LAST_SIGNAL_TIME,
            "last_bos_index": LAST_BOS_INDEX,
            "loss_cooldown": LOSS_COOLDOWN,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        if CONFIG.get("DEBUG_MODE"):
            logger.debug(f"State saved to {state_file}")
    except Exception as e:
        logger.error(f"Failed to save state: {e}")

# ===================== 🟢 NEW: CACHE CLEANUP =====================
async def cleanup_cache():
    """تنظيف دوري لـ _15M_CACHE لمنع memory leak"""
    global _15M_CACHE

    try:
        current_time = time.time()
        cache_ttl = 3600  # ساعة واحدة

        before_count = len(_15M_CACHE)
        _15M_CACHE = {
            symbol: (timestamp, df)
            for symbol, (timestamp, df) in _15M_CACHE.items()
            if current_time - timestamp < cache_ttl
        }
        after_count = len(_15M_CACHE)

        if before_count > after_count:
            logger.info(f"🧹 Cache cleanup: removed {before_count - after_count} stale entries (remaining: {after_count})")

    except Exception as e:
        logger.error(f"Cache cleanup error: {e}")

# ===================== MEMORY CLEANUP =====================
async def cleanup_stale_symbols(valid_symbols: set):
    global LAST_SIGNAL_TIME, LAST_BOS_INDEX, ACTIVE_TRADES, PENDING_ORDERS, _15M_CACHE, LOSS_COOLDOWN

    try:
        before_signal = len(LAST_SIGNAL_TIME)
        before_bos = len(LAST_BOS_INDEX)
        before_trades = len(ACTIVE_TRADES)
        before_orders = len(PENDING_ORDERS)
        before_cache = len(_15M_CACHE)
        before_loss = len(LOSS_COOLDOWN)

        LAST_SIGNAL_TIME = {k: v for k, v in LAST_SIGNAL_TIME.items() if k in valid_symbols}
        LAST_BOS_INDEX = {k: v for k, v in LAST_BOS_INDEX.items() if k in valid_symbols}
        _15M_CACHE = {k: v for k, v in _15M_CACHE.items() if k in valid_symbols}
        LOSS_COOLDOWN = {k: v for k, v in LOSS_COOLDOWN.items() if k in valid_symbols}

        cleanup_days = CONFIG.get("STALE_TRADE_CLEANUP_DAYS", 7)
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=cleanup_days)).isoformat()

        async with TRADES_LOCK:
            ACTIVE_TRADES = {
                k: v for k, v in ACTIVE_TRADES.items()
                if v.entry_time > cutoff_date
            }

        PENDING_ORDERS = {k: v for k, v in PENDING_ORDERS.items() if k in valid_symbols}

        removed_signal = before_signal - len(LAST_SIGNAL_TIME)
        removed_bos = before_bos - len(LAST_BOS_INDEX)
        removed_trades = before_trades - len(ACTIVE_TRADES)
        removed_orders = before_orders - len(PENDING_ORDERS)
        removed_cache = before_cache - len(_15M_CACHE)
        removed_loss = before_loss - len(LOSS_COOLDOWN)

        if any([removed_signal, removed_bos, removed_trades, removed_orders, removed_cache, removed_loss]):
            logger.info(
                f"🧹 Cleanup: removed {removed_signal} signals, {removed_bos} BOS, "
                f"{removed_trades} stale trades, {removed_orders} pending orders, {removed_cache} cache entries, {removed_loss} loss cooldowns"
            )

    except Exception as e:
        logger.error(f"Cleanup error: {e}")

# ===================== MARKET HOURS FILTER =====================
def is_good_scalping_time() -> bool:
    if not CONFIG.get("ENABLE_MARKET_HOURS_FILTER", True):
        return True
    current_hour = datetime.now(timezone.utc).hour
    best_hours = CONFIG.get("BEST_HOURS_UTC", [])
    return current_hour in best_hours

# ===================== INDICATORS =====================
def _normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['t', 'open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            df[col] = np.nan
    df = df.sort_values('t', ascending=True)
    try:
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True, errors='coerce')
    except Exception:
        df['timestamp'] = pd.NaT
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = df[df['close'] > 0].copy()
    df['volume'] = df['volume'].fillna(0)
    df = df.reset_index(drop=True)
    return df

def calculate_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) < 250:
        logger.warning(f"Insufficient data for EMA200: {len(df) if df is not None else 0}")
        return None
    try:
        df = _normalize_ohlcv_df(df)
        if len(df) < 250:
            return None
        df['ema20'] = ta.trend.ema_indicator(df['close'], 20)
        df['ema50'] = ta.trend.ema_indicator(df['close'], 50)
        df['ema200'] = ta.trend.ema_indicator(df['close'], 200)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
        df['atr_pct'] = (df['atr'] / df['close'].replace(0, np.nan)) * 100
        df['rsi'] = ta.momentum.rsi(df['close'], 14)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], 14)
        df['volume_sma'] = df['volume'].rolling(20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan)
        bb = ta.volatility.BollingerBands(df['close'], 20, 2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close'].replace(0, np.nan)
        df = df.ffill().bfill()
        return df
    except Exception as e:
        logger.error(f"[Indicators Error] {str(e)}")
        PERF_SIGNAL_GEN.record_error(f"Indicators: {str(e)}")
        return None

# ===================== 🟢 MARKET REGIME FILTER =====================
def classify_market_regime(df: pd.DataFrame) -> str:
    """
    تصنيف حالة السوق إلى:
    - TRENDING
    - NEUTRAL
    - VOLATILE
    """
    if df is None or len(df) < 14:
        return "NEUTRAL"

    last = df.iloc[-1]
    adx = safe_float(last.get('adx'), 0)
    atr_pct = safe_float(last.get('atr_pct'), 0)
    min_adx = CONFIG.get("MIN_ADX_FOR_TREND", 20)

    if adx >= min_adx:
        return "TRENDING"
    elif atr_pct > 3.0:  # إذا كان التقلب عالي بدون اتجاه
        return "VOLATILE"
    else:
        return "NEUTRAL"

# ===================== 🟢 BTC FILTER =====================
async def check_btc_trend(exchange) -> Dict:
    global BTC_TREND, BTC_LAST_CHECK

    if not CONFIG.get("ENABLE_BTC_FILTER", True):
        return {"trend": "NEUTRAL", "change_1h": 0, "safe_to_trade": True}

    if BTC_TREND and (time.time() - BTC_LAST_CHECK) < 300:
        return BTC_TREND

    btc_symbol = CONFIG.get("BTC_SYMBOL", "BTC/USDT")
    try:
        async with API_SEMAPHORE:
            data = await safe_api_call(
                lambda: exchange.fetch_ohlcv(btc_symbol, "1h", limit=100),
                label="fetch_btc_ohlcv"
            )

        if not data or len(data) < 20:
            return {"trend": "NEUTRAL", "change_1h": 0, "safe_to_trade": True}

        df = pd.DataFrame(data, columns=['t','open','high','low','close','volume'])
        current_price = safe_float(df['close'].iloc[-1])

        price_1h_ago = safe_float(df['close'].iloc[-2]) if len(df) >= 2 else current_price
        price_4h_ago = safe_float(df['close'].iloc[-5]) if len(df) >= 5 else price_1h_ago

        change_1h = ((current_price - price_1h_ago) / price_1h_ago) * 100 if price_1h_ago > 0 else 0
        change_4h = ((current_price - price_4h_ago) / price_4h_ago) * 100 if price_4h_ago > 0 else 0

        crash_thresh = CONFIG.get("BTC_CRASH_THRESHOLD", -3.0)
        warn_thresh = CONFIG.get("BTC_WARNING_THRESHOLD", -1.5)

        if change_1h <= crash_thresh or change_4h <= crash_thresh * 1.5:
            trend = "CRASH"
            safe_to_trade = False
        elif change_1h <= warn_thresh:
            trend = "WARNING"
            safe_to_trade = True
        elif change_1h >= -warn_thresh:
            trend = "BULLISH"
            safe_to_trade = True
        else:
            trend = "NEUTRAL"
            safe_to_trade = True

        BTC_TREND = {
            "trend": trend,
            "change_1h": round(change_1h, 2),
            "change_4h": round(change_4h, 2),
            "safe_to_trade": safe_to_trade,
            "price": current_price
        }

        BTC_LAST_CHECK = time.time()

        if trend == "CRASH":
            logger.warning(f"[BTC] 🚨 Crash detected! 1H: {change_1h:.2f}%, 4H: {change_4h:.2f}%")
            await send_telegram(
                f"⚠️ تحذير انهيار BTC\n\n"
                f"📉 التغير خلال ساعة: {change_1h:.2f}%\n"
                f"📉 التغير خلال 4 ساعات: {change_4h:.2f}%\n\n"
                f"🛑 تم إيقاف التداول مؤقتاً!",
                critical=True
            )

        return BTC_TREND

    except Exception as e:
        logger.error(f"[BTC Check Error] {str(e)[:100]}")
        return {"trend": "NEUTRAL", "change_1h": 0, "safe_to_trade": True}

# ===================== 🟢 VOLUME REGIME FILTER =====================
async def check_volume_regime(exchange, symbol: str) -> Tuple[bool, str]:
    """
    التحقق من أن الحجم اليومي أعلى من متوسط الأيام السابقة.
    """
    if not CONFIG.get("ENABLE_VOLUME_REGIME", True):
        return True, ""

    days = CONFIG.get("VOLUME_REGIME_DAYS", 5)
    min_ratio = CONFIG.get("VOLUME_REGIME_MIN_RATIO", 1.2)

    try:
        # جلب بيانات يومية
        async with API_SEMAPHORE:
            daily_data = await safe_api_call(
                lambda: exchange.fetch_ohlcv(symbol, "1d", limit=days+1),
                label=f"volume_regime_{symbol}"
            )

        if not daily_data or len(daily_data) < days:
            return True, ""  # بيانات غير كافية، نتجاهل الفلتر

        df_daily = pd.DataFrame(daily_data, columns=['t','open','high','low','close','volume'])
        # آخر يوم كامل هو الأقدم? نريد آخر شمعة مكتملة (اليوم السابق) لأن اليوم الحالي قد لا يكون مكتملاً
        # نفترض أن آخر شمعة هي اليوم الحالي (غير مكتمل)، نستخدم الأمس
        if len(df_daily) > 1:
            current_volume = df_daily.iloc[-2]['volume']  # حجم الأمس
            past_volumes = df_daily.iloc[:-2]['volume']   # الأيام قبل الأمس
        else:
            return True, ""

        if len(past_volumes) == 0:
            return True, ""

        avg_volume = past_volumes.mean()
        if avg_volume <= 0:
            return True, ""

        ratio = current_volume / avg_volume
        if ratio >= min_ratio:
            return True, f"volume_regime_ok({ratio:.2f})"
        else:
            return False, f"volume_too_low({ratio:.2f} < {min_ratio})"

    except Exception as e:
        logger.error(f"[Volume Regime Error] {symbol}: {e}")
        return True, ""  # في حالة خطأ نسمح بالمرور

# ===================== 🟢 ORDER FLOW LIGHT =====================
async def check_orderflow_light(exchange, symbol: str) -> Tuple[bool, float]:
    """
    فحص بسيط لتدفق الأوامر: imbalance من orderbook.
    ترجع (is_bullish, imbalance_value)
    """
    if not CONFIG.get("ENABLE_ORDERFLOW_LIGHT", True):
        return True, 0.0

    try:
        async with API_SEMAPHORE:
            orderbook = await safe_api_call(
                lambda: exchange.fetch_order_book(symbol, 10),
                label=f"orderflow_{symbol}"
            )

        bids = orderbook.get("bids") or []
        asks = orderbook.get("asks") or []

        if not bids or not asks:
            return True, 0.0

        # مجموع الحجم في أعلى 5 مستويات
        bid_volume = sum(bid[1] for bid in bids[:5])
        ask_volume = sum(ask[1] for ask in asks[:5])
        total = bid_volume + ask_volume
        if total == 0:
            return True, 0.0

        imbalance = (bid_volume - ask_volume) / total
        thresh = CONFIG.get("ORDERFLOW_IMBALANCE_THRESHOLD", 0.25)

        # إذا كان imbalance إيجابي كبير -> صاعد
        # إذا كان imbalance سلبي كبير -> هابط
        # نسمح فقط إذا لم يكن imbalance سلبي كبير (لأننا نشتري فقط)
        if imbalance < -thresh:
            return False, imbalance  # تدفق بيعي قوي
        return True, imbalance

    except Exception as e:
        logger.error(f"[OrderFlow Light Error] {symbol}: {e}")
        return True, 0.0

# ===================== 🟢 LOSS COOLDOWN =====================
def set_loss_cooldown(symbol: str):
    """وضع تبريد بعد صفقة خاسرة"""
    cooldown_sec = CONFIG.get("LOSS_COOLDOWN_SEC", 3600)
    if cooldown_sec > 0:
        LOSS_COOLDOWN[symbol] = time.time() + cooldown_sec
        logger.info(f"⏱️ Loss cooldown set for {symbol} until {datetime.fromtimestamp(LOSS_COOLDOWN[symbol]).isoformat()}")

def is_in_loss_cooldown(symbol: str) -> bool:
    if symbol not in LOSS_COOLDOWN:
        return False
    if time.time() < LOSS_COOLDOWN[symbol]:
        return True
    else:
        # انتهت المدة
        del LOSS_COOLDOWN[symbol]
        return False

# ===================== SCALPING FILTERS =====================
def check_market_environment_15m(df_15m: pd.DataFrame) -> Tuple[bool, str]:
    if df_15m is None or len(df_15m) < 200:
        return False, "insufficient_data"
    try:
        last = df_15m.iloc[-1]
        if CONFIG.get("MIN_EMA200_FILTER", True):
            if last['close'] <= last['ema200']:
                return False, "price_below_ema200"
        if CONFIG.get("MIN_EMA50_ABOVE_200", False):   # 🟢 أصبح اختيارياً
            if last['ema50'] <= last['ema200']:
                return False, "ema50_below_ema200"
        min_rsi = CONFIG.get("MIN_RSI", 45)            # 🟢 مخفف
        if last['rsi'] < min_rsi:
            return False, f"rsi_below_{min_rsi}"
        min_adx = CONFIG.get("MIN_ADX", 15)             # 🟢 مخفف
        if last['adx'] < min_adx:
            return False, f"adx_below_{min_adx}"
        return True, "ok"
    except Exception as e:
        logger.error(f"[Market Environment Filter Error] {str(e)}")
        return False, "filter_error"

# ===================== 🟢 ENHANCED: MICRO BOS DETECTION =====================
def detect_micro_bos_at_index(df: pd.DataFrame, idx: int) -> Optional[MicroBOS]:
    """
    🟢 ENHANCED: تحسين الكشف عن Micro BOS مع تأكيد الإغلاق فوق القمة
    """
    if df is None or len(df) < 10 or idx < 2 or idx >= len(df):
        return None
    try:
        lookback = CONFIG.get("MICRO_BOS_LOOKBACK", 5)
        min_break = CONFIG.get("MICRO_BOS_MIN_BREAK", 0.001)
        require_close_confirmation = CONFIG.get("MICRO_BOS_CLOSE_CONFIRMATION", True)

        start = max(0, idx - lookback - 3)
        end = idx
        highs = []

        # البحث عن القمم
        for i in range(start, end):
            if i < 1 or i + 1 >= len(df):
                continue
            current_high = df.iloc[i]['high']
            prev_high = df.iloc[i - 1]['high']
            next_high = df.iloc[i + 1]['high']
            if current_high > prev_high and current_high > next_high:
                highs.append((i, current_high))

        if len(highs) < 2:
            return None

        valid_highs = [(i, h) for i, h in highs if i < idx]
        if len(valid_highs) < 2:
            return None

        last_high_idx, last_high = valid_highs[-1]
        prev_high = valid_highs[-2][1]

        current = df.iloc[idx]

        # التحقق من كسر القمة
        break_pct = (current['high'] - last_high) / last_high
        if break_pct < min_break:
            return None

        # 🟢 NEW: تأكيد الإغلاق فوق القمة
        if require_close_confirmation:
            if current['close'] <= last_high:
                return None

            # 🟢 NEW: التأكد من أن الجسم قوي (ليس فتيل فقط)
            body = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            if total_range > 0:
                body_ratio = body / total_range
                if body_ratio < 0.3:  # الجسم أقل من 30% من الشمعة
                    return None

                # 🟢 NEW: التحقق من عدم وجود فتيل علوي طويل جداً
                upper_wick = current['high'] - max(current['open'], current['close'])
                if upper_wick > body * 2:  # فتيل أطول من ضعف الجسم
                    return None

        # حساب القوة
        avg_volume = df['volume'].iloc[max(0, idx - 10):idx].mean()
        volume_spike = current['volume'] > avg_volume * 1.2 if avg_volume and avg_volume > 0 else False

        strength = min(100, break_pct * 10000)
        if volume_spike:
            strength = min(100, strength * 1.2)

        return MicroBOS(
            detected=True,
            break_price=float(current['high']),
            previous_high=float(last_high),
            break_candle_index=int(idx),
            strength=float(strength),
            volume_spike=bool(volume_spike)
        )
    except Exception as e:
        logger.error(f"[Micro BOS Detection Error at {idx}] {str(e)}")
        return None

# ===================== RETEST ZONE DETECTION =====================
def find_retest_zone_at_index(df: pd.DataFrame, micro_bos: MicroBOS, bos_idx: int) -> Optional[RetestZone]:
    if df is None or micro_bos is None:
        return None
    try:
        start = max(0, bos_idx - CONFIG.get("ORDER_BLOCK_LOOKBACK", 10))
        for i in range(start, bos_idx):
            candle = df.iloc[i]
            if candle['close'] <= candle['open']:
                continue
            body = candle['close'] - candle['open']
            total_range = candle['high'] - candle['low']
            if total_range == 0:
                continue
            body_ratio = body / total_range
            if body_ratio < 0.6:
                continue
            if i > 0:
                prev_volume = df.iloc[i - 1]['volume']
                if candle['volume'] <= prev_volume:
                    continue
            zone_low = float(min(candle['open'], candle['close']))
            zone_high = float(max(candle['open'], candle['close']))
            zone_mid = float((zone_low + zone_high) / 2)
            freshness = int(bos_idx - i)
            return RetestZone(
                zone_type=ZoneType.ORDER_BLOCK.value,
                low=zone_low,
                high=zone_high,
                mid=zone_mid,
                freshness=freshness
            )
        last = df.iloc[-1]
        ema20 = float(last['ema20'])
        atr = float(last['atr'])
        return RetestZone(
            zone_type=ZoneType.EMA.value,
            low=float(ema20 - atr * 0.3),
            high=float(ema20 + atr * 0.3),
            mid=float(ema20),
            freshness=0
        )
    except Exception as e:
        logger.error(f"[Retest Zone Detection Error] {str(e)}")
        return None

# ===================== INSTITUTIONAL FILTERS =====================
def detect_liquidity_sweep_at_index(df: pd.DataFrame, zone: RetestZone, idx: int) -> bool:
    if not CONFIG.get("ENABLE_LIQUIDITY_SWEEP_FILTER", True):
        return True
    # 🟢 لم يعد إلزامياً، لكن الدالة موجودة للتعزيز
    if df is None or zone is None or idx < 1 or len(df) < 3:
        return False
    try:
        current = df.iloc[idx]
        lookback = int(CONFIG.get("LIQUIDITY_SWEEP_LOOKBACK", 5))
        window = df.iloc[max(0, idx - lookback):idx]
        if len(window) == 0:
            return False
        prev_lows = window['low'].values
        if len(prev_lows) == 0:
            return False
        swept_level = float(np.min(prev_lows))
        swept = (current['low'] < swept_level) and (current['close'] > swept_level)
        in_zone_context = (current['low'] <= zone.high) and (current['high'] >= zone.low)
        return bool(swept and in_zone_context)
    except Exception as e:
        logger.error(f"[Liquidity Sweep Error] {str(e)}")
        return False

def check_minimum_pullback_at_index(df: pd.DataFrame, micro_bos: MicroBOS, zone: RetestZone, bos_idx: int, retest_idx: int) -> bool:
    if df is None or micro_bos is None or zone is None or bos_idx >= retest_idx:
        return False
    try:
        last = df.iloc[retest_idx]
        atr = safe_float(last.get("atr", 0.0), 0.0)
        if atr <= 0:
            return False
        min_pb = atr * safe_float(CONFIG.get("MIN_PULLBACK_ATR", 0.3), 0.3)
        segment = df.iloc[bos_idx + 1:retest_idx + 1]
        if len(segment) == 0:
            return False
        post_bos_low = float(segment['low'].min())
        pullback_distance = float(micro_bos.previous_high - post_bos_low)
        return pullback_distance >= min_pb
    except Exception as e:
        logger.error(f"[Minimum Pullback Error] {str(e)}")
        return False

# ===================== ENTRY SIGNAL DETECTION =====================
def detect_scalp_entry_at_index(df: pd.DataFrame, retest_zone: RetestZone, idx: int) -> Optional[ScalpEntry]:
    if df is None or retest_zone is None or idx < 1:
        return None
    try:
        last = df.iloc[idx]
        prev = df.iloc[idx - 1]
        price_in_zone = (
            last['low'] <= retest_zone.high and
            last['high'] >= retest_zone.low
        )
        if not price_in_zone:
            return None
        is_bullish = last['close'] > last['open']
        body = abs(last['close'] - last['open'])
        total_range = last['high'] - last['low']
        if total_range == 0:
            return None
        lower_wick = min(last['open'], last['close']) - last['low']
        wick_ratio = lower_wick / total_range
        bullish_engulfing = (
            is_bullish and
            prev['close'] < prev['open'] and
            last['close'] > prev['open'] and
            last['open'] < prev['close']
        )
        hammer = (
            is_bullish and
            wick_ratio >= CONFIG.get("LG_WICK_MIN_RATIO", 0.4) and
            body / total_range >= 0.3
        )
        strong_rejection = (
            is_bullish and
            wick_ratio >= 0.5 and
            last['close'] > retest_zone.mid
        )
        rejection_wick = bool(hammer or strong_rejection)
        if not (bullish_engulfing or hammer or strong_rejection):
            return None
        avg_volume = df['volume'].iloc[max(0, idx - 20):idx].mean()
        volume_ratio = last['volume'] / avg_volume if avg_volume and avg_volume > 0 else 1.0
        min_volume_ratio = CONFIG.get("MIN_VOLUME_RATIO", 0.8)
        volume_ok = volume_ratio >= min_volume_ratio
        if not volume_ok:
            return None
        if retest_zone.zone_type == ZoneType.ORDER_BLOCK.value:
            base_entry = float(retest_zone.mid * 1.0003)
        else:
            base_entry = float(last['close'])
        entry_price = float(base_entry * (1 + CONFIG["ENTRY_SLIPPAGE_BUFFER"]))
        return ScalpEntry(
            detected=True,
            entry_price=entry_price,
            entry_type=f"RETEST_{retest_zone.zone_type}",
            retest_zone=retest_zone,
            confirmation_candle={
                'open': float(last['open']),
                'high': float(last['high']),
                'low': float(last['low']),
                'close': float(last['close']),
                'volume': float(last['volume'])
            },
            volume_ok=bool(volume_ok),
            rejection_wick=rejection_wick,
            liquidity_sweep=False,
            pullback_ok=False,
            bos_index=-1,
            retest_index=idx
        )
    except Exception as e:
        logger.error(f"[Scalp Entry Detection Error at {idx}] {str(e)}")
        return None

# ===================== ENTRY QUALITY FILTER =====================
def entry_quality_filter_at_index(df: pd.DataFrame, entry_signal: ScalpEntry, idx: int) -> Tuple[bool, str]:
    if not CONFIG.get("ENABLE_ENTRY_QUALITY_FILTER", True):
        return True, ""
    try:
        if df is None or idx < 1:
            return False, "no_data"
        last = df.iloc[idx]
        prev = df.iloc[idx - 1]
        atr_pct = safe_float(last.get("atr_pct", 0.0), 0.0)
        max_atr = CONFIG.get("ENTRY_QUALITY_MAX_ATR_PCT_5M", 4.5)   # 🟢 مخفف
        if atr_pct > max_atr:
            return False, f"atr_pct_too_high({atr_pct:.2f}%)"
        bb_width = safe_float(last.get("bb_width", 0.0), 0.0)
        max_bb = CONFIG.get("ENTRY_QUALITY_MAX_BB_WIDTH_5M", 0.08)  # 🟢 مخفف
        if bb_width > max_bb:
            return False, f"bb_width_too_high({bb_width:.3f})"
        # 🟢 تم إزالة فلتر Momentum لأنه يقتل فرص السكالبينج
        # price_change = (last['close'] - prev['close']) / prev['close']
        # max_momentum = CONFIG.get("MAX_MOMENTUM_CANDLE_PCT", 0.015)
        # if abs(price_change) > max_momentum:
        #     return False, "momentum_too_fast"
        if entry_signal.retest_zone and 'atr' in df.columns:
            zone_mid = float(entry_signal.retest_zone.mid)
            current_price = float(last['close'])
            atr = float(last['atr'])
            distance = abs(current_price - zone_mid)
            max_distance = atr * CONFIG.get("ENTRY_QUALITY_MAX_DISTANCE_FROM_ZONE_ATR", 1.2)  # 🟢 مخفف
            if distance > max_distance:
                return False, f"too_far_from_zone({distance:.6f})"
        return True, ""
    except Exception as e:
        logger.error(f"[Entry Quality Filter Error] {str(e)}")
        return True, ""

# ===================== RISK CALCULATION =====================
def calculate_risk_levels(
    entry: float,
    atr: float,
    retest_zone: Optional[RetestZone]
) -> Tuple[float, float, float, float]:
    if not validate_price(entry) or entry <= 0:
        return 0.0, 0.0, 0.0, 0.0
    if not validate_price(atr) or atr <= 0:
        atr = entry * 0.01
    if retest_zone:
        sl = float(retest_zone.low * 0.998)
    else:
        sl = float(entry - (atr * CONFIG["ATR_SL_MULT"]))
    max_sl_distance = float(entry * (CONFIG["MAX_SL_PCT"] / 100))
    hard_sl = float(entry - max_sl_distance)
    if sl < hard_sl:
        sl = hard_sl
    if sl <= 0 or sl >= entry:
        sl = float(entry * 0.995)
    risk = float(entry - sl)
    if risk <= 0:
        return 0.0, 0.0, 0.0, 0.0
    tp1 = float(entry + (risk * CONFIG["TP1_RR"]))
    tp2 = float(entry + (risk * CONFIG["TP2_RR"]))
    tp3 = float(entry + (risk * CONFIG["TP3_RR"]))
    return sl, tp1, tp2, tp3

# ===================== POSITION SIZING =====================
def calculate_position_size(entry: float, sl: float) -> Tuple[float, float]:
    if not validate_price(entry) or not validate_price(sl) or sl >= entry:
        return 0.0, 0.0
    account_size = float(CONFIG["ACCOUNT_SIZE_USDT"])
    risk_amount = float(account_size * (CONFIG["RISK_PER_TRADE_PCT"] / 100))
    risk_per_unit = float(entry - sl)
    if risk_per_unit <= 0:
        return 0.0, 0.0
    position_size = float(risk_amount / risk_per_unit)
    position_value = float(position_size * entry)
    min_size = float(CONFIG["MIN_POSITION_SIZE_USDT"])
    max_size = float(CONFIG["MAX_POSITION_SIZE_USDT"])
    if position_value < min_size:
        return 0.0, 0.0
    if position_value > max_size:
        position_value = max_size
        position_size = position_value / entry
    fee_rate = CONFIG.get("FEE_RATE", 0.001)
    position_value_after_fee = position_value * (1 - fee_rate)
    position_size_after_fee = position_value_after_fee / entry
    position_pct = float((position_value_after_fee / account_size) * 100) if account_size > 0 else 0.0
    return position_value_after_fee, position_pct

def calculate_net_pnl(entry_value: float, exit_value: float) -> float:
    """حساب صافي الربح بعد خصم رسوم الدخول والخروج"""
    fee_rate = CONFIG.get("FEE_RATE", 0.001)
    exit_after_fee = exit_value * (1 - fee_rate)
    return exit_after_fee - entry_value

def calculate_r_multiple_from_pnl(pnl: float, risk_amount: float) -> float:
    """حساب R multiple من الربح الفعلي"""
    if risk_amount <= 0:
        return 0.0
    return pnl / risk_amount

# ===================== QUANTUM SCORING (محسّن) =====================
def calculate_quantum_score(
    micro_bos: MicroBOS,
    entry_signal: ScalpEntry,
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
    retest_idx: int,
    orderflow_imbalance: float = 0.0,  # 🟢 إضافة
    volume_regime_ok: bool = True,      # 🟢 إضافة
    market_regime: str = "NEUTRAL"      # 🟢 إضافة
) -> float:
    score = 0.0
    last_15m = df_15m.iloc[-1]

    # أساسيات
    if last_15m['close'] > last_15m['ema200']:
        score += 10
    if last_15m['ema50'] > last_15m['ema200']:
        score += 10
    if last_15m['rsi'] > 50:
        score += 5
    if last_15m['adx'] > 25:
        score += 5

    # Micro BOS
    score += min(30, float(micro_bos.strength) * 0.3)
    if micro_bos.volume_spike:
        score += 5

    # Entry signal
    if entry_signal.rejection_wick:
        score += 15
    if entry_signal.volume_ok:
        score += 10
    if entry_signal.retest_zone and entry_signal.retest_zone.zone_type == ZoneType.ORDER_BLOCK.value:
        score += 5
    if entry_signal.liquidity_sweep:
        score += 8
    if entry_signal.pullback_ok:
        score += 7

    # EMA alignment على 5m
    if retest_idx >= 0 and df_5m.iloc[retest_idx]['ema20'] > df_5m.iloc[retest_idx]['ema50']:
        score += 10

    # 🟢 عوامل جديدة
    if orderflow_imbalance > CONFIG.get("ORDERFLOW_IMBALANCE_THRESHOLD", 0.25):
        score += 8
    if volume_regime_ok:
        score += 5
    if market_regime == "TRENDING":
        score += 15
    elif market_regime == "VOLATILE":
        score -= 5  # خصم في حالة التقلب العالي

    return float(min(100, max(0, score)))

# ===================== 🟢 ENHANCED: SPREAD FILTER =====================
async def check_spread(exchange, symbol: str) -> Tuple[bool, float]:
    """
    🟢 ENHANCED: حساب Spread بـ mid price بدلاً من bid
    """
    try:
        async with API_SEMAPHORE:
            ticker = await safe_api_call(lambda: exchange.fetch_ticker(symbol), label=f"fetch_ticker({symbol})")
        bid = ticker.get('bid')
        ask = ticker.get('ask')
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return False, 999.0

        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid * 100

        if spread_pct > CONFIG["MAX_SPREAD_PCT"]:
            return False, spread_pct
        return True, float(spread_pct)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"[Spread Check Error] {symbol}: {str(e)}")
        return False, 999.0

# ===================== SYMBOL FILTERING =====================
async def filter_symbols_by_liquidity(exchange) -> List[str]:
    try:
        logger.info("Fetching tickers for liquidity filtering...")
        async with API_SEMAPHORE:
            tickers = await safe_api_call(lambda: exchange.fetch_tickers(), label="fetch_tickers()")
        if not tickers:
            logger.warning("No tickers received, using all symbols")
            all_symbols = [s for s in exchange.markets.keys() if s.endswith(CONFIG["QUOTE"])]
            return all_symbols
        filtered = []
        min_vol = float(CONFIG["MIN_24H_VOLUME"])
        max_spread = float(CONFIG["MAX_SPREAD_PCT"])
        for symbol, ticker in tickers.items():
            if not symbol.endswith(CONFIG["QUOTE"]):
                continue
            quote_volume = ticker.get('quoteVolume')
            if quote_volume is None:
                quote_volume = ticker.get('quoteVolume24h')
            if quote_volume is None:
                quote_volume = ticker.get('baseVolume')
            if quote_volume is None:
                quote_volume = ticker.get('volume')
            quote_volume = safe_float(quote_volume, 0.0)
            if quote_volume < min_vol:
                continue
            bid = safe_float(ticker.get('bid'), 0.0)
            ask = safe_float(ticker.get('ask'), 0.0)
            if bid <= 0 or ask <= 0:
                continue

            mid = (bid + ask) / 2
            spread_pct = (ask - bid) / mid * 100

            if spread_pct > max_spread:
                continue
            filtered.append((symbol, quote_volume))
        filtered.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [s[0] for s in filtered[:int(CONFIG["MAX_SYMBOLS_TO_TRADE"])]]
        logger.info(f"✅ Filtered symbols: {len(top_symbols)} (from {len(tickers)})")
        return top_symbols
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"[Symbol Filtering Error] {str(e)}")
        all_symbols = [s for s in exchange.markets.keys() if s.endswith(CONFIG["QUOTE"])]
        return all_symbols

# ===================== SIGNAL GENERATOR (محسّن) =====================
async def generate_scalp_signal(exchange, symbol: str) -> Optional[SignalData]:
    start_time = time.time()
    try:
        if ERROR_TRACKER and not ERROR_TRACKER.is_allowed(symbol):
            if CONFIG.get("DEBUG_MODE"):
                logger.debug(f"[{symbol}] In error cooldown")
            return None

        # 🟢 Loss cooldown
        if is_in_loss_cooldown(symbol):
            if CONFIG.get("DEBUG_MODE"):
                logger.debug(f"[{symbol}] In loss cooldown")
            return None

        if not is_good_scalping_time():
            return None
        ok_cb, reason_cb = daily_circuit_breaker_ok()
        if not ok_cb:
            if CONFIG.get("DEBUG_MODE"):
                logger.debug(f"[{symbol}] Circuit breaker: {reason_cb}")
            return None
        now = time.time()
        if symbol in LAST_SIGNAL_TIME:
            if now - LAST_SIGNAL_TIME[symbol] < CONFIG["MIN_SIGNAL_INTERVAL_SEC"]:
                if CONFIG.get("DEBUG_MODE"):
                    logger.debug(f"[{symbol}] Cooldown active")
                return None

        cached = _15M_CACHE.get(symbol)
        cache_age = time.time() - cached[0] if cached else None
        if cached and cache_age is not None and cache_age < 7 * 60:   # 🟢 7 دقائق بدلاً من 14
            df_15m = cached[1]
            logger.debug(f"[{symbol}] Using cached 15m data ({cache_age/60:.1f} min old)")
        else:
            async with API_SEMAPHORE:
                data_15m = await safe_api_call(
                    lambda: exchange.fetch_ohlcv(symbol, CONFIG["TF_ANALYSIS"], limit=300),
                    label=f"fetch_ohlcv({symbol},{CONFIG['TF_ANALYSIS']})"
                )
            if not data_15m:
                if ERROR_TRACKER:
                    ERROR_TRACKER.record_error(symbol)
                return None
            df_15m = calculate_indicators(pd.DataFrame(data_15m, columns=['t', 'open', 'high', 'low', 'close', 'volume']))
            if df_15m is not None:
                _15M_CACHE[symbol] = (time.time(), df_15m)

        async with API_SEMAPHORE:
            data_5m = await safe_api_call(
                lambda: exchange.fetch_ohlcv(symbol, CONFIG["TF_ENTRY"], limit=300),
                label=f"fetch_ohlcv({symbol},{CONFIG['TF_ENTRY']})"
            )
        if not data_5m:
            if ERROR_TRACKER:
                ERROR_TRACKER.record_error(symbol)
            return None
        df_5m = calculate_indicators(pd.DataFrame(data_5m, columns=['t', 'open', 'high', 'low', 'close', 'volume']))
        if df_15m is None or df_5m is None:
            return None

        # 🟢 Market Regime Filter
        if CONFIG.get("ENABLE_MARKET_REGIME_FILTER", True):
            regime = classify_market_regime(df_15m)
            if regime != "TRENDING":
                if CONFIG.get("DEBUG_MODE"):
                    logger.debug(f"[{symbol}] Market regime {regime} not TRENDING")
                return None
        else:
            regime = "NEUTRAL"

        env_ok, env_reason = check_market_environment_15m(df_15m)
        if not env_ok:
            if CONFIG.get("DEBUG_MODE"):
                logger.debug(f"[{symbol}] Environment: {env_reason}")
            return None

        # 🟢 Volume Regime Filter
        volume_ok, vol_reason = await check_volume_regime(exchange, symbol)
        if not volume_ok:
            if CONFIG.get("DEBUG_MODE"):
                logger.debug(f"[{symbol}] Volume regime: {vol_reason}")
            return None

        spread_ok, spread_pct = await check_spread(exchange, symbol)
        if not spread_ok:
            if CONFIG.get("DEBUG_MODE"):
                logger.debug(f"[{symbol}] Spread: {spread_pct:.2f}%")
            return None

        lookahead = int(CONFIG["POST_BOS_LOOKAHEAD_CANDLES"])
        total_candles = len(df_5m) - 1
        start_idx = max(0, total_candles - lookahead)

        for bos_candidate_idx in range(start_idx, total_candles):
            if bos_candidate_idx >= total_candles - 1:
                continue
            if symbol in LAST_BOS_INDEX:
                if bos_candidate_idx <= LAST_BOS_INDEX[symbol]:
                    continue

            micro_bos = detect_micro_bos_at_index(df_5m, bos_candidate_idx)
            if not micro_bos or not micro_bos.detected:
                continue

            retest_zone = find_retest_zone_at_index(df_5m, micro_bos, bos_candidate_idx)
            if not retest_zone:
                continue

            for retest_idx in range(bos_candidate_idx + 1, total_candles):
                pullback_ok = check_minimum_pullback_at_index(df_5m, micro_bos, retest_zone, bos_candidate_idx, retest_idx)
                if not pullback_ok:
                    continue

                entry_signal = detect_scalp_entry_at_index(df_5m, retest_zone, retest_idx)
                if not entry_signal or not entry_signal.detected:
                    continue

                liquidity_sweep = detect_liquidity_sweep_at_index(df_5m, retest_zone, retest_idx)
                # 🟢 لم نعد نمنع الصفقة إذا لم يوجد liquidity sweep، بل نضيفه كعامل تعزيز
                # if CONFIG.get("LIQUIDITY_SWEEP_REQUIRED", True) and not liquidity_sweep:
                #     continue

                entry_signal.liquidity_sweep = liquidity_sweep
                entry_signal.pullback_ok = pullback_ok
                entry_signal.bos_index = bos_candidate_idx
                entry_signal.retest_index = retest_idx

                quality_ok, quality_reason = entry_quality_filter_at_index(df_5m, entry_signal, retest_idx)
                if not quality_ok:
                    if CONFIG.get("DEBUG_MODE"):
                        logger.debug(f"[{symbol}] Quality: {quality_reason}")
                    continue

                atr = float(df_5m.iloc[retest_idx]['atr'])
                sl, tp1, tp2, tp3 = calculate_risk_levels(
                    float(entry_signal.entry_price),
                    atr,
                    retest_zone
                )
                if sl == 0:
                    continue

                position_size_usdt, position_size_pct = calculate_position_size(entry_signal.entry_price, sl)
                if position_size_usdt == 0:
                    continue

                # 🟢 Order Flow Light
                of_ok, of_imbalance = await check_orderflow_light(exchange, symbol)
                if not of_ok:
                    if CONFIG.get("DEBUG_MODE"):
                        logger.debug(f"[{symbol}] Orderflow bearish (imbalance={of_imbalance:.2f})")
                    continue

                quantum_score = calculate_quantum_score(
                    micro_bos, entry_signal, df_15m, df_5m, retest_idx,
                    orderflow_imbalance=of_imbalance,
                    volume_regime_ok=volume_ok,
                    market_regime=regime
                )
                if quantum_score < CONFIG["MIN_QUANTUM_SCORE"]:
                    continue

                LAST_BOS_INDEX[symbol] = bos_candidate_idx
                LAST_SIGNAL_TIME[symbol] = now

                if ERROR_TRACKER:
                    ERROR_TRACKER.record_success(symbol)

                duration = time.time() - start_time
                PERF_SIGNAL_GEN.record(duration)

                signal: SignalData = {
                    "symbol": symbol,
                    "entry": float(entry_signal.entry_price),
                    "sl": float(sl),
                    "tp1": float(tp1),
                    "tp2": float(tp2),
                    "tp3": float(tp3),
                    "atr": float(atr),
                    "position_size_usdt": float(position_size_usdt),
                    "position_size_pct": float(position_size_pct),
                    "quantum_score": float(quantum_score),
                    "micro_bos": asdict(micro_bos),
                    "entry_signal": asdict(entry_signal),
                    "spread_pct": float(spread_pct),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                return signal
        return None
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"[Signal Generator Error] {symbol}: {str(e)}")
        PERF_SIGNAL_GEN.record_error(f"{symbol}: {str(e)}")
        if ERROR_TRACKER:
            ERROR_TRACKER.record_error(symbol)
        return None

# ===================== ORDER EXECUTION =====================
async def place_order_paper(signal: SignalData) -> Optional[Order]:
    """محاكاة أمر ورقي مع احتمالية تنفيذ"""
    symbol = signal["symbol"]
    entry_price = signal["entry"]
    amount = signal["position_size_usdt"] / entry_price
    order = Order(
        symbol=symbol,
        side='buy',
        order_type='limit',
        price=entry_price,
        amount=amount,
        cost=signal["position_size_usdt"],
        timeout_sec=CONFIG["ENTRY_LIMIT_TIMEOUT_SEC"],
        signal_snapshot=signal
    )
    STATS["paper_trades_attempted"] += 1

    fill_prob = CONFIG.get("FILL_PROBABILITY_LIMIT", 0.8)
    if random.random() < fill_prob:
        partial_prob = CONFIG.get("PARTIAL_FILL_PROBABILITY", 0.3)
        if random.random() < partial_prob:
            min_pct = CONFIG.get("PARTIAL_FILL_MIN_PCT", 0.3)
            max_pct = CONFIG.get("PARTIAL_FILL_MAX_PCT", 0.9)
            fill_pct = random.uniform(min_pct, max_pct)
            filled_amount = amount * fill_pct
            filled_cost = filled_amount * entry_price
            order.filled_amount = filled_amount
            order.filled_cost = filled_cost
            order.fill_price_avg = entry_price
            order.status = OrderStatus.PARTIAL
            STATS["paper_trades_partial"] += 1
            logger.info(f"[Paper] Partial fill {symbol}: {fill_pct*100:.1f}% at {entry_price}")
        else:
            order.filled_amount = amount
            order.filled_cost = signal["position_size_usdt"]
            order.fill_price_avg = entry_price
            order.status = OrderStatus.FILLED
            STATS["paper_trades_filled"] += 1
            logger.info(f"[Paper] Full fill {symbol} at {entry_price}")
    else:
        order.status = OrderStatus.CANCELLED
        STATS["paper_trades_cancelled"] += 1
        logger.info(f"[Paper] Order not filled {symbol}")

    order.updated_at = datetime.now(timezone.utc)
    return order

async def place_order_live(exchange, signal: SignalData) -> Optional[Order]:
    """تنفيذ أمر حقيقي عبر API"""
    symbol = signal["symbol"]
    entry_price = signal["entry"]
    amount = signal["position_size_usdt"] / entry_price
    # 🟢 TODO: تطبيق precision قبل إرسال الأمر
    # amount = exchange.amount_to_precision(symbol, amount)
    # price = exchange.price_to_precision(symbol, entry_price)
    try:
        params = {}
        time_in_force = CONFIG.get("ENTRY_ORDER_TIME_IN_FORCE", "IOC")
        if time_in_force in ["IOC", "FOK"]:
            params['timeInForce'] = time_in_force
        async with API_SEMAPHORE:
            order = await exchange.create_limit_buy_order(
                symbol,
                amount,
                entry_price,
                params
            )
        logger.info(f"[Live] Order placed: {order}")
        ord_obj = Order(
            symbol=symbol,
            side='buy',
            order_type='limit',
            price=entry_price,
            amount=amount,
            cost=signal["position_size_usdt"],
            status=OrderStatus.PENDING,
            order_id=order['id'],
            created_at=datetime.now(timezone.utc),
            timeout_sec=CONFIG["ENTRY_LIMIT_TIMEOUT_SEC"],
            signal_snapshot=signal
        )
        PENDING_ORDERS[symbol] = ord_obj
        return ord_obj
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"[Live] Order placement failed: {e}")
        return None

async def execute_signal(signal: SignalData, mode: ExecutionMode, exchange=None) -> Optional[TradeState]:
    """تنفيذ الإشارة حسب الوضع"""
    if mode == ExecutionMode.SIGNAL:
        return None

    if PRICE_FEED:
        current_price = await PRICE_FEED.get_price(signal["symbol"])
        if current_price:
            valid, reason = validate_trade_before_execution(signal, current_price)
            if not valid:
                logger.warning(f"[{signal['symbol']}] Signal validation failed: {reason}")
                return None

    order = None
    if mode == ExecutionMode.PAPER:
        order = await place_order_paper(signal)
    elif mode == ExecutionMode.LIVE and exchange:
        order = await place_order_live(exchange, signal)

    if not order or order.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
        return None

    entry_price = order.fill_price_avg if order.fill_price_avg > 0 else signal["entry"]
    position_size_usdt = order.filled_cost if order.filled_cost > 0 else signal["position_size_usdt"]
    position_size_asset = order.filled_amount if order.filled_amount > 0 else (position_size_usdt / entry_price)

    original_position_size_asset = position_size_asset

    used_signal = order.signal_snapshot if order.signal_snapshot else signal

    trade = TradeState(
        symbol=used_signal["symbol"],
        entry_price=entry_price,
        original_sl=used_signal["sl"],
        current_sl=used_signal["sl"],
        tp1=used_signal["tp1"],
        tp2=used_signal["tp2"],
        tp3=used_signal["tp3"],
        atr=used_signal["atr"],
        position_size_usdt=position_size_usdt,
        position_size_asset=position_size_asset,
        original_position_size_asset=original_position_size_asset,
        quantum_score=used_signal["quantum_score"],
        is_paper=(mode == ExecutionMode.PAPER),
        execution_mode=mode.value,
        order_id=order.order_id
    )
    if order.status == OrderStatus.PARTIAL:
        trade.fill_log.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "filled_amount": order.filled_amount,
            "filled_cost": order.filled_cost,
            "price": entry_price
        })

    async with TRADES_LOCK:
        ACTIVE_TRADES[used_signal["symbol"]] = trade

    logger.info(f"✅ Trade opened: {trade.symbol} at {trade.entry_price} (mode={mode.value})")
    return trade

# ===================== 🟢 ENHANCED: MONITOR OPEN TRADES =====================
async def monitor_open_trades(exchange=None):
    """
    🟢 ENHANCED: مراقبة الصفقات مع:
    - Lock للحماية من race conditions
    - PriceFeed موحد
    - معالجة partial fills قبل cancel
    - حساب TP جزئية صحيح باستخدام original_position_size_asset
    """
    while not SHUTDOWN_REQUESTED:
        try:
            if exchange and get_execution_mode() == ExecutionMode.LIVE:
                for symbol, order in list(PENDING_ORDERS.items()):
                    if order.status != OrderStatus.PENDING:
                        continue

                    try:
                        async with API_SEMAPHORE:
                            fetched = await safe_api_call(lambda: exchange.fetch_order(order.order_id, symbol), label=f"fetch_order({symbol})")

                        if fetched['status'] in ['closed', 'filled']:
                            order.status = OrderStatus.FILLED
                            order.filled_amount = fetched['filled']
                            order.filled_cost = fetched['cost']
                            order.fill_price_avg = fetched['average'] or order.price
                            order.updated_at = datetime.now(timezone.utc)

                            if order.signal_snapshot:
                                sig = order.signal_snapshot
                                trade = TradeState(
                                    symbol=symbol,
                                    entry_price=order.fill_price_avg,
                                    original_sl=sig["sl"],
                                    current_sl=sig["sl"],
                                    tp1=sig["tp1"],
                                    tp2=sig["tp2"],
                                    tp3=sig["tp3"],
                                    atr=sig["atr"],
                                    position_size_usdt=order.filled_cost,
                                    position_size_asset=order.filled_amount,
                                    original_position_size_asset=order.filled_amount,
                                    quantum_score=sig["quantum_score"],
                                    is_paper=False,
                                    execution_mode=ExecutionMode.LIVE.value,
                                    order_id=order.order_id
                                )
                                async with TRADES_LOCK:
                                    ACTIVE_TRADES[symbol] = trade
                                del PENDING_ORDERS[symbol]
                                logger.info(f"[Live] Order filled: {symbol}")

                        elif fetched['status'] in ['canceled', 'expired']:
                            order.status = OrderStatus.CANCELLED
                            del PENDING_ORDERS[symbol]
                            logger.info(f"[Live] Order cancelled/expired: {symbol}")

                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.error(f"Error fetching order {order.order_id}: {e}")

                    if CONFIG.get("ORDER_TIMEOUT_CANCEL", True):
                        age = (datetime.now(timezone.utc) - order.created_at).total_seconds()
                        if age > order.timeout_sec:
                            try:
                                async with API_SEMAPHORE:
                                    ord_status = await safe_api_call(lambda: exchange.fetch_order(order.order_id, symbol), label=f"fetch_order({symbol})")

                                if ord_status['filled'] > 0:
                                    order.filled_amount = ord_status['filled']
                                    order.filled_cost = ord_status['cost']
                                    order.status = OrderStatus.PARTIAL
                                    logger.info(f"[Live] Partial fill before cancel: {symbol} ({ord_status['filled']} filled)")

                                    if order.signal_snapshot:
                                        sig = order.signal_snapshot
                                        trade = TradeState(
                                            symbol=symbol,
                                            entry_price=ord_status['average'] or order.price,
                                            original_sl=sig["sl"],
                                            current_sl=sig["sl"],
                                            tp1=sig["tp1"],
                                            tp2=sig["tp2"],
                                            tp3=sig["tp3"],
                                            atr=sig["atr"],
                                            position_size_usdt=ord_status['cost'],
                                            position_size_asset=ord_status['filled'],
                                            original_position_size_asset=ord_status['filled'],
                                            quantum_score=sig["quantum_score"],
                                            is_paper=False,
                                            execution_mode=ExecutionMode.LIVE.value,
                                            order_id=order.order_id
                                        )
                                        async with TRADES_LOCK:
                                            ACTIVE_TRADES[symbol] = trade

                                await safe_api_call(lambda: exchange.cancel_order(order.order_id, symbol), label=f"cancel_order({symbol})")
                                logger.info(f"[Live] Order cancelled due to timeout: {symbol}")

                            except asyncio.CancelledError:
                                raise
                            except Exception as e:
                                logger.error(f"Cancel order failed: {e}")
                            finally:
                                if symbol in PENDING_ORDERS:
                                    del PENDING_ORDERS[symbol]

            async with TRADES_LOCK:
                active_symbols = list(ACTIVE_TRADES.keys())

            if active_symbols:
                if PRICE_FEED:
                    prices = await PRICE_FEED.get_prices_batch(active_symbols)
                else:
                    prices = {}

                for symbol in active_symbols:
                    try:
                        async with TRADES_LOCK:
                            if symbol not in ACTIVE_TRADES:
                                continue
                            trade = ACTIVE_TRADES[symbol]

                        if trade.closed:
                            continue

                        current_price = prices.get(symbol)
                        if not current_price:
                            if PRICE_FEED:
                                current_price = await PRICE_FEED.get_price(symbol)
                            if not current_price:
                                logger.warning(f"No price for {symbol}, skipping")
                                continue

                        trade.last_update = datetime.now(timezone.utc).isoformat()

                        risk_per_unit = trade.entry_price - trade.original_sl
                        if risk_per_unit <= 0:
                            continue
                        total_risk_amount = risk_per_unit * trade.original_position_size_asset

                        current_r = (current_price - trade.entry_price) / risk_per_unit

                        if not trade.be_moved and current_r >= CONFIG["BE_AT_R"]:
                            trade.current_sl = trade.entry_price + (CONFIG["BE_ATR_MULT"] * trade.atr)
                            trade.be_moved = True
                            logger.info(f"[{symbol}] Breakeven moved to {trade.current_sl:.6f}")

                        if not trade.trailing_active and current_r >= CONFIG["TRAIL_START_R"]:
                            trade.trailing_active = True
                            logger.info(f"[{symbol}] Trailing activated")

                        if trade.trailing_active:
                            new_sl = current_price - (CONFIG["TRAIL_ATR_MULT"] * trade.atr)
                            if new_sl > trade.current_sl:
                                trade.current_sl = new_sl
                                logger.debug(f"[{symbol}] Trailing updated to {new_sl:.6f}")

                        # TP1
                        if not trade.tp1_hit and current_price >= trade.tp1:
                            trade.tp1_hit = True
                            exit_pct = CONFIG["TP1_EXIT_PCT"]
                            exit_asset = trade.original_position_size_asset * exit_pct
                            exit_value = exit_asset * current_price
                            entry_part_value = exit_asset * trade.entry_price   # 🟢 جديد باستخدام الأصل
                            pnl = calculate_net_pnl(entry_part_value, exit_value)
                            r_multiple = calculate_r_multiple_from_pnl(pnl, total_risk_amount * exit_pct)
                            trade.total_realized_r += r_multiple
                            trade.remaining_pct -= exit_pct
                            trade.position_size_asset -= exit_asset
                            trade.position_size_usdt -= entry_part_value
                            STATS["tp1_hits"] += 1
                            logger.info(f"[{symbol}] TP1 hit, realized R: {r_multiple:.2f}")

                        # TP2
                        if not trade.tp2_hit and current_price >= trade.tp2:
                            trade.tp2_hit = True
                            exit_pct = CONFIG["TP2_EXIT_PCT"]
                            exit_asset = trade.original_position_size_asset * exit_pct
                            exit_value = exit_asset * current_price
                            entry_part_value = exit_asset * trade.entry_price   # 🟢 باستخدام الأصل
                            pnl = calculate_net_pnl(entry_part_value, exit_value)
                            r_multiple = calculate_r_multiple_from_pnl(pnl, total_risk_amount * exit_pct)
                            trade.total_realized_r += r_multiple
                            trade.remaining_pct -= exit_pct
                            trade.position_size_asset -= exit_asset
                            trade.position_size_usdt -= entry_part_value
                            STATS["tp2_hits"] += 1
                            logger.info(f"[{symbol}] TP2 hit, realized R: {r_multiple:.2f}")

                        # TP3
                        if not trade.tp3_hit and current_price >= trade.tp3:
                            trade.tp3_hit = True
                            exit_asset = trade.position_size_asset  # المتبقي (20% من الأصل)
                            exit_value = exit_asset * current_price
                            entry_part_value = exit_asset * trade.entry_price   # 🟢 باستخدام الأصل
                            pnl = calculate_net_pnl(entry_part_value, exit_value)
                            remaining_risk = total_risk_amount * trade.remaining_pct
                            r_multiple = calculate_r_multiple_from_pnl(pnl, remaining_risk)
                            trade.total_realized_r += r_multiple
                            trade.remaining_pct = 0
                            trade.position_size_asset = 0
                            trade.position_size_usdt = 0
                            STATS["tp3_hits"] += 1
                            logger.info(f"[{symbol}] TP3 hit, realized R: {r_multiple:.2f}")

                        # Stop loss
                        if current_price <= trade.current_sl and not trade.closed:
                            exit_asset = trade.position_size_asset
                            exit_value = exit_asset * current_price
                            entry_part_value = exit_asset * trade.entry_price
                            pnl = calculate_net_pnl(entry_part_value, exit_value)
                            remaining_risk = total_risk_amount * trade.remaining_pct
                            r_multiple = calculate_r_multiple_from_pnl(pnl, remaining_risk)
                            trade.total_realized_r += r_multiple

                            if TRADE_JOURNAL:
                                TRADE_JOURNAL.record_trade(trade, current_price, "stop_loss")

                            update_daily_stats(trade.total_realized_r)

                            # 🟢 Loss cooldown
                            if r_multiple < 0:
                                set_loss_cooldown(symbol)

                            logger.info(f"[{symbol}] Closed at SL, total R: {trade.total_realized_r:.2f}")
                            trade.closed = True

                            async with TRADES_LOCK:
                                if symbol in ACTIVE_TRADES:
                                    del ACTIVE_TRADES[symbol]

                        if trade.remaining_pct <= 0.01 and not trade.closed:
                            if TRADE_JOURNAL:
                                TRADE_JOURNAL.record_trade(trade, current_price, "full_tp")

                            update_daily_stats(trade.total_realized_r)
                            logger.info(f"[{symbol}] Fully closed, total R: {trade.total_realized_r:.2f}")
                            trade.closed = True

                            async with TRADES_LOCK:
                                if symbol in ACTIVE_TRADES:
                                    del ACTIVE_TRADES[symbol]

                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.error(f"Error monitoring trade {symbol}: {e}")

            await asyncio.sleep(5)

        except asyncio.CancelledError:
            # graceful exit
            break
        except Exception as e:
            logger.error(f"Monitor loop error: {e}")
            await asyncio.sleep(5)

# ===================== TELEGRAM (مع رابط TradingView) =====================
async def _send_single_telegram(msg: str):
    if not CONFIG["TG_TOKEN"] or not CONFIG["TG_CHAT"]:
        return
    try:
        url = f"https://api.telegram.org/bot{CONFIG['TG_TOKEN']}/sendMessage"
        async with HTTP_SESSION.post(url, json={
            "chat_id": CONFIG["TG_CHAT"],
            "text": msg,
            "parse_mode": "HTML"
        }, timeout=10) as resp:
            if resp.status != 200:
                logger.error(f"[TG Error] {resp.status}")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"[TG Exception] {str(e)}")

async def send_telegram(msg: str):
    if not CONFIG["TG_TOKEN"] or not CONFIG["TG_CHAT"] or CONFIG["SILENT_MODE"]:
        logger.info(f"[TG] {msg[:200]}")
        return
    try:
        if len(msg) > 4000:
            parts = msg.split('\n')
            current = ""
            for part in parts:
                if len(current) + len(part) + 1 < 3900:
                    current += part + '\n'
                else:
                    if current:
                        await _send_single_telegram(current)
                        await asyncio.sleep(0.5)
                    current = part + '\n'
            if current:
                await _send_single_telegram(current)
        else:
            await _send_single_telegram(msg)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"[TG Exception] {str(e)}")

def format_scalp_signal(signal: SignalData) -> str:
    entry = signal["entry"]
    sl = signal["sl"]
    tp1 = signal["tp1"]
    tp2 = signal["tp2"]
    tp3 = signal["tp3"]
    risk = entry - sl
    tp1_r = (tp1 - entry) / risk if risk > 0 else 0
    tp2_r = (tp2 - entry) / risk if risk > 0 else 0
    tp3_r = (tp3 - entry) / risk if risk > 0 else 0
    mode_badge = "✅ LIVE" if is_live_trading_enabled() else ("🟨 PAPER" if is_paper_trading_enabled() else "🟦 SIGNAL")

    # 🟢 رابط TradingView
    clean_symbol = escape_html(signal['symbol'].replace('/', ''))
    exchange_tv = CONFIG.get("EXCHANGE_NAME_FOR_TV", "MEXC")
    tv_link = f"https://www.tradingview.com/chart/?symbol={exchange_tv}:{clean_symbol}"

    message = f"""
⚡ MICRO BOS SCALP - {escape_html(signal['symbol'])}  ({mode_badge})

🎯 الدخول: {entry:.6f}
🛑 وقف الخسارة: {sl:.6f}

أهداف الربح (سكالب):
✅ الهدف 1 (R:{tp1_r:.1f}) - خروج 50%: {tp1:.6f}
✅ الهدف 2 (R:{tp2_r:.1f}) - خروج 30%: {tp2:.6f}
✅ الهدف 3 (R:{tp3_r:.1f}) - خروج 20%: {tp3:.6f}

💰 حجم الصفقة
━━━━━━━━━━━━━━━━
• الحجم: ${signal['position_size_usdt']:.2f}
• % من الحساب: {signal['position_size_pct']:.2f}%
• المخاطرة: {CONFIG['RISK_PER_TRADE_PCT']}%

📊 التحليل
━━━━━━━━━━━━━━━━
• النقاط: {signal['quantum_score']:.0f}/100
• Micro BOS Strength: {signal['micro_bos']['strength']:.0f}%
• Volume Spike: {'✅' if signal['micro_bos']['volume_spike'] else '❌'}

🎯 الدخول
• النوع: {escape_html(signal['entry_signal']['entry_type'])}
• Rejection Wick: {'✅' if signal['entry_signal']['rejection_wick'] else '❌'}
• Volume OK: {'✅' if signal['entry_signal']['volume_ok'] else '❌'}
• Liquidity Sweep: {'✅' if signal['entry_signal'].get('liquidity_sweep') else '❌'}
• Min Pullback OK: {'✅' if signal['entry_signal'].get('pullback_ok') else '❌'}
• السبريد الحالي: {signal.get('spread_pct', 0):.2f}%

🔍 <a href=\"{tv_link}\">فتح في TradingView</a>

⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
    return message

# ===================== HEALTH CHECK SERVER =====================
async def health_handler(request):
    perf_stats = PERF_SIGNAL_GEN.get_stats()
    api_stats = PERF_API_CALL.get_stats()

    error_stats = ERROR_TRACKER.get_stats() if ERROR_TRACKER else {}
    journal_stats = TRADE_JOURNAL.get_performance_stats() if TRADE_JOURNAL else {}

    return web.json_response({
        "status": "ok",
        "utc": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "signals_generated": STATS.get("signals_generated", 0),
            "daily_trades_count": STATS.get("daily_trades_count", 0),
            "daily_r": STATS.get("daily_r", 0.0),
            "api_errors": STATS.get("api_errors", 0),
            "loop_count": STATS.get("loop_count", 0),
            "paper_trades_filled": STATS.get("paper_trades_filled", 0),
            "paper_trades_partial": STATS.get("paper_trades_partial", 0),
        },
        "performance": {
            "signal_generation": perf_stats,
            "api_calls": api_stats,
        },
        "memory": {
            "symbols_tracked": request.app.get("symbols_count", 0),
            "last_signal_time_entries": len(LAST_SIGNAL_TIME),
            "last_bos_index_entries": len(LAST_BOS_INDEX),
            "active_trades": len(ACTIVE_TRADES),
            "pending_orders": len(PENDING_ORDERS),
            "cache_entries": len(_15M_CACHE),
            "loss_cooldown_entries": len(LOSS_COOLDOWN),
        },
        "error_tracking": error_stats,
        "trade_journal": journal_stats,
    })

async def start_health_server(app_state: Dict[str, Any]) -> Optional[web.AppRunner]:
    if not CONFIG.get("ENABLE_HEALTH_CHECK", True):
        return None
    try:
        port = int(CONFIG.get("HEALTH_CHECK_PORT", 8080))
        app = web.Application()
        app["symbols_count"] = app_state.get("symbols_count", None)
        app.router.add_get("/health", health_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host="0.0.0.0", port=port)
        await site.start()
        logger.info(f"✅ Health server running on 0.0.0.0:{port}/health")
        return runner
    except Exception as e:
        logger.error(f"[Health Server Error] {str(e)}")
        return None

async def stop_health_server(runner: Optional[web.AppRunner]):
    if runner is None:
        return
    try:
        await runner.cleanup()
        logger.info("Health server stopped")
    except Exception as e:
        logger.error(f"[Health] Stop error: {str(e)}")

# ===================== EXCHANGE CONTEXT MANAGER =====================
@asynccontextmanager
async def get_exchange():
    exchange = ccxt.mexc({
        'apiKey': CONFIG.get("MEXC_API_KEY") if is_live_trading_enabled() else None,
        'secret': CONFIG.get("MEXC_API_SECRET") if is_live_trading_enabled() else None,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    try:
        await exchange.load_markets()
        logger.info(f"✅ Exchange connected: {len(exchange.markets)} markets")
        yield exchange
    finally:
        try:
            await exchange.close()
        except Exception:
            pass
        logger.info("Exchange connection closed")

# ===================== MAIN LOOP =====================
async def main_loop():
    global HTTP_SESSION, API_SEMAPHORE, TRADES_LOCK, PRICE_FEED, ERROR_TRACKER, TRADE_JOURNAL

    logger.info("="*70)
    logger.info("🚀 MICRO BOS SCALPING BOT v2.5 - PRODUCTION FINAL (FULLY AUDITED + INSTITUTIONAL FEATURES)")
    logger.info("="*70)
    logger.info(f"Timeframes: {CONFIG['TF_ANALYSIS']} (Analysis) + {CONFIG['TF_ENTRY']} (Entry)")
    logger.info(f"Min Score: {CONFIG['MIN_QUANTUM_SCORE']}")
    logger.info(f"Risk/Trade: {CONFIG['RISK_PER_TRADE_PCT']}%")
    logger.info(f"Max SL: {CONFIG['MAX_SL_PCT']}%")
    logger.info(f"Execution Mode: {get_execution_mode().value}")
    logger.info(f"Post-BOS Lookahead: {CONFIG['POST_BOS_LOOKAHEAD_CANDLES']} candles")
    logger.info("="*70)
    logger.info("✅ CRITICAL FIXES APPLIED:")
    logger.info("  • Race condition protection with asyncio.Lock")
    logger.info("  • Correct TP partial calculation from original position")
    logger.info("  • Unified PriceFeed for paper & live trading")
    logger.info("  • Enhanced Micro BOS detection with close confirmation")
    logger.info("  • Periodic cache cleanup to prevent memory leak")
    logger.info("  • Spread calculation with mid price")
    logger.info("  • Partial fill handling before order cancel")
    logger.info("="*70)
    logger.info("✅ INSTITUTIONAL FEATURES ADDED:")
    logger.info("  • Market Regime Filter (TRENDING only)")
    logger.info("  • BTC Market Filter (prevents trading during dump)")
    logger.info("  • Volume Regime Filter (checks daily volume activity)")
    logger.info("  • Order Flow Light (bid/ask imbalance)")
    logger.info("  • Advanced Daily Loss Circuit Breaker")
    logger.info("  • Symbol Cooldown after losing trade")
    logger.info("  • Enhanced Quantum Scoring (includes new factors)")
    logger.info("  • TradingView chart link in Telegram messages")
    logger.info("="*70)
    logger.info("✅ POST-AUDIT TWEAKS (SCALPING OPTIMIZATION):")
    logger.info("  • Liquidity Sweep no longer mandatory")
    logger.info("  • Relaxed entry quality filters (ATR% ≤4.5, BB width ≤0.08)")
    logger.info("  • Relaxed market environment (RSI≥45, ADX≥15, EMA50>200 optional)")
    logger.info("  • Min Quantum Score reduced to 50")
    logger.info("  • PriceFeed cache reduced to 0.7s")
    logger.info("  • Removed momentum filter")
    logger.info("  • Lowered min 24h volume to 300k")
    logger.info("  • Fixed safe_api_call (accepts callable now)")
    logger.info("  • Fixed R calculation in TP3 and Stop Loss using remaining risk")
    logger.info("  • Reduced 15m cache TTL to 7 minutes")
    logger.info("="*70)
    logger.info("✅ COLAB/JUPYTER FIXES ADDED:")
    logger.info("  • Safe async execution in notebooks")
    logger.info("  • No duplicate log handlers")
    logger.info("  • Clean cancellation & shutdown")
    logger.info("="*70)

    load_state()

    req_per_sec = int(CONFIG.get("REQUESTS_PER_SECOND", 10))
    concurrency = max(1, min(req_per_sec, 10))

    API_SEMAPHORE = asyncio.Semaphore(concurrency)
    TRADES_LOCK = asyncio.Lock()
    assert TRADES_LOCK is not None, "TRADES_LOCK not initialized"

    connector = aiohttp.TCPConnector(limit=concurrency, ttl_dns_cache=300)
    HTTP_SESSION = aiohttp.ClientSession(connector=connector)
    logger.info("✅ HTTP Session created")

    loop = asyncio.get_running_loop()
    _safe_add_signal_handlers(loop)

    async with get_exchange() as exchange:
        PRICE_FEED = PriceFeed(exchange)
        ERROR_TRACKER = ErrorTracker(
            max_errors=CONFIG.get("ERROR_TRACK_MAX_ERRORS", 5),
            cooldown_sec=CONFIG.get("ERROR_TRACK_COOLDOWN_SEC", 3600)
        )
        TRADE_JOURNAL = TradeJournal()
        logger.info("✅ PriceFeed, ErrorTracker, TradeJournal initialized")

        symbols_to_trade = await filter_symbols_by_liquidity(exchange)
        logger.info(f"✅ Trading symbols: {len(symbols_to_trade)}")

        health_runner = await start_health_server({"symbols_count": len(symbols_to_trade)})

        monitor_task = None
        if get_execution_mode() in (ExecutionMode.PAPER, ExecutionMode.LIVE):
            monitor_task = asyncio.create_task(
                monitor_open_trades(exchange if get_execution_mode() == ExecutionMode.LIVE else None),
                name="monitor_open_trades"
            )

        await send_telegram(f"""
🚀 Micro BOS Scalping Bot v2.5 Started (FULLY AUDITED + INSTITUTIONAL FEATURES)

⚙️ إعدادات:
• Timeframes: {CONFIG['TF_ANALYSIS']} + {CONFIG['TF_ENTRY']}
• Risk/Trade: {CONFIG['RISK_PER_TRADE_PCT']}%
• Max Daily Trades: {CONFIG.get('DAILY_MAX_TRADES', 15)}
• Min Score: {CONFIG['MIN_QUANTUM_SCORE']}
• Execution Mode: {get_execution_mode().value}

✨ الميزات المؤسسية الجديدة:
🏆 Market Regime Filter
🏆 BTC Filter
📊 Volume Regime
💹 Order Flow Light
🔒 Daily Circuit Breaker
⏱️ Loss Cooldown
📈 Enhanced Scoring
🔗 TradingView Link

جاهز للتداول المؤسسي! ⚡
""")

        loop_count = 0
        last_symbol_refresh_time = time.time()
        last_state_save_time = time.time()
        last_cleanup_time = time.time()
        last_cache_cleanup_time = time.time()
        last_btc_check_time = time.time()

        while not SHUTDOWN_REQUESTED:
            try:
                loop_start = time.time()
                loop_count += 1
                STATS["loop_count"] = loop_count

                # 🟢 BTC Filter
                if CONFIG.get("ENABLE_BTC_FILTER", True):
                    if time.time() - last_btc_check_time > 60:  # تحقق كل دقيقة
                        btc_status = await check_btc_trend(exchange)
                        last_btc_check_time = time.time()
                    else:
                        btc_status = BTC_TREND or {"safe_to_trade": True}

                    if not btc_status.get("safe_to_trade", True):
                        logger.info(f"[Main] BTC not safe ({btc_status.get('trend')}) - waiting 60s")
                        await asyncio.sleep(60)
                        continue

                ok_cb, reason_cb = daily_circuit_breaker_ok()
                if not ok_cb:
                    logger.info(f"[Main] Circuit breaker active ({reason_cb}) - waiting")
                    await asyncio.sleep(60)
                    continue

                current_time = time.time()

                if current_time - last_symbol_refresh_time > CONFIG["SYMBOL_REFRESH_INTERVAL_HOURS"] * 3600:
                    logger.info("🔄 Refreshing symbol list...")
                    new_symbols = await filter_symbols_by_liquidity(exchange)
                    if new_symbols:
                        symbols_to_trade = new_symbols
                        logger.info(f"✅ Symbols updated: {len(symbols_to_trade)}")
                        await cleanup_stale_symbols(set(symbols_to_trade))
                        if health_runner and hasattr(health_runner, 'app'):
                            health_runner.app["symbols_count"] = len(symbols_to_trade)
                    last_symbol_refresh_time = current_time

                if current_time - last_state_save_time > CONFIG["STATE_SAVE_INTERVAL_SEC"]:
                    save_state()
                    last_state_save_time = current_time

                if current_time - last_cleanup_time > CONFIG["MEMORY_CLEANUP_INTERVAL_SEC"]:
                    await cleanup_stale_symbols(set(symbols_to_trade))
                    last_cleanup_time = current_time

                if current_time - last_cache_cleanup_time > CONFIG.get("CACHE_CLEANUP_INTERVAL_SEC", 900):
                    await cleanup_cache()
                    if PRICE_FEED:
                        PRICE_FEED.clear_cache()
                    last_cache_cleanup_time = current_time

                mode = get_execution_mode()
                for symbol in symbols_to_trade:
                    if SHUTDOWN_REQUESTED:
                        break
                    try:
                        async with TRADES_LOCK:
                            has_active = symbol in ACTIVE_TRADES
                        if has_active:
                            continue

                        signal_data = await generate_scalp_signal(exchange, symbol)
                        if signal_data:
                            STATS["signals_generated"] += 1
                            STATS["scalp_signals"] += 1

                            message = format_scalp_signal(signal_data)
                            await send_telegram(message)

                            logger.info(f"⚡ [SCALP SIGNAL] {symbol} - Score: {signal_data['quantum_score']:.0f}")

                            if mode != ExecutionMode.SIGNAL:
                                trade = await execute_signal(signal_data, mode, exchange if mode == ExecutionMode.LIVE else None)
                                if trade:
                                    pass

                            await asyncio.sleep(1)

                        # 🟢 Colab stability: yield للـ event loop كل شوية
                        await asyncio.sleep(0.1)

                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.error(f"[Symbol Error] {symbol}: {str(e)}")
                        continue

                loop_time = time.time() - loop_start
                sleep_time = max(15, 20 - loop_time)

                if loop_count % 10 == 0:
                    perf = PERF_SIGNAL_GEN.get_stats()
                    logger.info(
                        f"[Loop {loop_count}] "
                        f"Time: {loop_time:.1f}s, "
                        f"Signals: {STATS['signals_generated']}, "
                        f"Avg gen: {perf['avg_ms']:.0f}ms, "
                        f"P95: {perf['p95_ms']:.0f}ms"
                    )

                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except KeyboardInterrupt:
                logger.info("\n[System] Shutdown signal received")
                break
            except Exception as e:
                logger.error(f"[Loop Error] {str(e)}")
                logger.debug(traceback.format_exc())
                await asyncio.sleep(30)

    # Cleanup
    logger.info("🧹 Performing cleanup...")

    if 'monitor_task' in locals() and monitor_task:
        try:
            monitor_task.cancel()
            await asyncio.gather(monitor_task, return_exceptions=True)
        except Exception:
            pass

    try:
        save_state()
    except Exception as e:
        logger.error(f"Final state save error: {e}")

    if TRADE_JOURNAL:
        final_stats = TRADE_JOURNAL.get_performance_stats()
        logger.info("="*70)
        logger.info("📊 FINAL PERFORMANCE SUMMARY:")
        logger.info(f"  Total Trades: {final_stats.get('total_trades', 0)}")
        logger.info(f"  Win Rate: {final_stats.get('win_rate', 0)*100:.1f}%")
        logger.info(f"  Avg Win R: {final_stats.get('avg_win_r', 0):.2f}")
        logger.info(f"  Avg Loss R: {final_stats.get('avg_loss_r', 0):.2f}")
        logger.info(f"  Profit Factor: {final_stats.get('profit_factor', 0):.2f}")
        logger.info(f"  Total R: {final_stats.get('total_r', 0):.2f}")
        logger.info(f"  Expectancy R: {final_stats.get('expectancy_r', 0):.2f}")
        logger.info("="*70)

    try:
        await stop_health_server(locals().get("health_runner"))
    except Exception as e:
        logger.error(f"Health server stop error: {e}")

    try:
        if HTTP_SESSION:
            await HTTP_SESSION.close()
            logger.info("HTTP Session closed")
    except Exception as e:
        logger.error(f"HTTP session close error: {e}")

    logger.info("✅ Shutdown complete")

def main():
    """
    🟢 Colab/Jupyter behavior:
      - لا نستخدم asyncio.run
      - نرجّع Task واحدة فقط (منع التكرار)
    """
    global _MAIN_LOOP_TASK

    try:
        if _is_running_in_notebook():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                if _MAIN_LOOP_TASK and not _MAIN_LOOP_TASK.done():
                    logger.warning("⚠️ main_loop already running in this notebook kernel.")
                    return _MAIN_LOOP_TASK
                logger.warning("⚠️ Running in Jupyter/Colab - start with: await main_loop()  (or use returned task)")
                _MAIN_LOOP_TASK = asyncio.create_task(main_loop(), name="micro_bos_main_loop")
                return _MAIN_LOOP_TASK
            else:
                # fallback (rare in notebook)
                return asyncio.run(main_loop())
        else:
            return asyncio.run(main_loop())

    except KeyboardInterrupt:
        logger.info("\n👋 Bot stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
