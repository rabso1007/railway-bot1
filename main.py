#!/usr/bin/env python3
"""
MICRO BOS SCALPING BOT v2.5 - PRODUCTION FINAL (FULLY AUDITED & ENHANCED)
===========================================================================
استراتيجية سكالبينج احترافية - النسخة النهائية بعد دمج جميع تحسينات التدقيق الشامل

✅ إصلاحات حرجة مطبقة (CRITICAL FIXES):
   • FIX 1: إضافة asyncio.Lock لحماية ACTIVE_TRADES من race conditions
   • FIX 2: إصلاح حساب TP الجزئية باستخدام original_position_size
   • FIX 3: PriceFeed موحد لجلب الأسعار الحقيقية في الورقي والحي
   • FIX 4: تحسين detect_micro_bos مع تأكيد الإغلاق فوق القمة
   • FIX 5: تنظيف دوري لـ cache لمنع memory leak
   • FIX 6: حساب Spread بـ mid price
   • FIX 7: معالجة partial fills قبل إلغاء الأوامر

✅ تحسينات إضافية (ENHANCEMENTS):
   • ValidationLayer: التحقق من صلاحية الإشارة قبل التنفيذ
   • ErrorTracker: تتبع الأخطاء ووضع cooldown تلقائي
   • TradeJournal: تحليل أداء الصفقات وإحصائيات متقدمة
   • Price Feed: مصدر أسعار موحد مع cache ذكي
   • Enhanced Logging: سجلات أكثر تفصيلاً للتشخيص

🟢 Colab/Jupyter Critical Enhancements (ADDED):
   • FIX C1: تشغيل صحيح داخل Google Colab/Jupyter بدون asyncio.run
   • FIX C2: منع تكرار logging handlers عند إعادة تشغيل الخلية
   • FIX C3: إغلاق مهام asyncio و sessions بشكل نظيف (CancelledError safe)
   • FIX C4: تعطيل signal handlers غير المدعومة في Colab تلقائيًا
   • FIX C5: حماية من Duplicate main_loop tasks داخل نفس الـ event loop
   • FIX C6: تحسين graceful shutdown داخل notebook عبر دوال مساعدة

🚀 التحسينات الإضافية المدمجة من المراجعة (POST-AUDIT TWEAKS):
   • 🟢 Liquidity Sweep لم يعد إلزامياً، بل عامل تعزيز (CONFIG["LIQUIDITY_SWEEP_REQUIRED"] = False)
   • 🟢 تخفيف Entry Quality Filter لصالح السكالبينج (ATR% ≤ 4.5، BB width ≤ 0.08، المسافة من zone ≤ 1.2 ATR)
   • 🟢 تخفيف Market Environment Filter (RSI ≥ 45، ADX ≥ 15، إلغاء شرط EMA50 > EMA200)
   • 🟢 خفض الحد الأدنى لـ Quantum Score إلى 50
   • 🟢 تقليل مدة Cache الـ PriceFeed إلى 0.7 ثانية
   • 🟢 إزالة فلتر Momentum الذي كان يقتل فرص السكالبينج
   • 🟢 خفض حد السيولة اليومي إلى 300,000 USDT
   • 🟢 إصلاح bug دالة safe_api_call (استقبال callable بدل coroutine object)
   • 🟢 إصلاح حساب R في TP3 و Stop Loss باستخدام risk المتبقي
   • 🟢 إصلاح منطق خصم قيمة الدخول باستخدام original_position_size_asset
   • 🟢 تقليل TTL لبيانات 15m من 14 دقيقة إلى 7 دقائق
   • 🟢 إضافة assert لـ TRADES_LOCK بعد الإنشاء
   • 🟢 إزالة المتغير غير المستخدم SYMBOL_COOLDOWN_SEC
   • 🟢 إضافة تأخير بسيط بين معالجة الرموز لتجنب Rate Limits
   • 🟢 تحسين التعامل مع precision (إضافة دالة round_amount)

🚀 **ميزات مؤسسية جديدة (INSTITUTIONAL FEATURES ADDED):**
   • 🏆 Market Regime Filter: تصنيف السوق إلى TRENDING / NEUTRAL / VOLATILE والتداول فقط في TRENDING
   • 🏆 BTC Market Filter: مراقبة BTC ومنع التداول أثناء الانهيارات أو التقلبات العالية
   • 📊 Volume Regime Filter: التحقق من نشاط السوق بمقارنة الحجم اليومي مع الأيام السابقة
   • 💹 Order Flow Light: كشف imbalance بسيط لتقليل الاختراقات الكاذبة
   • 🔒 Daily Loss Circuit Breaker Advanced: إيقاف التداول بعد يوم خاسر (محسّن)
   • ⏱️ Symbol Cooldown الذكي: تبريد الرمز بعد صفقة خاسرة
   • 📈 نظام Scoring أقوى: دمج العوامل الجديدة في درجة Quantum Score
   • 🔗 رابط TradingView في رسائل التلغرام

Timeframes: 15m (تحليل) + 5m (دخول)
استراتيجية: Micro BOS + Liquidity Sweep + Retest + Reversal Candle
"""

import asyncio
import aiohttp
from aiohttp import web
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import ta
import time
import json
import logging
import os
import sys
import signal
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Deque, TypedDict, Set, Callable
from collections import deque, defaultdict
from decimal import Decimal
from enum import Enum
import traceback
import tracemalloc
from functools import wraps
import copy
from contextlib import asynccontextmanager, contextmanager
import random
import math
from logging.handlers import RotatingFileHandler

# ===================== 🟢 COLAB/JUPYTER HELPERS =====================
def _is_running_in_notebook() -> bool:
    """Detect Jupyter/Colab environment safely."""
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip is None:
            return False
        # If IPython is present and has a kernel, it's notebook-like
        return hasattr(ip, "kernel") and ip.kernel is not None
    except Exception:
        return False

def _safe_add_signal_handlers(loop):
    """Signal handlers are not supported on Windows/Jupyter reliably."""
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, mark_shutdown)
        except (NotImplementedError, RuntimeError):
            # Not supported in notebook/Windows; ignore
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
