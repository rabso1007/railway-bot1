import numpy as np
import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

@dataclass
class TradeResult:
    r: float
    win: bool
    tp: bool
    sl: bool
    score: float
    timestamp: float = field(default_factory=time.time)


class EdgeState:

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.trades = deque(maxlen=window_size)

        self.edge_score = 0.5
        self.system_state = "NORMAL"

        self.winrate = 0.0
        self.avg_r = 0.0

        self._lock = asyncio.Lock()

    async def record_trade(self, r_multiple, quantum_score, exit_type):

        async with self._lock:

            result = TradeResult(
                r=r_multiple,
                win=r_multiple > 0,
                tp=exit_type != "SL",
                sl=exit_type == "SL",
                score=quantum_score,
            )

            self.trades.append(result)
            await self._recalculate()

    async def _recalculate(self):

        if len(self.trades) < 10:
            return

        wins = sum(1 for t in self.trades if t.win)
        self.winrate = wins / len(self.trades)

        self.avg_r = np.mean([t.r for t in self.trades])

        self.edge_score = np.tanh(self.avg_r) * 0.5 + self.winrate * 0.5

        if self.edge_score > 0.65:
            self.system_state = "AGGRESSIVE"
        elif self.edge_score > 0.45:
            self.system_state = "NORMAL"
        elif self.edge_score > 0.30:
            self.system_state = "DEFENSIVE"
        elif self.edge_score > 0.20:
            self.system_state = "SURVIVAL"
        else:
            self.system_state = "PAUSED"

    def risk_multiplier(self):

        mapping = {
            "AGGRESSIVE": 1.2,
            "NORMAL": 1.0,
            "DEFENSIVE": 0.7,
            "SURVIVAL": 0.4,
            "PAUSED": 0.0,
        }

        return mapping.get(self.system_state, 1.0)

    def should_trade(self):
        return self.system_state != "PAUSED"


_edge_engine = None
_lock = asyncio.Lock()


async def get_edge_engine():
    global _edge_engine

    async with _lock:
        if _edge_engine is None:
            _edge_engine = EdgeState()

        return _edge_engine
