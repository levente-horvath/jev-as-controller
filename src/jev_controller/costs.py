"""Offline cost planning and a hard, thread-safe ceiling on paid requests."""

import threading
from typing import Any

PRICE_PER_MTOK = 0.042
TOKENS_PER_CALL = 1000  # legacy logs measured 675–935; includes question and state
RESERVE_TOKENS_PER_CALL = 1200  # planning reserve, not a tokenizer guarantee


class RequestLimitReached(RuntimeError):
    pass


class LimitedClient:
    def __init__(self, client: Any, max_calls: int) -> None:
        self.client = client
        self.max_calls = max_calls
        self.calls = 0
        self.lock = threading.Lock()

    def system_one(self, **kwargs: Any) -> Any:
        with self.lock:
            if self.calls >= self.max_calls:
                raise RequestLimitReached(f"Paid request cap reached ({self.max_calls}); no further calls sent.")
            self.calls += 1
        return self.client.system_one(**kwargs)

    def close(self) -> None:
        self.client.close()


def estimate(calls: int) -> str:
    typical = calls * TOKENS_PER_CALL * PRICE_PER_MTOK / 1e6
    reserve = calls * RESERVE_TOKENS_PER_CALL * PRICE_PER_MTOK / 1e6
    return (
        f"At most {calls:,} API calls; ${typical:.4f} at {TOKENS_PER_CALL:,} input tokens/call; "
        f"${reserve:.4f} with {RESERVE_TOKENS_PER_CALL:,} tokens/call.\n"
        f"Price: ${PRICE_PER_MTOK}/million input tokens; outputs free; automatic retries disabled.\n"
        "These are planning estimates, not dollar caps. No API calls made by estimate."
    )
