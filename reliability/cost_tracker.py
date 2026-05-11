"""
Per-query token budget enforcement.
production cost control.
"""

import logging
from dataclasses import dataclass, field
from config import MAX_COST_PER_QUERY
logger = logging.getLogger("cortex.cost_tracker")

# OpenAI pricing per token (approximate as of 2025)
COST_PER_TOKEN = {
    "gpt-4o-mini":            0.00000015,   # $0.15 / 1M tokens
    "gpt-4o":                 0.000005,     # $5.00 / 1M tokens
    "text-embedding-3-small": 0.00000002,   # $0.02 / 1M tokens
    "text-embedding-ada-002": 0.0000001,    # $0.10 / 1M tokens
}


@dataclass
class QueryBudget:
    """Tracks token usage and cost for a single query."""
    max_usd: float = field(
        default_factory=lambda: float(MAX_COST_PER_QUERY)
    )
    spent: float = 0.0
    tokens_used: int = 0

    def add(self, tokens: int, model: str = "gpt-4o-mini") -> None:
        """Record token usage for a model call."""
        cost = tokens * COST_PER_TOKEN.get(model, 0.00000015)
        self.spent += cost
        self.tokens_used += tokens
        logger.debug(
            "[Budget] +%d tokens (%s) = $%.6f | total $%.6f / $%.2f",
            tokens, model, cost, self.spent, self.max_usd,
        )

    def check(self) -> bool:
        """Returns True if budget is still available."""
        return self.spent < self.max_usd

    def remaining_usd(self) -> float:
        return max(0.0, self.max_usd - self.spent)

    def summary(self) -> str:
        return (
            f"${self.spent:.4f} / ${self.max_usd:.2f} "
            f"({self.tokens_used} tokens)"
        )

    def enforce(self) -> None:
        """Raise BudgetExceeded if over limit."""
        if not self.check():
            raise BudgetExceeded(
                f"Query budget exceeded: {self.summary()}"
            )


class BudgetExceeded(Exception):
    """Raised when a query exceeds its token cost budget."""
    pass
