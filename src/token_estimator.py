"""Token usage estimator — forecast monthly MiMo API consumption."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class UsageScenario:
    """Parameters describing a deployment scenario."""
    name: str
    daily_sessions: int
    avg_turns_per_session: int
    tokens_per_turn: int = 3_500
    batch_docs_per_week: int = 0
    tokens_per_batch_doc: int = 1_200
    deep_analysis_calls_per_month: int = 0
    tokens_per_deep_analysis: int = 200_000


class TokenEstimator:
    """
    Estimates monthly token consumption based on deployment scenario.

    Usage::

        est = TokenEstimator()
        est.estimate(UsageScenario(
            name="Production",
            daily_sessions=500,
            avg_turns_per_session=10,
            batch_docs_per_week=1000,
        ))
    """

    OVERHEAD_FACTOR = 1.15  # 15% overhead for retries, failed calls, etc.

    def estimate(self, scenario: UsageScenario) -> dict:
        """
        Calculate total monthly token consumption for a scenario.

        Returns:
            Dict with breakdown and total.
        """
        # Multi-turn sessions: tokens grow each turn
        # Average turn cost ≈ tokens_per_turn * (1 + 2 + ... + N) / N ≈ tokens * avg_turn
        avg_turn_multiplier = (scenario.avg_turns_per_session + 1) / 2
        session_tokens_per_day = (
            scenario.daily_sessions
            * scenario.avg_turns_per_session
            * scenario.tokens_per_turn
            * avg_turn_multiplier
        )
        monthly_session_tokens = int(session_tokens_per_day * 30)

        # Batch processing
        monthly_batch_tokens = int(
            scenario.batch_docs_per_week * 4.3  # weeks/month
            * scenario.tokens_per_batch_doc
        )

        # Deep analysis
        monthly_deep_tokens = (
            scenario.deep_analysis_calls_per_month
            * scenario.tokens_per_deep_analysis
        )

        raw_total = monthly_session_tokens + monthly_batch_tokens + monthly_deep_tokens
        total_with_overhead = int(raw_total * self.OVERHEAD_FACTOR)

        return {
            "scenario": scenario.name,
            "monthly_session_tokens": monthly_session_tokens,
            "monthly_batch_tokens": monthly_batch_tokens,
            "monthly_deep_analysis_tokens": monthly_deep_tokens,
            "subtotal": raw_total,
            "overhead_15pct": total_with_overhead - raw_total,
            "total_monthly_tokens": total_with_overhead,
            "total_monthly_tokens_human": f"{total_with_overhead/1_000_000:.1f}M",
        }

    def print_report(self, scenario: UsageScenario) -> None:
        """Pretty-print a token usage report for a scenario."""
        r = self.estimate(scenario)
        print(f"\n{'='*55}")
        print(f"  Token Usage Estimate: {r['scenario']}")
        print(f"{'='*55}")
        print(f"  Multi-turn sessions : {r['monthly_session_tokens']:>15,} tokens")
        print(f"  Batch summarization : {r['monthly_batch_tokens']:>15,} tokens")
        print(f"  Deep analysis calls : {r['monthly_deep_analysis_tokens']:>15,} tokens")
        print(f"  {'─'*40}")
        print(f"  Subtotal            : {r['subtotal']:>15,} tokens")
        print(f"  Overhead (15%)      : {r['overhead_15pct']:>15,} tokens")
        print(f"  {'─'*40}")
        print(f"  TOTAL / MONTH       : {r['total_monthly_tokens']:>15,} tokens")
        print(f"                        ({r['total_monthly_tokens_human']} tokens)")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    est = TokenEstimator()

    scenarios = [
        UsageScenario("Small Team (10 users)", daily_sessions=50, avg_turns_per_session=6, batch_docs_per_week=100),
        UsageScenario("Medium (50 users)", daily_sessions=250, avg_turns_per_session=8, batch_docs_per_week=500, deep_analysis_calls_per_month=20),
        UsageScenario("Production (100 users)", daily_sessions=500, avg_turns_per_session=10, batch_docs_per_week=1000, deep_analysis_calls_per_month=50),
    ]

    for s in scenarios:
        est.print_report(s)
