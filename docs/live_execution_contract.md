# Live Execution Contract (Phase 11)

Purpose: Harden live adapters for predictable behavior under stress.

Execution requirements
- Order throttling (rate limits)
- Slippage modeling (PAPER) / tolerance enforcement (LIVE)
- Retry logic with backoff
- Cancel-replace logic
- Partial fill handling
- Position reconciliation (real-time vs internal state)
- End-of-day flattening (optional)
- Error classification and propagation

Acceptance
- Adapters must behave deterministically under stress; retries and throttling must prevent runaway order flow.
- Reconciliation must converge positions; partial fills must be tracked accurately.
