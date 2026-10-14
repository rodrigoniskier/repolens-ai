# Acme Toolkit

Small sample project used as a fixed evaluation fixture for RepoLens AI.

## Modules

- `auth/tokens.py` issues and validates signed session tokens.
- `billing/invoice.py` computes invoice totals, tax and discounts.
- `cache/lru.py` implements a least-recently-used cache.
- `utils/retry.py` retries flaky operations with exponential backoff.
