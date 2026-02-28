from __future__ import annotations


class ExitCode:
    SUCCESS = 0
    BUSINESS_REJECT = 10
    BUSINESS_SKIP = 11
    DATA_INVALID = 20
    STATE_CORRUPT = 30
    LOCK_TIMEOUT = 40
    INTERNAL_ERROR = 50
    ANOMALY_HALT = 60
