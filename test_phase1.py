"""
test_phase1.py
────────────────────────────────────────────────────────────────────────────
Run this from your project root to verify Phase 1 is complete.

    cd deepfake_auth
    python test_phase1.py

You should see 6 green PASS lines and a final success message.
A log file should also appear at logs/deepfake_auth.log.
────────────────────────────────────────────────────────────────────────────
"""

import sys
import traceback

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
results = []


def check(label: str, fn):
    try:
        fn()
        print(f"{PASS}  {label}")
        results.append(True)
    except Exception as e:
        print(f"{FAIL}  {label}")
        print(f"        {type(e).__name__}: {e}")
        traceback.print_exc()
        results.append(False)


# ── Test 1: Logger imports and returns a logger ────────────────────────────
def test_logger_import():
    from infrastructure.logger import get_logger
    log = get_logger(__name__)
    log.info("Phase 1 test — logger working")
    assert log is not None

check("Logger: get_logger() returns a Logger", test_logger_import)


# ── Test 2: Config loads without error ────────────────────────────────────
def test_config_loads():
    from infrastructure.config_loader import load_config
    cfg = load_config()
    assert isinstance(cfg, dict), "load_config() must return a dict"
    assert "device" in cfg, "base_config.yaml must define 'device'"
    assert "verification" in cfg, "model_config.yaml must define 'verification'"

check("Config: load_config() returns merged dict", test_config_loads)


# ── Test 3: Config values are correct types ───────────────────────────────
def test_config_values():
    from infrastructure.config_loader import load_config
    cfg = load_config()
    threshold = cfg["verification"]["accept_threshold"]
    assert isinstance(threshold, float), f"accept_threshold must be float, got {type(threshold)}"
    assert 0.0 < threshold < 1.0, f"accept_threshold must be between 0 and 1, got {threshold}"

check("Config: verification threshold is a valid float", test_config_values)


# ── Test 4: Exceptions can be raised and caught ───────────────────────────
def test_exceptions():
    from infrastructure.exceptions import (
        DeepfakeAuthError,
        FaceNotFoundError,
        LivenessFailedError,
        IdentityNotFoundError,
    )

    # FaceNotFoundError is a DeepfakeAuthError — catch-all works
    try:
        raise FaceNotFoundError("test")
    except DeepfakeAuthError:
        pass

    # LivenessFailedError carries score and threshold
    try:
        raise LivenessFailedError(score=0.12, threshold=0.50)
    except LivenessFailedError as e:
        assert e.score == 0.12
        assert e.threshold == 0.50

    # IdentityNotFoundError carries user_id
    try:
        raise IdentityNotFoundError(user_id="test_user")
    except IdentityNotFoundError as e:
        assert e.user_id == "test_user"
        assert "test_user" in str(e)

check("Exceptions: hierarchy, attributes, and catch-all work", test_exceptions)


# ── Test 5: utils.l2_normalize works correctly ────────────────────────────
def test_l2_normalize():
    import numpy as np
    from infrastructure.utils import l2_normalize

    v = np.array([3.0, 4.0])          # norm = 5.0
    result = l2_normalize(v)
    assert abs(np.linalg.norm(result) - 1.0) < 1e-6, "Normalised vector must have unit norm"

    zero = np.array([0.0, 0.0])       # edge case — should not crash
    result_zero = l2_normalize(zero)
    assert result_zero is zero or (result_zero == zero).all()

check("Utils: l2_normalize produces unit-norm vectors", test_l2_normalize)


# ── Test 6: timer decorator runs and logs ─────────────────────────────────
def test_timer():
    import time
    from infrastructure.utils import timer

    @timer
    def slow_add(a, b):
        time.sleep(0.01)
        return a + b

    result = slow_add(2, 3)
    assert result == 5, "Decorated function must return correct value"

check("Utils: @timer decorator preserves return value", test_timer)


# ── Test 7: Log file was created on disk ──────────────────────────────────
def test_log_file_exists():
    from pathlib import Path
    log_file = Path(__file__).parent / "logs" / "deepfake_auth.log"
    assert log_file.exists(), (
        f"Log file not found at {log_file}. "
        f"Check logging_config.yaml FileHandler path."
    )

check("Logger: log file created at logs/deepfake_auth.log", test_log_file_exists)


# ── Summary ───────────────────────────────────────────────────────────────
print()
passed = sum(results)
total = len(results)

if passed == total:
    print(f"\033[92m✓ Phase 1 complete — {passed}/{total} checks passed.\033[0m")
    print("  You can now start Phase 2: core_vision/face_detector.py\n")
    sys.exit(0)
else:
    print(f"\033[91m✗ {total - passed}/{total} checks failed — fix errors above before Phase 2.\033[0m\n")
    sys.exit(1)