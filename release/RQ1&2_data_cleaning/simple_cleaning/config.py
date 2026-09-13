

from __future__ import annotations

import os
from pathlib import Path


# Optional: Run only one or both datasets simultaneously.
# ["asqa"]
# ["haluevalqa"]
# ["asqa", "haluevalqa"]
DATASETS_TO_RUN = ["haluevalqa"]

# Both datasets go directly to NO1 from the original input; for the official full run, change to None.
TEST_LIMIT = None

# Every time you modify the prompt, model, or TEST_LIMIT, you should use a new name and avoid mixing it with old results.
RUN_TAG = "full_10000_original_cleaning"

# True: Stop after executing NO1.
# False: Execute the entire route.
DRY_RUN = False

# Keep it True after interruption, and you can continue from the JSONL ledger.
RESUME = True

# DeepSeek official API configuration. The key is only read from the environment variable DEEPSEEK_API_KEY.
MODEL_NAME = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
THINKING_MODE = "disabled"

# Call parameters.
BOOLEAN_MAX_TOKENS = 5
INJECTION_MAX_TOKENS = 512
MAX_API_RETRIES = 4
MAX_INJECTION_ATTEMPTS = 3
REQUEST_INTERVAL_SECONDS = 0.1
CHECKPOINT_EVERY = 25


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "data"

# When testing in other directories, you can set METARAG_DATA_DIR; if it is adjacent to the data folder, no setting is needed.
DATA_DIR = Path(os.getenv("METARAG_DATA_DIR", str(DEFAULT_DATA_DIR)))
OUTPUT_ROOT = PROJECT_ROOT / "runs_simple_v4"
INPUT_FILES = {
    "asqa": DATA_DIR / "asqa_eval_data_top5.json",
    "haluevalqa": DATA_DIR / "halueval_qa_data.json",
}


def validate_config() -> None:
    Check the configuration before any file/API operations.
    allowed = {"asqa", "haluevalqa"}
    if not DATASETS_TO_RUN:
        raise ValueError("DATASETS_TO_RUN cannot be empty.")
    unknown = set(DATASETS_TO_RUN) - allowed
    if unknown:
        raise ValueError(f"Unsupported dataset：{sorted(unknown)}")
    if TEST_LIMIT is not None and TEST_LIMIT < 1:
        raise ValueError("TEST_LIMIT must be None or a positive integer.")
    if not RUN_TAG.strip():
        raise ValueError("RUN_TAG cannot be empty.")
    if CHECKPOINT_EVERY < 1:
        raise ValueError("CHECKPOINT_EVERY must be at least 1.")
    if REQUEST_INTERVAL_SECONDS < 0:
        raise ValueError("REQUEST_INTERVAL_SECONDS cannot be negative.")
    if MODEL_NAME != "deepseek-v4-flash":
        raise ValueError(
            "This reproduction protocol fixes MODEL_NAME='deepseek-v4-flash'; if the model is changed, a new version of the protocol should be created instead of reusing the results of this version."
        )
    if THINKING_MODE != "disabled":
        raise ValueError("This reproduction protocol sets THINKING_MODE='disabled'.")
