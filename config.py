import os

# ==================================================
# API
# ==================================================
#TODO
DEEPSEEK_API_KEY = "sk-"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"

# ==================================================
# Dataset
# ==================================================

DATASET_MAPPING = {
    "example_ASQA": "./data/example_ASQA.json",
    "HaluEvalQA": "./data/HaluEvalQA.json",
    "ASQA96": "./data/ASQA_96.json",
    "Factoids":"./data/Factoid_dataset.json"
}

# ==================================================git remote add origin https://github.com/Justggy567/CAVeR-MetaRAG.git
# Global Experiment Settings
# ==================================================

THRESHOLD = 0.5
N_MUTATIONS = 2
RANDOM_SEED = 42


# ==================================================
# RQ1
# ==================================================
VERIFIER_MODEL="phi4"

# VERIFIER_MODELS = {
#         "deepseek-v4-flash": {"model": "deepseek-v4-flash", "is_local": False},
#
#         "Gemma3-4B": {"model": "gemma3:4b", "is_local": True},
#
#         "Phi4-14B": {"model": "phi4", "is_local": True},
#
#         "Gemma3-12B": {"model": "gemma3:12b", "is_local": True},
#     }

# ==================================================
# RQ2
# ==================================================


SMALL_VERIFIER = "gemma3:4b"
STAGE2_VERIFIER = "deepseek-v4-flash"

# RQ2 main_rule_configuration
ESCALATION_POLICY = "structure_aware"   # strict / structure_aware / paradox_only
MIN_ANOMALY_COUNT = 2
USE_RISK_PRIOR = False


RQ2_CONFIG = {
    "small_verifier": "gemma3:4b",
    "stage2_verifier": "deepseek-v4-flash",

    "escalation_policy": "structure_aware",

    "min_anomaly_count": 2,

    "use_risk_prior": False
}


# ==================================================
# RQ3
# ==================================================




# ==================================================
# Output
# ==================================================

RQ1_CSV= "./results/csv/demo.csv"
RQ1_JSON="./results/json/demo.json"

RQ2_OUTPUT_CSV="./results/csv/rq2_test_results.csv"

RQ2_FACTOID_OUTPUT_CSV = "./results/csv/rq2_test_results_factoids.csv"

RQ2_SUMMARY_JSON = "../results/json/rq2_test_results.json"