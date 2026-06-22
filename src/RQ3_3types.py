import json
import os
import pandas as pd
from tqdm import tqdm
from src.metaragT_2 import MetaRAGDetector
from config import *
# =====================================================================
#  (Configuration Area)
# =====================================================================

# 1. (Verifier)
ACTIVE_VERIFIER = "Gemma3-4B"

VERIFIER_REGISTRY = {
    "Gemma3-4B": {"model_name": "gemma3:4b", "is_local": True},
    "deepseek-v4-flash": {"model_name": "deepseek-v4-flash", "is_local": False}
}

# 2.(Strategy Selection)
# 可选值：
# - "paradox_only"    : Only detects logical paradoxes and confidence collapse
# - "strict"          : Fully capture perfectly symmetrical consistency judgments
# - "structure_aware" : Structural perception judgment introduces the original sentence's uncertainty and deviation threshold
EVAL_MODE = "structure_aware"

# 3. file_and_directory_configuration
DATASET_PATH = "./factoids/example_dataset.json"
CACHED_DATASET_PATH = "./factoids/example_dataset_T_2.json"

# Automatically maps the default output directory according to different modes
OUTPUT_DIR_MAPPING = {
    "paradox_only": "./factoids/results/rq3_results_g3_4b_paradox_only",
    "strict": "./factoids/results/rq3_results_g3_4b_strict_newdata",
    "structure_aware": "./factoids/results/rq3_results_g3_4b_structure_aware"
}
OUTPUT_DIR = OUTPUT_DIR_MAPPING[EVAL_MODE]


# =====================================================================
#    (Evaluation Logic Registry)
# =====================================================================

def evaluate_paradox_only(ground_truth, orig_res, syn_res_list, ant_res_list, category=None):

    is_escalated = False
    triggers = []
    failures = []

    syn_answers = set(syn_res_list)
    ant_answers = set(ant_res_list)

    #  1： (All NOT SURE)
    if syn_answers == {'NOT SURE'} and ant_answers == {'NOT SURE'}:
        is_escalated = True
        triggers.append('confidence collapse')
        failures.append('All NOT SURE')

    #  2： (Logical Paradox)
    else:
        has_positive_yes = orig_res == 'YES' or 'YES' in syn_answers
        has_positive_no = orig_res == 'NO' or 'NO' in syn_answers

        has_negative_yes = 'YES' in ant_answers
        has_negative_no = 'NO' in ant_answers

        if has_positive_yes and has_negative_yes:
            is_escalated = True
            triggers.append('logical paradox')
            failures.append('Contradictory Both YES')

        if has_positive_no and has_negative_no:
            is_escalated = True
            if 'logical paradox' not in triggers:
                triggers.append('logical paradox')
            failures.append('Contradictory Both NO')


    if not is_escalated:
        if ground_truth and orig_res in ['NO', 'NOT SURE']:
            return False, 'Silent Failure', 'Missed True Fact'
        if not ground_truth and orig_res == 'YES':
            return False, 'Silent Failure', 'Missed Hallucination'

    if is_escalated:
        return True, ' + '.join(triggers), ' + '.join(failures)

    return False, 'None', 'Success'


def evaluate_strict(ground_truth, orig_res, syn_res_list, ant_res_list, category=None):

    is_escalated = False
    triggers = []
    failures = []

    if ground_truth:

        if orig_res in ["NO", "NOT SURE"]:
            is_escalated = True
            triggers.append("original")
            failures.append("False Negative / Not Sure")


        if any(res in ["NO", "NOT SURE"] for res in syn_res_list):
            is_escalated = True
            triggers.append("synonym inconsistency")
            failures.append("Inconsistent on Synonym")


        if any(res in ["YES", "NOT SURE"] for res in ant_res_list):
            is_escalated = True
            triggers.append("antonym inconsistency")
            failures.append("Failed to reject Antonym")

    else:

        if orig_res in ["YES", "NOT SURE"]:
            is_escalated = True
            triggers.append("original")
            failures.append("False Positive / Not Sure")


        if any(res in ["YES", "NOT SURE"] for res in syn_res_list):
            is_escalated = True
            triggers.append("synonym inconsistency")
            failures.append("Inconsistent on Synonym (False Positive)")


        if any(res in ["NO", "NOT SURE"] for res in ant_res_list):
            is_escalated = True
            triggers.append("antonym inconsistency")
            failures.append("Inconsistent on Antonym")

    if is_escalated:
        return True, " + ".join(triggers), " + ".join(failures)

    return False, "None", "None"


def evaluate_structure_aware(ground_truth, orig_res, syn_res_list, ant_res_list, category=None):

    is_escalated = False
    triggers = []
    failures = []

    # ==========================================
    #  1： (Original Instability)
    # ==========================================
    if orig_res == "NOT SURE":
        is_escalated = True
        triggers.append("original uncertainty")
        failures.append("Original Factoid NOT SURE")

    # ==========================================
    #  2： (Logical Paradox)
    # ==========================================
    syn_answers = set(syn_res_list)
    ant_answers = set(ant_res_list)

    has_positive_yes = orig_res == 'YES' or 'YES' in syn_answers
    has_positive_no = orig_res == 'NO' or 'NO' in syn_answers
    has_negative_yes = 'YES' in ant_answers
    has_negative_no = 'NO' in ant_answers

    if has_positive_yes and has_negative_yes:
        is_escalated = True
        if 'logical paradox' not in triggers:
            triggers.append('logical paradox')
        failures.append('Contradictory Both YES')

    if has_positive_no and has_negative_no:
        is_escalated = True
        if 'logical paradox' not in triggers:
            triggers.append('logical paradox')
        failures.append('Contradictory Both NO')

    # ==========================================
    #  3： (Anomaly Threshold)
    # ==========================================
    if orig_res in ["YES", "NO"]:
        syn_errors = 0
        ant_errors = 0


        for s in syn_res_list:
            if s != orig_res:
                syn_errors += 1


        expected_ant = "NO" if orig_res == "YES" else "YES"
        for a in ant_res_list:
            if a != expected_ant:
                ant_errors += 1


        if ant_errors >= 2:
            is_escalated = True
            if "threshold anomaly" not in triggers:
                triggers.append("threshold anomaly")
            failures.append("Antonym Defense Breached (>=2 errors)")


        if syn_errors >= 1 and ant_errors >= 1:
            is_escalated = True
            if "threshold anomaly" not in triggers:
                triggers.append("threshold anomaly")
            failures.append("Dual-sided Defense Breached (Syn>=1 & Ant>=1)")


    if not is_escalated:
        if ground_truth and orig_res in ['NO', 'NOT SURE']:
            return False, 'Silent Failure', 'Missed True Fact'
        if not ground_truth and orig_res == 'YES':
            return False, 'Silent Failure', 'Missed Hallucination'

    if is_escalated:
        unique_triggers = list(dict.fromkeys(triggers))
        return True, ' + '.join(unique_triggers), ' + '.join(failures)

    return False, 'None', 'Success'



STRATEGY_ROUTER = {
    "paradox_only": evaluate_paradox_only,
    "strict": evaluate_strict,
    "structure_aware": evaluate_structure_aware
}


# =====================================================================
#  (Statistics Modules)
# =====================================================================

def analyze_and_save_stats(results, stats_file, mode):

    if not results:
        print(" If the result list is empty, the report cannot be generated.")
        return

    print("\n" + "=" * 50)
    print(f" Generating statistical analysis reports for the mode [{mode}]...")

    if mode == "paradox_only":
        df_data = []
        for r in results:
            df_data.append({
                "Category": r["source_category"],
                "Escalated": 1 if r["meta_rag_evaluation"]["escalation_triggered"] else 0,
                "Failure_Type": r["meta_rag_evaluation"]["small_verifier_failure_type"],
                "Trigger_Source": r["meta_rag_evaluation"]["trigger_source"]
            })
        df = pd.DataFrame(df_data)

        escalation_rate = df.groupby("Category")["Escalated"].mean().map(lambda x: f"{x:.2%}").to_frame(
            "Escalation Rate")

        fail_type_stats = pd.crosstab(df["Category"], df["Failure_Type"])
        if "None" in fail_type_stats.columns:
            fail_type_stats.drop(columns=["None"], inplace=True)

        trigger_stats = pd.crosstab(df["Category"], df["Trigger_Source"])
        if "None" in trigger_stats.columns:
            trigger_stats.drop(columns=["None"], inplace=True)

        final_stats = pd.concat([escalation_rate, fail_type_stats, trigger_stats], axis=1).fillna(0)
        final_stats.to_csv(stats_file)
        print(final_stats.to_string())

    elif mode == "strict":
        df_data = []
        for r in results:
            df_data.append({
                "Category": r["source_category"],
                "Escalated": 1 if r["meta_rag_evaluation"]["escalation_triggered"] else 0,
                "Trigger_Source": r["meta_rag_evaluation"]["trigger_source"]
            })
        df = pd.DataFrame(df_data)

        escalation_rate = df.groupby("Category")["Escalated"].mean().map(lambda x: f"{x:.2%}").to_frame(
            "Escalation Rate")

        df["Hits_Original"] = df["Trigger_Source"].apply(lambda x: 1 if "original" in str(x) else 0)
        df["Hits_Synonym"] = df["Trigger_Source"].apply(lambda x: 1 if "synonym" in str(x) else 0)
        df["Hits_Antonym"] = df["Trigger_Source"].apply(lambda x: 1 if "antonym" in str(x) else 0)
        df["Hits_Composite(>=2)"] = df["Trigger_Source"].apply(lambda x: 1 if "+" in str(x) else 0)

        independent_stats = df.groupby("Category")[
            ["Hits_Original", "Hits_Synonym", "Hits_Antonym", "Hits_Composite(>=2)"]].sum()
        independent_stats.columns = [" Original", " Synonym", " Antonym", "total_number_of_collapses_and_repetitions"]

        exact_combinations = pd.crosstab(df["Category"], df["Trigger_Source"])
        if "None" in exact_combinations.columns:
            exact_combinations.drop(columns=["None"], inplace=True)
        exact_combinations = exact_combinations.add_prefix("detailed_combinations: ")

        final_stats = pd.concat([escalation_rate, independent_stats, exact_combinations], axis=1).fillna(0)
        final_stats.to_csv(stats_file)
        print(final_stats.iloc[:, :5].to_string())

    elif mode == "structure_aware":
        df_data = []
        for r in results:
            gt = r["ground_truth_supported"]
            orig_clean = r["probes"]["original"]["verifier_output"]
            is_escalated = r["meta_rag_evaluation"]["escalation_triggered"]
            trigger_src = str(r["meta_rag_evaluation"]["trigger_source"])

            is_orig_correct = (gt == True and orig_clean == "YES") or (not gt and orig_clean == "NO")

            if is_escalated and not is_orig_correct:
                status = "Valid Escalation"
            elif is_escalated and is_orig_correct:
                status = "Over_escalation"
            elif not is_escalated and is_orig_correct:
                status = "Success Pass"
            elif not is_escalated and not is_orig_correct:
                status = "Missed Detection"

            df_data.append({
                "Category": r["source_category"],
                "Escalated": 1 if is_escalated else 0,
                "Trigger_Source": trigger_src,
                "Status": status
            })
        df = pd.DataFrame(df_data)

        escalation_rate = df.groupby("Category")["Escalated"].mean().map(lambda x: f"{x:.2%}").to_frame(
            "Escalation Rate")

        status_dummies = pd.get_dummies(df["Status"]).groupby(df["Category"]).sum()
        for col in ["Valid Escalation", "Over_escalation", "Missed Detection", "Success Pass"]:
            if col not in status_dummies.columns:
                status_dummies[col] = 0

        status_dummies = status_dummies[["Missed Detection", "Over_escalation", "Valid Escalation", "Success Pass"]]
        status_dummies = status_dummies.add_prefix("status: ")

        df[" Category Prior"] = df["Trigger_Source"].apply(lambda x: 1 if "category prior" in x else 0)
        df[" Original Uncertainty"] = df["Trigger_Source"].apply(lambda x: 1 if "original uncertainty" in x else 0)
        df[" Logical Paradox"] = df["Trigger_Source"].apply(lambda x: 1 if "logical paradox" in x else 0)
        df[" Threshold Anomaly"] = df["Trigger_Source"].apply(lambda x: 1 if "threshold anomaly" in x else 0)
        df["total_number_of_collapses_and_repetitions"] = df["Trigger_Source"].apply(lambda x: 1 if "+" in x else 0)

        independent_stats = df.groupby("Category")[
            [" Category Prior", " Original Uncertainty", " Logical Paradox", " Threshold Anomaly",
             "total_number_of_collapses_and_repetitions"]
        ].sum()

        exact_combinations = pd.crosstab(df["Category"], df["Trigger_Source"])
        if "None" in exact_combinations.columns:
            exact_combinations.drop(columns=["None"], inplace=True)
        exact_combinations = exact_combinations.add_prefix("detailed_combinations: ")

        final_stats = pd.concat([escalation_rate, status_dummies, independent_stats, exact_combinations],
                                axis=1).fillna(0)
        final_stats.to_csv(stats_file)

        cols_to_print = ["Escalation Rate"] + list(status_dummies.columns)
        print(final_stats[cols_to_print].to_string())


# =====================================================================
# (Core Experiment Loop)
# =====================================================================

def run_experiment_single_model():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if ACTIVE_VERIFIER not in VERIFIER_REGISTRY:
        raise ValueError(f"Model {ACTIVE_VERIFIER} is not registered in the configuration dictionary!")

    active_config = VERIFIER_REGISTRY[ACTIVE_VERIFIER]
    output_file = os.path.join(OUTPUT_DIR, f"rq3_details_{ACTIVE_VERIFIER}.json")
    stats_file = os.path.join(OUTPUT_DIR, f"rq3_stats_{ACTIVE_VERIFIER}.csv")


    load_path = CACHED_DATASET_PATH if os.path.exists(CACHED_DATASET_PATH) else DATASET_PATH
    print(f" load_the_test_set: {load_path}")
    with open(load_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)


    print(f" initialize_the_verifier_model_validation_only: {ACTIVE_VERIFIER}")
    verifier = MetaRAGDetector(
        verifier_model=active_config["model_name"],
        use_local_verifier=active_config["is_local"]
    )

    generator = None
    results = []
    dataset_modified = False


    print(f" start_probe_testing_mode_eval_mode..")
    for idx, item in enumerate(tqdm(dataset)):
        source_factoid = item.get("factoid")
        context = item.get("context")
        question = item.get("question")
        ground_truth_supported = item.get("context_support")
        factoid_type = item.get("factoid_type")


        if "mutations_list" not in item:
            if generator is None:
                print("   [system_notification] awaken Generator (DeepSeek API)...")
                generator = MetaRAGDetector(model="deepseek-chat", verifier_model="dummy", use_local_verifier=False)

            try:
                syns, ants = generator.generate_mutations(question, source_factoid, n=2)
                item["mutations_list"] = {
                    "synonyms": syns if syns else [],
                    "antonyms": ants if ants else []
                }
                dataset_modified = True
            except Exception as e:
                print(f"mutation_generation_failed ID {item.get('id')}: {e}")
                continue

        syn_texts = item["mutations_list"].get("synonyms", [])
        ant_texts = item["mutations_list"].get("antonyms", [])


        orig_raw = verifier.verify_statement(source_factoid, context).strip().upper()
        orig_clean = "YES" if "YES" in orig_raw else ("NO" if "NO" in orig_raw else "NOT SURE")


        syn_cleans = []
        for s_text in syn_texts:
            s_raw = verifier.verify_statement(s_text, context).strip().upper()
            syn_cleans.append("YES" if "YES" in s_raw else ("NO" if "NO" in s_raw else "NOT SURE"))


        ant_cleans = []
        for a_text in ant_texts:
            a_raw = verifier.verify_statement(a_text, context).strip().upper()
            ant_cleans.append("YES" if "YES" in a_raw else ("NO" if "NO" in a_raw else "NOT SURE"))


        eval_func = STRATEGY_ROUTER.get(EVAL_MODE)
        if not eval_func:
            raise ValueError(f"unsupported_review_modes: {EVAL_MODE}")

        is_escalated, trigger_src, fail_type = eval_func(
            ground_truth_supported, orig_clean, syn_cleans, ant_cleans, factoid_type
        )

        results.append({
            "seq_id": item.get("id"),
            "question": question,
            "context": context,
            "source_factoid": source_factoid,
            "source_category": factoid_type,
            "ground_truth_supported": ground_truth_supported,
            "probes": {
                "original": {"text": source_factoid, "verifier_output": orig_clean},
                "synonyms": [{"text": t, "output": o} for t, o in zip(syn_texts, syn_cleans)],
                "antonyms": [{"text": t, "output": o} for t, o in zip(ant_texts, ant_cleans)]
            },
            "meta_rag_evaluation": {
                "escalation_triggered": is_escalated,
                "trigger_source": trigger_src,
                "small_verifier_failure_type": fail_type
            }
        })

        if idx % 50 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)


    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


    if dataset_modified:
        print(" 将新生成的变异句保存到缓存数据集...")
        with open(CACHED_DATASET_PATH, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)


    analyze_and_save_stats(results, stats_file, EVAL_MODE)

    print(f" 测试完成！详细日志已存为 {output_file}")
    print(f" 统计报表已存为 {stats_file}\n")


if __name__ == "__main__":
    run_experiment_single_model()