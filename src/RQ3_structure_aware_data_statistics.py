import json
import os
import pandas as pd
from tqdm import tqdm

from src.metaragT_2 import MetaRAGDetector
from config import *

# =====================================================================
#  (Configuration Area)
# =====================================================================


VERIFIER_REGISTRY = {
    "Gemma3-4B": {"model_name": "gemma3:4b", "is_local": True},
    "deepseek-api": {"model_name": "deepseek-chat", "is_local": False}
}

DATASET_PATH = "./factoids/example_dataset.json"
CACHED_DATASET_PATH = "./factoids/example_dataset_T_2.json"

OUTPUT_DIR = "./factoids"



def evaluate_failure_multi_structure_aware(ground_truth, orig_res, syn_res_list, ant_res_list, category):

    is_escalated = False
    triggers = []
    failures = []

    # ==========================================
    # (Original Instability)
    # ==========================================
    if orig_res == "NOT SURE":
        is_escalated = True
        triggers.append("original uncertainty")
        failures.append("Original Factoid NOT SURE")

    # ==========================================
    #  (Logical Paradox)
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
    #  (Anomaly Threshold)
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


# =====================================================================
#   (Atomic Real-time Save)
# =====================================================================

def save_realtime_results(results_list, output_file_path):

    tmp_path = output_file_path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, output_file_path)


def is_original_correct(ground_truth_supported, orig_output):
    return (ground_truth_supported is True and orig_output == "YES") or \
           (ground_truth_supported is False and orig_output == "NO")


# =====================================================================
#  (Analytical Report Compiler)
# =====================================================================

def compile_reports(results, gemma_stats_path, deepseek_stats_path, comp_stats_path):

    print("\n" + "=" * 60)
    print(" Running a built-in automatic analysis engine to generate final experimental reports...")

    gemma_rows = []
    for r in results:
        gt = r["ground_truth_supported"]
        g_data = r["gemma"]
        is_escalated = g_data["escalation_triggered"]
        trigger_src = str(g_data["trigger_source"])
        is_orig_correct = g_data["is_correct"]

        if is_escalated and not is_orig_correct:
            status = "Valid Escalation"
        elif is_escalated and is_orig_correct:
            status = "Over_escalation"
        elif not is_escalated and is_orig_correct:
            status = "Success Pass"
        else:
            status = "Missed Detection"

        gemma_rows.append({
            "Category": r["source_category"],
            "Escalated": 1 if is_escalated else 0,
            "Trigger_Source": trigger_src,
            "Status": status
        })

    df_gemma = pd.DataFrame(gemma_rows)
    esc_gemma = df_gemma.groupby("Category")["Escalated"].mean().map(lambda x: f"{x:.2%}").to_frame("Escalation Rate")
    status_gemma = pd.get_dummies(df_gemma["Status"]).groupby(df_gemma["Category"]).sum()
    for col in ["Valid Escalation", "Over_escalation", "Missed Detection", "Success Pass"]:
        if col not in status_gemma.columns:
            status_gemma[col] = 0
    status_gemma = status_gemma[["Missed Detection", "Over_escalation", "Valid Escalation", "Success Pass"]].add_prefix("status: ")

    df_gemma["Original Uncertainty"] = df_gemma["Trigger_Source"].apply(lambda x: 1 if "original uncertainty" in x else 0)
    df_gemma["Logical Paradox"] = df_gemma["Trigger_Source"].apply(lambda x: 1 if "logical paradox" in x else 0)
    df_gemma["Threshold Anomaly"] = df_gemma["Trigger_Source"].apply(lambda x: 1 if "threshold anomaly" in x else 0)
    df_gemma["total_number_of_collapses_and_repetitions"] = df_gemma["Trigger_Source"].apply(lambda x: 1 if "+" in x else 0)

    hits_gemma = df_gemma.groupby("Category")[["Original Uncertainty", "Logical Paradox", "Threshold Anomaly", "total_number_of_collapses_and_repetitions"]].sum()
    pd.concat([esc_gemma, status_gemma, hits_gemma], axis=1).fillna(0).to_csv(gemma_stats_path)
    print(f" Gemma3-4B the_independent_analysis_report_has_been_saved_to: {gemma_stats_path}")


    ds_rows = []
    for r in results:
        gt = r["ground_truth_supported"]
        ds_data = r["deepseek"]
        is_escalated = ds_data["escalation_triggered"]
        trigger_src = str(ds_data["trigger_source"])
        is_orig_correct = ds_data["is_correct"]

        if is_escalated and not is_orig_correct:
            status = "Valid Escalation"
        elif is_escalated and is_orig_correct:
            status = "Over_escalation"
        elif not is_escalated and is_orig_correct:
            status = "Success Pass"
        else:
            status = "Missed Detection"

        ds_rows.append({
            "Category": r["source_category"],
            "Escalated": 1 if is_escalated else 0,
            "Trigger_Source": trigger_src,
            "Status": status
        })

    df_ds = pd.DataFrame(ds_rows)
    esc_ds = df_ds.groupby("Category")["Escalated"].mean().map(lambda x: f"{x:.2%}").to_frame("Escalation Rate")
    status_ds = pd.get_dummies(df_ds["Status"]).groupby(df_ds["Category"]).sum()
    for col in ["Valid Escalation", "Over_escalation", "Missed Detection", "Success Pass"]:
        if col not in status_ds.columns:
            status_ds[col] = 0
    status_ds = status_ds[["Missed Detection", "Over_escalation", "Valid Escalation", "Success Pass"]].add_prefix("status: ")

    df_ds["Original Uncertainty"] = df_ds["Trigger_Source"].apply(lambda x: 1 if "original uncertainty" in x else 0)
    df_ds["Logical Paradox"] = df_ds["Trigger_Source"].apply(lambda x: 1 if "logical paradox" in x else 0)
    df_ds["Threshold Anomaly"] = df_ds["Trigger_Source"].apply(lambda x: 1 if "threshold anomaly" in x else 0)
    df_ds["total_number_of_collapses_and_repetitions"] = df_ds["Trigger_Source"].apply(lambda x: 1 if "+" in x else 0)

    hits_ds = df_ds.groupby("Category")[["Original Uncertainty", "Logical Paradox", "Threshold Anomaly", "total_number_of_collapses_and_repetitions"]].sum()
    pd.concat([esc_ds, status_ds, hits_ds], axis=1).fillna(0).to_csv(deepseek_stats_path)
    print(f"The DeepSeek-API independent analysis report has been saved to: {deepseek_stats_path}")


    comp_rows = []
    for r in results:
        comp_rows.append({
            "Category": r["source_category"],
            "Escalated": r["gemma"]["escalation_triggered"],
            "Status": r["comparative_metrics"]["status"],
            "Rescued": r["comparative_metrics"]["rescued"],
            "Harmed": r["comparative_metrics"]["harmed"]
        })

    df_comp = pd.DataFrame(comp_rows)
    summary = []
    for cat, sub in df_comp.groupby("Category"):
        n = len(sub)
        escalation_count = int(sub["Escalated"].sum())
        missed_count = int((sub["Status"] == "Missed Detection").sum())
        Over_escalation_count = int((sub["Status"] == "Over_escalation").sum())
        success_count = int((sub["Status"] == "Success Pass").sum())
        valid_count = int((sub["Status"] == "Valid Escalation").sum())

        rescued_count = int(sub["Rescued"].sum())
        harmed_count = int(sub["Harmed"].sum())

        rescue_rate = rescued_count / escalation_count if escalation_count > 0 else 0.0
        err_cond_rescue = rescued_count / valid_count if valid_count > 0 else 0.0
        harm_rate = harmed_count / escalation_count if escalation_count > 0 else 0.0
        efficiency = ((rescued_count - harmed_count) / escalation_count * 100) if escalation_count > 0 else 0.0

        summary.append({
            "Factoid Type": cat,
            "N": n,
            "Escalation": escalation_count / n,
            "Missed Detection": missed_count / n,
            "Over_escalation": Over_escalation_count / n,
            "Success Pass": success_count / n,
            "Valid Escalation": valid_count / n,
            "Rescue Rate": rescue_rate,
            "Error-cond. Rescue": err_cond_rescue,
            "Harm Rate": harm_rate,
            "Escalation Efficiency": efficiency
        })

    summary_df = pd.DataFrame(summary).sort_values("Factoid Type")

    formatted_df = summary_df.copy()
    rate_cols = [
        "Escalation", "Missed Detection", "Over_escalation", "Success Pass",
        "Valid Escalation", "Rescue Rate", "Error-cond. Rescue", "Harm Rate"
    ]
    for c in rate_cols:
        formatted_df[c] = formatted_df[c].map(lambda x: f"{x*100:.2f}%")
    formatted_df["Escalation Efficiency"] = formatted_df["Escalation Efficiency"].map(lambda x: f"{x:.2f}")

    formatted_df.to_csv(comp_stats_path, index=False, encoding="utf-8-sig")
    print(f" The multi-model joint comparative evaluation report has been saved to: {comp_stats_path}\n")

    print("--------------------  (Gemma3-4B to DeepSeek-API) --------------------")
    print(formatted_df.to_markdown(index=False))
    print("=" * 60)


# =====================================================================
#  (Double-Verifier Sequential Pipeline)
# =====================================================================

def run_combined_experiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_file = os.path.join(OUTPUT_DIR, "rq3_combined_details.json")
    gemma_stats_file = os.path.join(OUTPUT_DIR, "rq3_stats_Gemma3-4B.csv")
    deepseek_stats_file = os.path.join(OUTPUT_DIR, "rq3_stats_deepseek-api.csv")
    comp_stats_file = os.path.join(OUTPUT_DIR, "rq3_comparative_stats.csv")


    completed_ids = set()
    results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                completed_ids = {r["seq_id"] for r in results}
            print(f" Detected existing progress, successfully loaded {len(completed_ids)} processed records. These items have been skipped.")
        except Exception as e:
            print(f" If reading an existing file fails ({e}), a new evaluation will be restarted.")
            results = []


    load_path = CACHED_DATASET_PATH if os.path.exists(CACHED_DATASET_PATH) else DATASET_PATH
    print(f" load_the_test_dataset: {load_path}")
    with open(load_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)


    gemma_verifier = MetaRAGDetector(
        verifier_model=VERIFIER_REGISTRY["Gemma3-4B"]["model_name"],
        use_local_verifier=VERIFIER_REGISTRY["Gemma3-4B"]["is_local"]
    )
    deepseek_verifier = MetaRAGDetector(
        verifier_model=VERIFIER_REGISTRY["deepseek-api"]["model_name"],
        use_local_verifier=VERIFIER_REGISTRY["deepseek-api"]["is_local"]
    )

    generator = None
    dataset_modified = False

    print(" Start running the dual-model synchronous detection and evaluation cycle...")
    for idx, item in enumerate(tqdm(dataset)):
        seq_id = item.get("id")
        if seq_id in completed_ids:
            continue

        source_factoid = item.get("factoid")
        context = item.get("context")
        question = item.get("question")
        ground_truth_supported = item.get("context_support")
        factoid_type = item.get("factoid_type")


        if "mutations_list" not in item:
            if generator is None:
                print("   Mutation Generator (DeepSeek API)...")
                generator = MetaRAGDetector(model="deepseek-chat", verifier_model="dummy", use_local_verifier=False)

            try:
                syns, ants = generator.generate_mutations(question, source_factoid, n=2)
                item["mutations_list"] = {
                    "synonyms": syns if syns else [],
                    "antonyms": ants if ants else []
                }
                dataset_modified = True
            except Exception as e:
                print(f" mutation_generation_failed ID {seq_id}: {e}")
                continue

        syn_texts = item["mutations_list"].get("synonyms", [])
        ant_texts = item["mutations_list"].get("antonyms", [])

        # ---  detection_phase A: Gemma3-4B ---
        gemma_orig_raw = gemma_verifier.verify_statement(source_factoid, context).strip().upper()
        gemma_orig_clean = "YES" if "YES" in gemma_orig_raw else ("NO" if "NO" in gemma_orig_raw else "NOT SURE")

        gemma_syn_cleans = []
        for s_text in syn_texts:
            s_raw = gemma_verifier.verify_statement(s_text, context).strip().upper()
            gemma_syn_cleans.append("YES" if "YES" in s_raw else ("NO" if "NO" in s_raw else "NOT SURE"))

        gemma_ant_cleans = []
        for a_text in ant_texts:
            a_raw = gemma_verifier.verify_statement(a_text, context).strip().upper()
            gemma_ant_cleans.append("YES" if "YES" in a_raw else ("NO" if "NO" in a_raw else "NOT SURE"))

        g_escalated, g_trigger_src, g_fail_type = evaluate_failure_multi_structure_aware(
            ground_truth_supported, gemma_orig_clean, gemma_syn_cleans, gemma_ant_cleans, factoid_type
        )

        # ---  detection_phase B: DeepSeek-API ---
        ds_orig_raw = deepseek_verifier.verify_statement(source_factoid, context).strip().upper()
        ds_orig_clean = "YES" if "YES" in ds_orig_raw else ("NO" if "NO" in ds_orig_raw else "NOT SURE")

        ds_syn_cleans = []
        for s_text in syn_texts:
            s_raw = deepseek_verifier.verify_statement(s_text, context).strip().upper()
            ds_syn_cleans.append("YES" if "YES" in s_raw else ("NO" if "NO" in s_raw else "NOT SURE"))

        ds_ant_cleans = []
        for a_text in ant_texts:
            a_raw = deepseek_verifier.verify_statement(a_text, context).strip().upper()
            ds_ant_cleans.append("YES" if "YES" in a_raw else ("NO" if "NO" in a_raw else "NOT SURE"))

        ds_escalated, ds_trigger_src, ds_fail_type = evaluate_failure_multi_structure_aware(
            ground_truth_supported, ds_orig_clean, ds_syn_cleans, ds_ant_cleans, factoid_type
        )


        gemma_orig_correct = is_original_correct(ground_truth_supported, gemma_orig_clean)
        deep_orig_correct = is_original_correct(ground_truth_supported, ds_orig_clean)

        if g_escalated and (not gemma_orig_correct):
            comp_status = "Valid Escalation"
        elif g_escalated and gemma_orig_correct:
            comp_status = "Over_escalation"
        elif (not g_escalated) and gemma_orig_correct:
            comp_status = "Success Pass"
        else:
            comp_status = "Missed Detection"

        rescued = g_escalated and (not gemma_orig_correct) and deep_orig_correct
        harmed = g_escalated and gemma_orig_correct and (not deep_orig_correct)


        results.append({
            "seq_id": seq_id,
            "question": question,
            "context": context,
            "source_category": factoid_type,
            "ground_truth_supported": ground_truth_supported,
            "probes": {
                "original": source_factoid,
                "synonyms": syn_texts,
                "antonyms": ant_texts
            },
            "gemma": {
                "original_output": gemma_orig_clean,
                "synonyms_outputs": gemma_syn_cleans,
                "antonyms_outputs": gemma_ant_cleans,
                "escalation_triggered": g_escalated,
                "trigger_source": g_trigger_src,
                "failure_type": g_fail_type,
                "is_correct": gemma_orig_correct
            },
            "deepseek": {
                "original_output": ds_orig_clean,
                "synonyms_outputs": ds_syn_cleans,
                "antonyms_outputs": ds_ant_cleans,
                "escalation_triggered": ds_escalated,
                "trigger_source": ds_trigger_src,
                "failure_type": ds_fail_type,
                "is_correct": deep_orig_correct
            },
            "comparative_metrics": {
                "status": comp_status,
                "rescued": rescued,
                "harmed": harmed
            }
        })


        save_realtime_results(results, output_file)


        if dataset_modified and idx % 20 == 0:
            with open(CACHED_DATASET_PATH, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)
            dataset_modified = False


    if dataset_modified:
        print(" Save the latest probes to the variant sentence cache database...")
        with open(CACHED_DATASET_PATH, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

    compile_reports(results, gemma_stats_file, deepseek_stats_file, comp_stats_file)


if __name__ == "__main__":
    run_combined_experiment()