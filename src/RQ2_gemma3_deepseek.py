import json
import random
import pandas as pd
import os
import time
from src.metarag03 import MetaRAGDetector
from config import *

# ================= configure_the_area =================


DATA_FILE = DATASET_MAPPING.get("example_ASQA")

OUTPUT_CSV = RQ2_OUTPUT_CSV

FACTOID_OUTPUT_CSV = RQ2_FACTOID_OUTPUT_CSV

THRESHOLD = THRESHOLD

RANDOM_SEED = RANDOM_SEED

SUMMARY_JSON = RQ2_SUMMARY_JSON
SMALL_VERIFIER = SMALL_VERIFIER
STAGE2_VERIFIER = STAGE2_VERIFIER
USE_LOCAL = True
N_MUTATIONS = N_MUTATIONS

# RQ2 main_rule_configuration
ESCALATION_POLICY = ESCALATION_POLICY   # strict / structure_aware / paradox_only
MIN_ANOMALY_COUNT = MIN_ANOMALY_COUNT

USE_RISK_PRIOR = USE_RISK_PRIOR

VERIFIER_MODEL = f"{SMALL_VERIFIER} -> DeepSeek API ({ESCALATION_POLICY})"
# =============================================================


def load_local_data(filepath, sample_size=None, seed=42):
    print(f" local_data_is_loading: {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"error: no_documents_found {filepath}，please_confirm_the_path_is_correct.")

    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                data = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f" data_loading_failed: {e}")
        return []

    if sample_size and len(data) > sample_size:
        random.seed(seed)
        data = random.sample(data, sample_size)

    print(f" loading_successfully {len(data)} data_strips。")
    return data


def init_confusion():
    return {"TP": 0, "FP": 0, "TN": 0, "FN": 0}


def update_confusion(conf, pred_positive: bool, gold_positive: bool):
    if pred_positive and gold_positive:
        conf["TP"] += 1
    elif pred_positive and not gold_positive:
        conf["FP"] += 1
    elif not pred_positive and gold_positive:
        conf["FN"] += 1
    else:
        conf["TN"] += 1


def compute_prf(conf):
    tp, fp, tn, fn = conf["TP"], conf["FP"], conf["TN"], conf["FN"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    return precision, recall, f1, accuracy


def append_run_row(results, factoid_rows, item_id, q, context, answer_text, answer_role, gold_positive, detector_metrics, final_score, threshold):
    stage1_score = detector_metrics.get("stage1_final_score", 0.0)
    cascade_score = detector_metrics.get("cascade_final_score", final_score)

    stage1_pred = stage1_score >= threshold
    cascade_pred = cascade_score >= threshold
    stage1_correct = (stage1_pred == gold_positive)
    cascade_correct = (cascade_pred == gold_positive)

    response_escalated = detector_metrics.get("escalation_count", 0) > 0
    response_rescued = response_escalated and (not stage1_correct) and cascade_correct
    response_hurt = response_escalated and stage1_correct and (not cascade_correct)

    api_decompose_tokens = detector_metrics.get("api_decompose_tokens", 0)
    api_mutation_tokens = detector_metrics.get("api_mutation_tokens", 0)
    api_stage2_verify_tokens = detector_metrics.get("api_stage2_verify_tokens", 0)
    local_verify_tokens = detector_metrics.get("local_prompt_tokens", 0) + detector_metrics.get("local_completion_tokens", 0)
    total_cloud_tokens = detector_metrics.get("api_prompt_tokens", 0) + detector_metrics.get("api_completion_tokens", 0)
    total_tokens = total_cloud_tokens + local_verify_tokens

    results.append({
        "ID": item_id,
        "Question": q,
        "Context_Snippet": context[:80] + "..." if len(context) > 80 else context,
        "Answer_Type": answer_role,
        "Gold_Label": int(gold_positive),
        "Answer_Text": answer_text,

        "Stage1_Final_Score": round(stage1_score, 4),
        "Cascade_Final_Score": round(cascade_score, 4),
        "Stage1_Pred": int(stage1_pred),
        "Cascade_Pred": int(cascade_pred),
        "Stage1_Correct": int(stage1_correct),
        "Cascade_Correct": int(cascade_correct),

        "Response_Escalated": int(response_escalated),
        "Response_Rescued": int(response_rescued),
        "Response_Hurt": int(response_hurt),
        "Factoids_Total": detector_metrics.get("total_factoids", 0),
        "Factoid_Escalation_Count": detector_metrics.get("escalation_count", 0),
        "Escalation_Reasons": " | ".join(detector_metrics.get("escalation_reasons", [])),

        "Latency_s": round(detector_metrics.get("total_latency_sec", 0.0), 2),
        "Tokens_Total": int(total_tokens),
        "Tokens_Cloud_Total": int(total_cloud_tokens),
        "Tokens_Local_Verify": int(local_verify_tokens),
        "Tokens_API_Decompose": int(api_decompose_tokens),
        "Tokens_API_Mutation": int(api_mutation_tokens),
        "Tokens_API_Stage2_Verify": int(api_stage2_verify_tokens),
    })

    for flog in detector_metrics.get("factoid_logs", []):
        factoid_rows.append({
            "ID": item_id,
            "Question": q,
            "Answer_Type": answer_role,
            "Gold_Label": int(gold_positive),
            "Factoid_ID": flog.get("factoid_id"),
            "Factoid_Text": flog.get("factoid_text", ""),
            "Escalated": int(flog.get("escalated", False)),
            "Trigger_Reasons": " | ".join(flog.get("trigger_reasons", [])),
            "Stage1_Factoid_Score": flog.get("stage1_factoid_score", 0.0),
            "Cascade_Factoid_Score": flog.get("cascade_factoid_score", 0.0),
            "Stage1_Syn_Decisions": json.dumps(flog.get("stage1_syn_decisions", []), ensure_ascii=False),
            "Stage1_Ant_Decisions": json.dumps(flog.get("stage1_ant_decisions", []), ensure_ascii=False),
            "Final_Syn_Decisions": json.dumps(flog.get("final_syn_decisions", []), ensure_ascii=False),
            "Final_Ant_Decisions": json.dumps(flog.get("final_ant_decisions", []), ensure_ascii=False),
            "Syn_Anomaly_Count": flog.get("syn_anomaly_count", 0),
            "Ant_Anomaly_Count": flog.get("ant_anomaly_count", 0),
            "Syn_Anomaly_Ratio": flog.get("syn_anomaly_ratio", 0.0),
            "Ant_Anomaly_Ratio": flog.get("ant_anomaly_ratio", 0.0),
            "Logical_Paradox": int(flog.get("logical_paradox", False)),
            "Both_Yes": int(flog.get("both_yes", False)),
            "Both_No": int(flog.get("both_no", False)),
            "Risk_Prior": flog.get("risk_prior", ""),
        })

    return {
        "stage1_pred": stage1_pred,
        "cascade_pred": cascade_pred,
        "stage1_correct": stage1_correct,
        "cascade_correct": cascade_correct,
        "response_escalated": response_escalated,
        "response_rescued": response_rescued,
        "response_hurt": response_hurt,
        "response_escalated_wrong": response_escalated and (not stage1_correct),
        "factoids_total": detector_metrics.get("total_factoids", 0),
        "factoid_escalations": detector_metrics.get("escalation_count", 0),
        "latency": detector_metrics.get("total_latency_sec", 0.0),
        "total_tokens": total_tokens,
        "cloud_tokens": total_cloud_tokens,
        "local_verify_tokens": local_verify_tokens,
        "api_decompose_tokens": api_decompose_tokens,
        "api_mutation_tokens": api_mutation_tokens,
        "api_stage2_verify_tokens": api_stage2_verify_tokens,
    }


def run_benchmark():
    print(f" initialize_the_metarag_cascade_detector ({VERIFIER_MODEL})...")
    detector = MetaRAGDetector(
        verifier_model=SMALL_VERIFIER,
        large_verifier_model=STAGE2_VERIFIER,
        use_local_verifier=USE_LOCAL,
        escalation_policy=ESCALATION_POLICY,
        min_anomaly_count=MIN_ANOMALY_COUNT,
        use_risk_prior=USE_RISK_PRIOR,
    )

    test_samples = load_local_data(DATA_FILE, seed=RANDOM_SEED)
    if not test_samples:
        return

    results = []
    factoid_rows = []

    stage1_conf = init_confusion()
    cascade_conf = init_confusion()

    total_latency = 0.0
    total_tokens = 0
    total_cloud_tokens = 0
    total_local_verify_tokens = 0
    total_api_decompose_tokens = 0
    total_api_mutation_tokens = 0
    total_api_stage2_verify_tokens = 0

    total_responses = 0
    total_escalated_responses = 0
    total_rescued_responses = 0
    total_hurt_responses = 0
    total_small_correct = 0
    total_cascade_correct = 0
    total_escalated_wrong_responses = 0

    total_factoids = 0
    total_escalated_factoids = 0

    start_time = time.time()
    print(f"\n  begin_batch_evaluation {len(test_samples)} sample_strip...")

    for i, item in enumerate(test_samples):
        print(f"[{i + 1}/{len(test_samples)}] processing...", end="\r")
        item_id = item.get('id', i + 1)
        q = item.get('question', '')
        context = item.get('knowledge', '') or item.get('context', '')


        h_ans = item.get('hallucinated_answer', '')
        h_score, h_metrics = detector.detect(q, h_ans, context, n_mutations=N_MUTATIONS)
        h_stats = append_run_row(results, factoid_rows, item_id, q, context, h_ans, "hallucinated", True, h_metrics, h_score, THRESHOLD)

        update_confusion(stage1_conf, h_stats["stage1_pred"], True)
        update_confusion(cascade_conf, h_stats["cascade_pred"], True)


        r_ans = item.get('right_answer', '')
        r_score, r_metrics = detector.detect(q, r_ans, context, n_mutations=N_MUTATIONS)
        r_stats = append_run_row(results, factoid_rows, item_id, q, context, r_ans, "right", False, r_metrics, r_score, THRESHOLD)

        update_confusion(stage1_conf, r_stats["stage1_pred"], False)
        update_confusion(cascade_conf, r_stats["cascade_pred"], False)

        for stats in [h_stats, r_stats]:
            total_responses += 1
            total_escalated_responses += int(stats["response_escalated"])
            total_rescued_responses += int(stats["response_rescued"])
            total_hurt_responses += int(stats["response_hurt"])
            total_small_correct += int(stats["stage1_correct"])
            total_cascade_correct += int(stats["cascade_correct"])
            total_escalated_wrong_responses += int(stats["response_escalated_wrong"])

            total_factoids += int(stats["factoids_total"])
            total_escalated_factoids += int(stats["factoid_escalations"])

            total_latency += stats["latency"]
            total_tokens += stats["total_tokens"]
            total_cloud_tokens += stats["cloud_tokens"]
            total_local_verify_tokens += stats["local_verify_tokens"]
            total_api_decompose_tokens += stats["api_decompose_tokens"]
            total_api_mutation_tokens += stats["api_mutation_tokens"]
            total_api_stage2_verify_tokens += stats["api_stage2_verify_tokens"]

    # ====== summary_indicators ======
    s_precision, s_recall, s_f1, s_acc = compute_prf(stage1_conf)
    c_precision, c_recall, c_f1, c_acc = compute_prf(cascade_conf)

    response_escalation_rate = total_escalated_responses / total_responses if total_responses else 0.0
    factoid_escalation_rate = total_escalated_factoids / total_factoids if total_factoids else 0.0
    rescue_rate = total_rescued_responses / total_escalated_responses if total_escalated_responses else 0.0
    error_cond_rescue_rate = total_rescued_responses / total_escalated_wrong_responses if total_escalated_wrong_responses else 0.0
    harm_rate = total_hurt_responses / total_escalated_responses if total_escalated_responses else 0.0
    net_correct_gain = total_cascade_correct - total_small_correct
    escalation_efficiency = (net_correct_gain / total_escalated_responses * 100) if total_escalated_responses else 0.0

    time_avg = total_latency / total_responses if total_responses else 0.0
    tokens_avg_total = total_tokens / total_responses if total_responses else 0.0
    tokens_avg_cloud = total_cloud_tokens / total_responses if total_responses else 0.0
    tokens_avg_local_verify = total_local_verify_tokens / total_responses if total_responses else 0.0
    tokens_avg_api_decompose = total_api_decompose_tokens / total_responses if total_responses else 0.0
    tokens_avg_api_mutation = total_api_mutation_tokens / total_responses if total_responses else 0.0
    tokens_avg_api_stage2_verify = total_api_stage2_verify_tokens / total_responses if total_responses else 0.0

    print("\n" + "=" * 60)
    print(f" final_evaluation_report ({VERIFIER_MODEL})")
    print("=" * 60)
    print("[Small-only baseline]")
    print(f"Precision : {s_precision:.2%}")
    print(f"Recall    : {s_recall:.2%}")
    print(f"F1 Score  : {s_f1:.2%}")
    print(f"Accuracy  : {s_acc:.2%}")
    print("-" * 60)
    print("[Cascade main_result]")
    print(f"Precision : {c_precision:.2%}")
    print(f"Recall    : {c_recall:.2%}")
    print(f"F1 Score  : {c_f1:.2%}")
    print(f"Accuracy  : {c_acc:.2%}")
    print(f"Response Escalation Rate : {response_escalation_rate:.2%} ({total_escalated_responses}/{total_responses})")
    print(f"Factoid Escalation Rate  : {factoid_escalation_rate:.2%} ({total_escalated_factoids}/{total_factoids})")
    print(f"Rescue Rate (response)   : {rescue_rate:.2%} ({total_rescued_responses}/{total_escalated_responses if total_escalated_responses else 0})")
    print(f"Error-conditional Rescue Rate: {error_cond_rescue_rate:.2%} ({total_rescued_responses}/{total_escalated_wrong_responses if total_escalated_wrong_responses else 0})")
    print(f"Harm Rate (response)     : {harm_rate:.2%} ({total_hurt_responses}/{total_escalated_responses if total_escalated_responses else 0})")
    print(f"Net Correct Gain         : {net_correct_gain:+d}")
    print(f"Escalation Efficiency    : {escalation_efficiency:.2f} correct gains / 100 escalations")
    print("-" * 60)
    print("[cost_breakdown]")
    print(f"Avg Latency              : {time_avg:.2f} sec/response")
    print(f"Avg Tokens Total         : {tokens_avg_total:.0f} /response")
    print(f"Avg Tokens Cloud Total   : {tokens_avg_cloud:.0f} /response")
    print(f"Avg Tokens Local Verify  : {tokens_avg_local_verify:.0f} /response")
    print(f"Avg API Decompose        : {tokens_avg_api_decompose:.0f} /response")
    print(f"Avg API Mutation         : {tokens_avg_api_mutation:.0f} /response")
    print(f"Avg API Stage2 Verify    : {tokens_avg_api_stage2_verify:.0f} /response")
    print("=" * 60)

    total_time = time.time() - start_time
    print(f"\n Test complete! Total time: {total_time:.2f} seconds")

    try:
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        pd.DataFrame(factoid_rows).to_csv(FACTOID_OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f" the_response_level_log_has_been_saved_to {OUTPUT_CSV}")
        print(f" factoid_level_logs_have_been_saved_to: {FACTOID_OUTPUT_CSV}")
    except Exception as e:
        print(f" failed_to_save_csv: {e}")

    summary = {
        "model": VERIFIER_MODEL,
        "policy": ESCALATION_POLICY,
        "small_verifier": SMALL_VERIFIER,
        "stage2_verifier": STAGE2_VERIFIER,
        "threshold": THRESHOLD,
        "n_mutations": N_MUTATIONS,
        "sample_size": len(test_samples),

        "Stage1_Precision": s_precision,
        "Stage1_Recall": s_recall,
        "Stage1_F1": s_f1,
        "Stage1_Accuracy": s_acc,

        "Cascade_Precision": c_precision,
        "Cascade_Recall": c_recall,
        "Cascade_F1": c_f1,
        "Cascade_Accuracy": c_acc,

        "Response_Escalation_Rate": response_escalation_rate,
        "Factoid_Escalation_Rate": factoid_escalation_rate,
        "Rescue_Rate_Response": rescue_rate,
        "Error_Conditional_Rescue_Rate": error_cond_rescue_rate,
        "Harm_Rate_Response": harm_rate,
        "Net_Correct_Gain": net_correct_gain,
        "Escalation_Efficiency_per_100": escalation_efficiency,

        "Time_avg": time_avg,
        "Tokens_avg_total": tokens_avg_total,
        "Tokens_avg_cloud_total": tokens_avg_cloud,
        "Tokens_avg_local_verify": tokens_avg_local_verify,
        "Tokens_avg_api_decompose": tokens_avg_api_decompose,
        "Tokens_avg_api_mutation": tokens_avg_api_mutation,
        "Tokens_avg_api_stage2_verify": tokens_avg_api_stage2_verify,
    }

    os.makedirs(os.path.dirname(SUMMARY_JSON), exist_ok=True)
    all_summaries = []
    if os.path.exists(SUMMARY_JSON):
        with open(SUMMARY_JSON, 'r', encoding='utf-8') as f:
            all_summaries = json.load(f)
    all_summaries = [s for s in all_summaries if s.get("model") != VERIFIER_MODEL]
    all_summaries.append(summary)
    with open(SUMMARY_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=4, ensure_ascii=False)
        print(f" the_summary_indicators_have_been_updated_to: {SUMMARY_JSON}")


if __name__ == "__main__":
    run_benchmark()
