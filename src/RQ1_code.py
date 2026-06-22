import json
import random
import pandas as pd
import os
import time



from src.metaragT_2 import MetaRAGDetector
from config import *

# def run_benchmark(
#         dataset_path,
#         output_csv,
#         summary_json,
#         verifier_model="deepseek-chat"
# ):


# ================= configure_the_area =================


DATA_FILE =DATASET_MAPPING.get("example_ASQA")
OUTPUT_CSV = RQ1_CSV
THRESHOLD = THRESHOLD


SUMMARY_JSON = RQ1_JSON
# validator_model_configuration
VERIFIER_MODEL = VERIFIER_MODEL
USE_LOCAL = True
N_MUTATIONS = N_MUTATIONS
# =============================================================

def load_local_data(filepath):

    print(f"local_data_is_loading: {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: File {filepath} not found, please confirm the path is correct.")

    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:

            try:
                data = json.load(f)
            except json.JSONDecodeError:

                f.seek(0)
                data = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f" data_loading_failure {e}")
        return []

    print(f" successfully_loaded_len_data_data_entry")
    return data


def run_benchmark():

    detector = MetaRAGDetector(verifier_model=VERIFIER_MODEL, use_local_verifier=USE_LOCAL)
    test_samples = load_local_data(DATA_FILE)
    if not test_samples: return

    results = []

    TP = FN = TN = FP = 0
    total_latency = 0.0
    total_tokens = 0
    total_cloud_tokens = 0
    total_local_tokens = 0

    print(f"\n Start batch evaluation of {len(test_samples)} data entries...")
    start_time = time.time()

    for i, item in enumerate(test_samples):
        print(f"[{i + 1}/{len(test_samples)}] processing", end="\r")
        item_id = item.get('id', i + 1)
        q = item.get('question', '')
        context = item.get('context', '')


        h_ans = item.get('hallucinated_answer', '')
        h_score, h_metrics = detector.detect(q, h_ans, context, n_mutations=N_MUTATIONS)
        h_detected = h_score >= THRESHOLD
        if h_detected:
            TP += 1
        else:
            FN += 1


        r_ans = item.get('right_answer', '')
        r_score, r_metrics = detector.detect(q, r_ans, context, n_mutations=N_MUTATIONS)
        r_false_alarm = r_score >= THRESHOLD
        if r_false_alarm:
            FP += 1
        else:
            TN += 1



        r_run_latency = r_metrics.get("total_latency_sec", 0.0)
        r_run_tokens = sum(r_metrics.values()) - r_metrics["total_latency_sec"]


        r_cloud_tokens = r_metrics.get("api_prompt_tokens", 0) + r_metrics.get("api_completion_tokens", 0)


        r_local_tokens = r_metrics.get("local_prompt_tokens", 0) + r_metrics.get("local_completion_tokens", 0)


        total_latency += r_run_latency
        total_tokens += r_run_tokens
        total_cloud_tokens += r_cloud_tokens
        total_local_tokens += r_local_tokens


        H_run_latency = h_metrics.get("total_latency_sec", 0.0)
        H_run_tokens = sum(h_metrics.values()) - h_metrics["total_latency_sec"]


        H_cloud_tokens = h_metrics.get("api_prompt_tokens", 0) + h_metrics.get("api_completion_tokens", 0)


        H_local_tokens = h_metrics.get("local_prompt_tokens", 0) + h_metrics.get("local_completion_tokens", 0)


        total_latency += H_run_latency
        total_tokens += H_run_tokens
        total_cloud_tokens += H_cloud_tokens
        total_local_tokens += H_local_tokens


        results.append({
            "ID": item_id,
            "Question": q,
            "Context_Snippet": context[:50] + "..." if len(context) > 50 else context,
            "H_Answer": h_ans, "H_Score": h_score, "H_Detected(TP)": h_detected,
            "H_Latency(s)": round(H_run_latency, 2),
            "H_Tokens": int(H_run_tokens),
            "H_Cloud_Tokens": int(H_cloud_tokens),
            "H_Local_Tokens": int(H_local_tokens),

            "R_Answer": r_ans, "R_Score": r_score, "R_FalseAlarm(FP)": r_false_alarm,
            "R_Latency(s)": round(r_run_latency, 2),
            "R_Tokens": int(r_run_tokens),
            "R_Cloud_Tokens": int(r_cloud_tokens),
            "R_Local_Tokens": int(r_local_tokens),

        })


    total_runs = len(test_samples) * 2
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0.0
    Accuracy = (TP + TN) / total_runs if total_runs > 0 else 0.0
    Time_avg = total_latency / total_runs if total_runs > 0 else 0.0
    Tokens_avg = total_tokens / total_runs if total_runs > 0 else 0

    Cloud_Tokens_avg = total_cloud_tokens / total_runs if total_runs > 0 else 0
    Local_Tokens_avg = total_local_tokens / total_runs if total_runs > 0 else 0

    print("\n" + "=" * 50)
    print(f" final_evaluation_report ({VERIFIER_MODEL})")
    print("=" * 50)
    print(f"1. Precision  : {Precision:.2%}")
    print(f"2. Recall    : {Recall:.2%}")
    print(f"3. F1 Score   : {F1:.2%}")
    print(f"4. Accuracy  : {Accuracy:.2%}")
    print(f"5. Time(avg) : {Time_avg:.2f} PerSecond")
    print(f"6. Tokens(avg): {Tokens_avg:.0f} tokens_use")
    print(f"7. Cloud Tokens : {Cloud_Tokens_avg:.0f} tokens_use ")
    print(f"8. Local Tokens : {Local_Tokens_avg:.0f} tokens_use ")
    print("-" * 50)

    total_time = time.time() - start_time
    print(f"\n Test complete! Total time: {total_time:.2f} seconds")


    try:

        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f" detailed_log_saved_to_output_csv")
    except Exception as e:
        print(f" failed_to_save_csv {e}")


    summary = {
        "model": VERIFIER_MODEL,
        "Precision": Precision, "Recall": Recall, "F1": F1, "Accuracy": Accuracy,
        "Time_avg": Time_avg, "Tokens_avg": Cloud_Tokens_avg
    }


    all_summaries = []
    if os.path.exists(SUMMARY_JSON):
        with open(SUMMARY_JSON, 'r') as f:
            all_summaries = json.load(f)

    all_summaries = [s for s in all_summaries if s["model"] != VERIFIER_MODEL]
    all_summaries.append(summary)
    with open(SUMMARY_JSON, 'w') as f:
        json.dump(all_summaries, f, indent=4)
        print(f" Model performance metrics have been updated to: {SUMMARY_JSON}")

if __name__ == "__main__":
    run_benchmark()