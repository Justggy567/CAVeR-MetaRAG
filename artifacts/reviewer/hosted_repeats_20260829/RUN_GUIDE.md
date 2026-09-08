# Hosted DeepSeek repeat runs

The package uses frozen probes and a fresh application cache for each replicate.
Do not copy a primary-run cache into any replicate cache path.

Run the following commands from the repository root after setting DEEPSEEK_API_KEY:

    python main.py run --config artifacts/reviewer/hosted_repeats_20260829/configs/asqa_top5_76_replicate_01.json
    python main.py run --config artifacts/reviewer/hosted_repeats_20260829/configs/asqa_top5_76_replicate_02.json
    python main.py run --config artifacts/reviewer/hosted_repeats_20260829/configs/asqa_top5_76_replicate_03.json
    python main.py run --config artifacts/reviewer/hosted_repeats_20260829/configs/haluevalqa_3206_replicate_01.json
    python main.py run --config artifacts/reviewer/hosted_repeats_20260829/configs/haluevalqa_3206_replicate_02.json
    python main.py run --config artifacts/reviewer/hosted_repeats_20260829/configs/haluevalqa_3206_replicate_03.json

After all run manifests report status=complete:

    python main.py analyze-hosted-repeats --config configs/reviewer/hosted_repeats_analysis_20260829.json
