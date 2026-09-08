"""Command-line entry point for the canonical implementation."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from .audit import audit_data_config, environment_report, parameter_report, write_audit
from .cascade import CanonicalCascade
from .clients import StaticSequenceClient
from .configuration import load_config
from .experiment import prepare_from_config, run_from_config
from .labels import BinaryTarget, Verdict
from .probes import make_probe_set
from .routing import RoutingPolicy, compute_signals, should_escalate
from .policy_analysis import run_policy_suite
from .publication import export_publication_tables
from .rq1_analysis import run_rq1_analysis
from .rq3_analysis import run_rq3_analysis
from .schema import VerificationBundle
from .verifiers import EvidenceVerifier


def validate_method() -> dict[str, int | bool]:
    combinations = 0
    escalations = 0
    for values in product(Verdict, repeat=5):
        bundle = VerificationBundle(
            original=values[0],
            synonyms=(values[1], values[2]),
            antonyms=(values[3], values[4]),
        )
        signals = compute_signals(bundle)
        routed = should_escalate(signals, RoutingPolicy.STRUCTURE_AWARE)
        expected = (
            signals.original_uncertainty
            or signals.explicit_paradox
            or signals.antonym_anomaly_count >= 2
            or (signals.synonym_anomaly_count >= 1 and signals.antonym_anomaly_count >= 1)
        )
        if routed != expected:
            raise AssertionError(f"Routing formula mismatch for {values}")
        combinations += 1
        escalations += int(routed)

    isolated_synonym = VerificationBundle(
        original=Verdict.YES,
        synonyms=(Verdict.NOT_SURE, Verdict.YES),
        antonyms=(Verdict.NO, Verdict.NO),
    )
    if should_escalate(compute_signals(isolated_synonym)):
        raise AssertionError("An isolated synonym-only anomaly must not escalate.")
    isolated_antonym = VerificationBundle(
        original=Verdict.YES,
        synonyms=(Verdict.YES, Verdict.YES),
        antonyms=(Verdict.NOT_SURE, Verdict.NO),
    )
    if should_escalate(compute_signals(isolated_antonym)):
        raise AssertionError("An isolated non-paradox antonym anomaly must not escalate.")
    return {
        "combinations_checked": combinations,
        "escalating_combinations": escalations,
        "paper_formula_verified": True,
    }


def smoke_test() -> dict[str, object]:
    probes = make_probe_set(
        "smoke:f0",
        "Paris is in France.",
        ("Paris lies in France.", "France contains Paris."),
        ("Paris is not in France.", "France does not contain Paris."),
    )
    stage1_client = StaticSequenceClient(["NO", "YES", "NO", "YES", "NO"], model_id="small-static")
    stage2_client = StaticSequenceClient(["YES", "YES", "YES", "NO", "NO"], model_id="strong-static")
    engine = CanonicalCascade(
        stage1_verifier=EvidenceVerifier(stage1_client),
        stage2_verifier=EvidenceVerifier(stage2_client),
        routing_policy=RoutingPolicy.STRUCTURE_AWARE,
    )
    result = engine.process_response(
        response_id="smoke",
        question_id="smoke",
        gold_target=BinaryTarget.SUPPORTED,
        context="Paris is the capital city of France.",
        probe_sets=(probes,),
    )
    if not result.factoids[0].routed:
        raise AssertionError("Smoke factoid should be routed.")
    if len(result.factoids[0].observations) != 10:
        raise AssertionError("Stage 1 and Stage 2 must each verify the same five statements.")
    if result.stage1_hallucinated is not True or result.final_hallucinated is not False:
        raise AssertionError("Smoke transition should be rescued.")
    return {
        "status": "ok",
        "stage1_score": result.stage1_score,
        "final_score": result.final_score,
        "routed": result.factoids[0].routed,
        "route_reasons": list(result.factoids[0].route_reasons),
        "verification_calls": len(result.factoids[0].observations),
        "probe_hash_reused": result.factoids[0].probe_hash == probes.probe_hash,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Canonical paper-aligned CAVeR-MetaRAG experiment runner."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="Freeze factoids and probes.")
    prepare_parser.add_argument("--config", required=True, type=Path)
    run_parser = subparsers.add_parser("run", help="Run a verifier configuration on frozen probes.")
    run_parser.add_argument("--config", required=True, type=Path)
    subparsers.add_parser("validate-method", help="Exhaustively validate the paper routing formula.")
    subparsers.add_parser("smoke", help="Run an offline ten-call cascade smoke test.")
    audit_data_parser = subparsers.add_parser("audit-data", help="Audit frozen datasets and hashes.")
    audit_data_parser.add_argument("--config", required=True, type=Path)
    audit_data_parser.add_argument("--output", type=Path)
    environment_parser = subparsers.add_parser(
        "audit-environment", help="Capture hardware/software/model details."
    )
    environment_parser.add_argument("--output", required=True, type=Path)
    parameter_parser = subparsers.add_parser(
        "audit-parameters",
        help="Validate reviewer-requested model, decoding, latency, and resource parameters.",
    )
    parameter_parser.add_argument("--output", required=True, type=Path)
    policy_parser = subparsers.add_parser(
        "analyze-policies", help="Replay RQ2/RQ3 policies from one full matrix ledger."
    )
    policy_parser.add_argument("--config", required=True, type=Path)
    rq1_parser = subparsers.add_parser(
        "analyze-rq1", help="Consolidate four-verifier RQ1 runs with paired statistics."
    )
    rq1_parser.add_argument("--config", required=True, type=Path)
    rq3_parser = subparsers.add_parser(
        "analyze-rq3", help="Summarize the fixed main RQ3 policy by factoid type."
    )
    rq3_parser.add_argument("--config", required=True, type=Path)
    publication_parser = subparsers.add_parser(
        "export-publication",
        help="Export completed RQ1/RQ2/RQ3 analysis reports to manuscript CSV tables.",
    )
    publication_parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    if args.command == "validate-method":
        print(json.dumps(validate_method(), indent=2, sort_keys=True))
        return 0
    if args.command == "smoke":
        print(json.dumps(smoke_test(), indent=2, sort_keys=True))
        return 0
    if args.command == "audit-environment":
        output = args.output if args.output.is_absolute() else repository_root / args.output
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite environment report: {output}")
        write_audit(output, environment_report())
        print(output)
        return 0
    if args.command == "audit-parameters":
        output = args.output if args.output.is_absolute() else repository_root / args.output
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite parameter report: {output}")
        write_audit(output, parameter_report(repository_root))
        print(output)
        return 0
    config = load_config(args.config)
    if args.command == "audit-data":
        report = audit_data_config(config, repository_root=repository_root)
        if args.output:
            output = args.output if args.output.is_absolute() else repository_root / args.output
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite data audit: {output}")
            write_audit(output, report)
            print(output)
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare":
        path = prepare_from_config(config, repository_root=repository_root)
        print(path)
        return 0
    if args.command == "run":
        path = run_from_config(config, repository_root=repository_root)
        print(path)
        return 0
    if args.command == "analyze-policies":
        print(run_policy_suite(config, repository_root=repository_root))
        return 0
    if args.command == "analyze-rq1":
        print(run_rq1_analysis(config, repository_root=repository_root))
        return 0
    if args.command == "analyze-rq3":
        print(run_rq3_analysis(config, repository_root=repository_root))
        return 0
    if args.command == "export-publication":
        print(export_publication_tables(config, repository_root=repository_root))
        return 0
    parser.error("Unknown command.")
    return 2
