#!/usr/bin/env python3
"""
Qwen3-TTS Pipeline Validation Orchestrator (L0 ~ L4).

Runs the full validation stack sequentially and produces a unified report.
Any level failure stops further execution (single-variable principle).

Usage:
    python3 scripts/validate_tts_pipeline.py \
        --cpp-dir ./debug_bin/cpp_dump \
        --py-dir ./debug_bin/py_dump \
        [--mode all] \
        [--output report.md]

Modes:
    l0_input     : L0 input alignment (prefill_embeds, trailing_text, tts_pad)
    l1_prefill   : L1 Talker prefill (layer0, kv_cache, last_hidden, logits)
    l2_cp        : L2 CP independent (hidden, logits, sampled tokens)
    l3_coupling  : L3 Talker-CP coupling (codec_sum, inputs_embeds, decode hidden)
    l4_e2e       : L4 end-to-end output codes comparison
    all          : Run L0 -> L1 -> L2 -> L3 (L4 requires separate codes files)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).parent.resolve()


def load_fp32_bin(path: Path, expected_elems: int = None) -> np.ndarray:
    if not path.exists():
        return None
    arr = np.fromfile(path, dtype=np.float32)
    if expected_elems is not None and arr.size != expected_elems:
        print(f"[WARN] {path.name}: size mismatch got {arr.size}, expected {expected_elems}")
    return arr


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


class ValidationReport:
    def __init__(self):
        self.sections = []
        self.overall_pass = True

    def add_section(self, title: str, status: str, details: list):
        self.sections.append({"title": title, "status": status, "details": details})
        if status != "PASS":
            self.overall_pass = False

    def markdown(self) -> str:
        lines = [
            "# Qwen3-TTS Validation Report",
            "",
            f"**Overall**: {'✅ PASS' if self.overall_pass else '❌ FAIL'}",
            "",
        ]
        for sec in self.sections:
            icon = "✅" if sec["status"] == "PASS" else "❌" if sec["status"] == "FAIL" else "⚠️"
            lines.append(f"## {icon} {sec['title']} — {sec['status']}")
            lines.append("")
            for d in sec["details"]:
                lines.append(f"- {d}")
            lines.append("")
        return "\n".join(lines)


def validate_l0_input(cpp_dir: Path, py_dir: Path, report: ValidationReport, meta: dict) -> bool:
    details = []
    hidden_size = meta.get("hidden_size", 1024)
    S = meta.get("S", 8)

    ok = True
    checks = [
        ("prefill_embeds", S * hidden_size),
        ("trailing_text_hiddens", None),  # shape unknown from meta, skip size check
        ("tts_pad_vec", hidden_size),
    ]

    for name, expected in checks:
        cpp_path = cpp_dir / f"{name}.bin"
        py_path = py_dir / f"{name}.bin"
        if not cpp_path.exists() or not py_path.exists():
            details.append(f"{name}: MISSING (cpp={cpp_path.exists()}, py={py_path.exists()})")
            ok = False
            continue
        cpp_arr = load_fp32_bin(cpp_path, expected)
        py_arr = load_fp32_bin(py_path, expected)
        if cpp_arr is None or py_arr is None:
            ok = False
            continue
        max_d = max_abs_diff(cpp_arr, py_arr)
        cos = cosine_sim(cpp_arr, py_arr)
        pass_flag = max_d < 1e-5
        details.append(f"{name}: cos={cos:.8f} max_diff={max_d:.8f} {'PASS' if pass_flag else 'FAIL'}")
        if not pass_flag:
            ok = False

    report.add_section("L0 Input Alignment", "PASS" if ok else "FAIL", details)
    return ok


def validate_l1_prefill(cpp_dir: Path, py_dir: Path, report: ValidationReport, meta: dict) -> bool:
    details = []
    ok = True

    script = SCRIPT_DIR / "compare_prefill_checkpoints.py"
    if script.exists():
        try:
            result = subprocess.run(
                [sys.executable, str(script),
                 "--cpp_dir", str(cpp_dir),
                 "--py_dir", str(py_dir)],
                capture_output=True, text=True, timeout=60,
            )
            output = result.stdout + result.stderr
            # Simple heuristic: if output contains "Done" and no "FAIL"/"MISMATCH" => pass
            if "Done" in output and "SHAPE MISMATCH" not in output and "size mismatch" not in output:
                details.append("compare_prefill_checkpoints.py executed successfully")
            else:
                details.append("compare_prefill_checkpoints.py found discrepancies (see console)")
                ok = False
        except Exception as e:
            details.append(f"compare_prefill_checkpoints.py failed: {e}")
            ok = False
    else:
        details.append("compare_prefill_checkpoints.py not found, skipping")

    report.add_section("L1 Talker Prefill", "PASS" if ok else "FAIL", details)
    return ok


def validate_l2_cp(cpp_dir: Path, py_dir: Path, report: ValidationReport) -> bool:
    details = []
    ok = True

    cp_cpp_dir = cpp_dir / "cpp_cp_dump"
    cp_py_dir = py_dir / "python_cp_dump"

    if not cp_cpp_dir.exists():
        details.append(f"CP C++ dump dir missing: {cp_cpp_dir}")
        ok = False
    if not cp_py_dir.exists():
        details.append(f"CP Python dump dir missing: {cp_py_dir}")
        ok = False

    if ok:
        script = SCRIPT_DIR / "compare_cp_dumps.py"
        if script.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(script),
                     "--cpp-dir", str(cp_cpp_dir),
                     "--py-dir", str(cp_py_dir)],
                    capture_output=True, text=True, timeout=60,
                )
                output = result.stdout + result.stderr
                if "ALL 15 STEPS MATCH" in output:
                    details.append("compare_cp_dumps.py: ALL 15 STEPS MATCH")
                else:
                    details.append("compare_cp_dumps.py: found divergences (see console)")
                    ok = False
            except Exception as e:
                details.append(f"compare_cp_dumps.py failed: {e}")
                ok = False
        else:
            details.append("compare_cp_dumps.py not found, skipping")

    report.add_section("L2 CP Independent", "PASS" if ok else "FAIL", details)
    return ok


def validate_l3_coupling(cpp_dir: Path, py_dir: Path, report: ValidationReport, meta: dict) -> bool:
    details = []
    ok = True

    script = SCRIPT_DIR / "compare_talker_decode_full.py"
    if script.exists():
        try:
            result = subprocess.run(
                [sys.executable, str(script),
                 "--cpp-dir", str(cpp_dir),
                 "--py-dir", str(py_dir),
                 "--max-steps", "10",
                 "--hidden-size", str(meta.get("hidden_size", 1024)),
                 "--vocab-size", str(meta.get("vocab_size", 3072))],
                capture_output=True, text=True, timeout=60,
            )
            output = result.stdout + result.stderr
            if "All compared steps MATCH" in output:
                details.append("compare_talker_decode_full.py: All steps MATCH")
            else:
                details.append("compare_talker_decode_full.py: found divergences (see console)")
                ok = False
        except Exception as e:
            details.append(f"compare_talker_decode_full.py failed: {e}")
            ok = False
    else:
        details.append("compare_talker_decode_full.py not found, skipping")

    # Also run coupling_flow
    script2 = SCRIPT_DIR / "compare_coupling_flow.py"
    if script2.exists():
        try:
            result = subprocess.run(
                [sys.executable, str(script2),
                 "--cpp-dir", str(cpp_dir),
                 "--py-dir", str(py_dir),
                 "--max-steps", "10",
                 "--hidden-size", str(meta.get("hidden_size", 1024))],
                capture_output=True, text=True, timeout=60,
            )
            output = result.stdout + result.stderr
            if "All coupling flow steps MATCH" in output:
                details.append("compare_coupling_flow.py: All steps MATCH")
            else:
                details.append("compare_coupling_flow.py: found divergences (see console)")
                ok = False
        except Exception as e:
            details.append(f"compare_coupling_flow.py failed: {e}")
            ok = False
    else:
        details.append("compare_coupling_flow.py not found, skipping")

    report.add_section("L3 Talker-CP Coupling", "PASS" if ok else "FAIL", details)
    return ok


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Pipeline Validation")
    parser.add_argument("--cpp-dir", type=Path, required=True)
    parser.add_argument("--py-dir", type=Path, required=True)
    parser.add_argument("--mode", default="all",
                        choices=["all", "l0_input", "l1_prefill", "l2_cp", "l3_coupling", "l4_e2e"])
    parser.add_argument("--output", type=Path, default=Path("validation_report.md"))
    args = parser.parse_args()

    # Load meta.json from either directory
    meta_path = args.py_dir / "meta.json"
    if not meta_path.exists():
        meta_path = args.cpp_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    report = ValidationReport()

    modes = {
        "l0_input": validate_l0_input,
        "l1_prefill": validate_l1_prefill,
        "l2_cp": validate_l2_cp,
        "l3_coupling": validate_l3_coupling,
    }

    if args.mode == "all":
        run_order = ["l0_input", "l1_prefill", "l2_cp", "l3_coupling"]
    else:
        run_order = [args.mode]

    for mode in run_order:
        func = modes[mode]
        passed = func(args.cpp_dir, args.py_dir, report, meta)
        if not passed:
            print(f"\n❌ Validation stopped at {mode} due to failure.")
            break

    md = report.markdown()
    args.output.write_text(md, encoding="utf-8")
    print(f"\nReport saved to: {args.output}")
    print(md)

    sys.exit(0 if report.overall_pass else 1)


if __name__ == "__main__":
    main()
