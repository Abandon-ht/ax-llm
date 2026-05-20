#!/usr/bin/env python3
"""
Automated validation pipeline: infer.py (Python AXEngine) vs C++ AXEngine dumps.

Steps:
1. Archive old dumps
2. Run infer.py with QWEN3_TTS_DUMP_DIR to generate Python dumps + C++ inputs
3. Sync C++ inputs to pyramid
4. Run C++ qwen3_tts_infer on pyramid with --debug-dump-dir
5. Download C++ dumps back
6. Run comparison scripts (L0-L3)
7. Generate timestamped report

Usage:
    python3 scripts/run_infer_vs_cpp_validation.py
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

# Config
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEBUG_BIN = PROJECT_ROOT / "debug_bin"
PY_DUMP_DIR = DEBUG_BIN / "py_dump"
CPP_DUMP_DIR = DEBUG_BIN / "cpp_dump"
ARCHIVE_DIR = DEBUG_BIN / "archive"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
BUILD_BIN = PROJECT_ROOT / "build" / "install" / "bin" / "qwen3_tts_infer"
PYRAMID_HOST = "pyramid"
PYRAMID_DEBUG_BIN = Path("/root/debug_bin")
PYRAMID_BUILD_BIN = Path("/root/build/bin/qwen3_tts_infer")


def run_cmd(cmd, cwd=None, check=True):
    print(f"[CMD] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {cmd}")
    return result


def archive_old_dumps():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = ARCHIVE_DIR / f"archive_{timestamp}"
    archive_path.mkdir(parents=True, exist_ok=True)

    # Move existing py_dump and cpp_dump to archive
    for sub in [PY_DUMP_DIR, CPP_DUMP_DIR]:
        if sub.exists():
            dest = archive_path / sub.name
            print(f"[ARCHIVE] Moving {sub} -> {dest}")
            shutil.move(str(sub), str(dest))

    # Also archive any loose files in debug_bin (except archive dir and meta)
    for item in DEBUG_BIN.iterdir():
        if item.is_file() and item.name not in {"meta.json", ".gitkeep"}:
            shutil.move(str(item), str(archive_path / item.name))

    print(f"[ARCHIVE] Old dumps archived to {archive_path}")
    return archive_path


def run_infer_py():
    """Run infer.py to generate Python dumps and C++ inputs."""
    env = os.environ.copy()
    env["QWEN3_TTS_DUMP_DIR"] = str(PY_DUMP_DIR)

    cmd = [
        sys.executable, str(PROJECT_ROOT / "infer.py"),
        "--no-do_sample",
        "--no-subtalker_dosample",
        "--qwen_tts_root", "Qwen3-TTS",
        "--hf_model_path", "rsp/Qwen3-TTS-12Hz-0.6B-Base/",
        "--talker_compiled_model_path", "rsp/Qwen3-TTS-12Hz-0.6B-Base-AX650/talker",
        "--code_predictor_compiled_model_path", "rsp/Qwen3-TTS-12Hz-0.6B-Base-AX650/code-predictor",
        "--ref_audio", "likes.wav",
        "--ref_text", "可莉喜欢毛茸茸的东西。比如嘟嘟可、蒲公英，还有雷泽的头发。",
        "--text", "为了守护蒙德城周边的安定，我曾经发动过不少次「远征」，但比起这一次，都算不上什么…比如清剿达达乌帕谷、联合千岩军扫荡石门、从鹰翔海滩出发迎击外海魔物…嗯？你说难怪在这些地方都遇不到什么强敌…我应该还是留了些下来给人练手的吧？",
        "--dump-cpp-input-dir", str(PY_DUMP_DIR),
        "--skip_wav_generation",
        "--max_new_tokens", "300",
        "--seed", "1234",
    ]

    print("=" * 70)
    print("Running infer.py (Python AXEngine) with dump enabled")
    print("=" * 70)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"infer.py failed with code {result.returncode}")

    # Verify outputs
    expected = [
        PY_DUMP_DIR / "prefill_embeds.bin",
        PY_DUMP_DIR / "meta.json",
        PY_DUMP_DIR / "tts_pad_vec.bin",
    ]
    for f in expected:
        if not f.exists():
            print(f"[WARN] Expected file missing: {f}")

    # Check for talker decode dumps
    talker_decode_dir = PY_DUMP_DIR / "python_talker_decode"
    if talker_decode_dir.exists():
        files = list(talker_decode_dir.glob("*.bin"))
        print(f"[INFO] Python talker decode dumps: {len(files)} files")
    else:
        print("[WARN] No python_talker_decode dumps found")

    # Check for CP dumps
    cp_dump_dir = PY_DUMP_DIR / "python_cp_dump"
    if cp_dump_dir.exists():
        files = list(cp_dump_dir.glob("*.bin"))
        print(f"[INFO] Python CP dumps: {len(files)} files")
    else:
        print("[WARN] No python_cp_dump dumps found")


def sync_to_pyramid():
    """Sync C++ inputs and binary to pyramid."""
    print("=" * 70)
    print("Syncing to pyramid")
    print("=" * 70)

    # Ensure pyramid has debug_bin dir with inputs
    run_cmd(["ssh", PYRAMID_HOST, f"mkdir -p {PYRAMID_DEBUG_BIN}"])

    # Sync inputs
    for fname in ["prefill_embeds.bin", "meta.json", "tts_pad_vec.bin"]:
        src = PY_DUMP_DIR / fname
        if src.exists():
            run_cmd(["scp", str(src), f"{PYRAMID_HOST}:{PYRAMID_DEBUG_BIN}/{fname}"])

    # Sync binary if needed
    result = subprocess.run(
        ["ssh", PYRAMID_HOST, f"test -f {PYRAMID_BUILD_BIN} && echo OK || echo MISSING"],
        capture_output=True, text=True
    )
    if "MISSING" in result.stdout or "--force-upload" in sys.argv:
        if BUILD_BIN.exists():
            run_cmd(["scp", str(BUILD_BIN), f"{PYRAMID_HOST}:{PYRAMID_BUILD_BIN}"])
        else:
            raise RuntimeError(f"Local binary not found: {BUILD_BIN}")


def run_cpp_on_pyramid():
    """Run C++ inference on pyramid with debug dump."""
    print("=" * 70)
    print("Running C++ inference on pyramid")
    print("=" * 70)

    cmd = (
        f"cd /root && {PYRAMID_BUILD_BIN} "
        f"~/rsp/Qwen3-TTS-12Hz-0.6B-Base-AX650/talker/ "
        f"{PYRAMID_DEBUG_BIN} "
        f"--max_new_tokens 300 "
        f"--seed 1234 "
        f"--debug-dump-dir {PYRAMID_DEBUG_BIN}/cpp_dump "
        f"--codec_eos_token_id 2150"
    )
    result = subprocess.run(
        ["ssh", PYRAMID_HOST, cmd],
        capture_output=False, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"C++ inference failed with code {result.returncode}")


def download_cpp_dumps():
    """Download C++ dumps from pyramid."""
    print("=" * 70)
    print("Downloading C++ dumps from pyramid")
    print("=" * 70)

    CPP_DUMP_DIR.mkdir(parents=True, exist_ok=True)

    # Use rsync or scp to download
    result = subprocess.run(
        ["scp", "-r", f"{PYRAMID_HOST}:{PYRAMID_DEBUG_BIN}/cpp_dump/", str(CPP_DUMP_DIR)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        # Try tar approach
        tar_cmd = (
            f"cd {PYRAMID_DEBUG_BIN} && tar czf cpp_dump.tar.gz cpp_dump/"
        )
        subprocess.run(["ssh", PYRAMID_HOST, tar_cmd], capture_output=True, text=True)
        subprocess.run(
            ["scp", f"{PYRAMID_HOST}:{PYRAMID_DEBUG_BIN}/cpp_dump.tar.gz", str(DEBUG_BIN)],
            capture_output=True, text=True
        )
        tar_path = DEBUG_BIN / "cpp_dump.tar.gz"
        if tar_path.exists():
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=DEBUG_BIN)
            # Move extracted files
            extracted = DEBUG_BIN / "cpp_dump"
            if extracted.exists():
                for item in extracted.iterdir():
                    dest = CPP_DUMP_DIR / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(str(dest))
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(dest))
                extracted.rmdir()

    # List downloaded files
    if CPP_DUMP_DIR.exists():
        files = list(CPP_DUMP_DIR.rglob("*.bin"))
        print(f"[INFO] C++ dump files downloaded: {len(files)}")


def run_comparisons():
    """Run L0-L3 comparison scripts."""
    print("=" * 70)
    print("Running comparisons")
    print("=" * 70)

    report_parts = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # L0: Input alignment
    print("\n--- L0: Input Alignment ---")
    meta_path = PY_DUMP_DIR / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    hidden_size = meta.get("hidden_size", 1024)
    S = meta.get("S", 8)

    l0_details = []
    l0_pass = True
    for name in ["prefill_embeds", "trailing_text_hiddens", "tts_pad_vec"]:
        cpp_f = CPP_DUMP_DIR / f"{name}.bin"
        py_f = PY_DUMP_DIR / f"{name}.bin"
        if not cpp_f.exists() or not py_f.exists():
            l0_details.append(f"{name}: MISSING (cpp={cpp_f.exists()}, py={py_f.exists()})")
            l0_pass = False
            continue
        import numpy as np
        cpp_arr = np.fromfile(cpp_f, dtype=np.float32)
        py_arr = np.fromfile(py_f, dtype=np.float32)
        max_d = float(np.max(np.abs(cpp_arr - py_arr)))
        cos = float(np.dot(cpp_arr, py_arr) / (np.linalg.norm(cpp_arr) * np.linalg.norm(py_arr))) if np.linalg.norm(cpp_arr) > 0 and np.linalg.norm(py_arr) > 0 else 0.0
        ok = max_d < 1e-5
        l0_details.append(f"{name}: cos={cos:.8f} max_diff={max_d:.8f} {'PASS' if ok else 'FAIL'}")
        if not ok:
            l0_pass = False
    report_parts.append(("L0 Input Alignment", l0_pass, l0_details))

    # L1: Talker Prefill
    print("\n--- L1: Talker Prefill ---")
    script = SCRIPTS_DIR / "compare_prefill_checkpoints.py"
    l1_pass = True
    l1_details = []
    if script.exists():
        result = subprocess.run(
            [sys.executable, str(script),
             "--cpp_dir", str(CPP_DUMP_DIR),
             "--py_dir", str(PY_DUMP_DIR)],
            capture_output=True, text=True, timeout=120,
        )
        output = result.stdout + result.stderr
        l1_details.append("compare_prefill_checkpoints.py output captured")
        # Write raw output
        (DEBUG_BIN / f"l1_prefill_output_{timestamp}.txt").write_text(output, encoding="utf-8")
        if "Done" in output and "SHAPE MISMATCH" not in output and "size mismatch" not in output.lower():
            l1_details.append("Script completed without structural errors")
        else:
            l1_details.append("Script found discrepancies (see l1_prefill_output_*.txt)")
            l1_pass = False
    else:
        l1_details.append("compare_prefill_checkpoints.py not found")
        l1_pass = False
    report_parts.append(("L1 Talker Prefill", l1_pass, l1_details))

    # L2: CP Independent
    print("\n--- L2: CP Independent ---")
    script = SCRIPTS_DIR / "compare_cp_dumps.py"
    l2_pass = True
    l2_details = []
    if script.exists():
        result = subprocess.run(
            [sys.executable, str(script),
             "--cpp-dir", str(CPP_DUMP_DIR / "cpp_cp_dump"),
             "--py-dir", str(PY_DUMP_DIR / "python_cp_dump"),
             "--frame", "0", "-v"],
            capture_output=True, text=True, timeout=120,
        )
        output = result.stdout + result.stderr
        l2_details.append("compare_cp_dumps.py output captured")
        (DEBUG_BIN / f"l2_cp_output_{timestamp}.txt").write_text(output, encoding="utf-8")
        if "ALL 15 STEPS MATCH" in output:
            l2_details.append("ALL 15 STEPS MATCH")
        else:
            l2_details.append("Found divergences (see l2_cp_output_*.txt)")
            l2_pass = False
    else:
        l2_details.append("compare_cp_dumps.py not found")
        l2_pass = False
    report_parts.append(("L2 CP Independent", l2_pass, l2_details))

    # L3: Talker Decode + Coupling
    print("\n--- L3: Talker Decode & Coupling ---")
    l3_pass = True
    l3_details = []

    script1 = SCRIPTS_DIR / "compare_talker_decode_full.py"
    if script1.exists():
        result = subprocess.run(
            [sys.executable, str(script1),
             "--cpp-dir", str(CPP_DUMP_DIR),
             "--py-dir", str(PY_DUMP_DIR),
             "--max-steps", "10",
             "--hidden-size", str(hidden_size),
             "--vocab-size", str(meta.get("vocab_size", 3072))],
            capture_output=True, text=True, timeout=120,
        )
        output = result.stdout + result.stderr
        l3_details.append("compare_talker_decode_full.py output captured")
        (DEBUG_BIN / f"l3_decode_output_{timestamp}.txt").write_text(output, encoding="utf-8")
        if "All compared steps MATCH" in output:
            l3_details.append("All talker decode steps MATCH")
        else:
            l3_details.append("Talker decode divergences found (see l3_decode_output_*.txt)")
            l3_pass = False
    else:
        l3_details.append("compare_talker_decode_full.py not found")
        l3_pass = False

    script2 = SCRIPTS_DIR / "compare_coupling_flow.py"
    if script2.exists():
        result = subprocess.run(
            [sys.executable, str(script2),
             "--cpp-dir", str(CPP_DUMP_DIR),
             "--py-dir", str(PY_DUMP_DIR),
             "--max-steps", "10",
             "--hidden-size", str(hidden_size)],
            capture_output=True, text=True, timeout=120,
        )
        output = result.stdout + result.stderr
        l3_details.append("compare_coupling_flow.py output captured")
        (DEBUG_BIN / f"l3_coupling_output_{timestamp}.txt").write_text(output, encoding="utf-8")
        if "All coupling flow steps MATCH" in output:
            l3_details.append("All coupling flow steps MATCH")
        else:
            l3_details.append("Coupling flow divergences found (see l3_coupling_output_*.txt)")
            l3_pass = False
    else:
        l3_details.append("compare_coupling_flow.py not found")
        l3_pass = False

    report_parts.append(("L3 Talker-CP Coupling", l3_pass, l3_details))

    # Generate report
    report_path = DEBUG_BIN / f"validation_report_{timestamp}.md"
    lines = [
        "# Qwen3-TTS Validation Report (infer.py vs C++)",
        "",
        f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Python Dump**: {PY_DUMP_DIR}",
        f"**C++ Dump**: {CPP_DUMP_DIR}",
        "",
    ]
    overall_pass = True
    for title, passed, details in report_parts:
        icon = "✅" if passed else "❌"
        lines.append(f"## {icon} {title} — {'PASS' if passed else 'FAIL'}")
        lines.append("")
        for d in details:
            lines.append(f"- {d}")
        lines.append("")
        if not passed:
            overall_pass = False

    lines.append("## Overall Result")
    lines.append("")
    lines.append(f"{'✅ ALL LEVELS PASSED' if overall_pass else '❌ SOME LEVELS FAILED'}")
    lines.append("")
    lines.append("## Raw Outputs")
    lines.append("")
    lines.append(f"- L1: `l1_prefill_output_{timestamp}.txt`")
    lines.append(f"- L2: `l2_cp_output_{timestamp}.txt`")
    lines.append(f"- L3: `l3_decode_output_{timestamp}.txt`, `l3_coupling_output_{timestamp}.txt`")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[REPORT] Saved to {report_path}")
    print(report_path.read_text(encoding="utf-8"))
    return overall_pass


def main():
    parser = argparse.ArgumentParser(description="Automated infer.py vs C++ validation")
    parser.add_argument("--skip-archive", action="store_true", help="Skip archiving old dumps")
    parser.add_argument("--skip-infer", action="store_true", help="Skip running infer.py")
    parser.add_argument("--skip-cpp", action="store_true", help="Skip running C++ on pyramid")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading cpp dumps")
    parser.add_argument("--force-upload", action="store_true", help="Force upload binary to pyramid")
    args = parser.parse_args()

    print("=" * 70)
    print("Qwen3-TTS Automated Validation: infer.py vs C++")
    print("=" * 70)

    if not args.skip_archive:
        archive_old_dumps()

    if not args.skip_infer:
        run_infer_py()

    if not args.skip_cpp:
        sync_to_pyramid()
        run_cpp_on_pyramid()

    if not args.skip_download:
        download_cpp_dumps()

    overall_pass = run_comparisons()
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
