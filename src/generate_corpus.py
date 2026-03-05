"""Generate the full test dataset corpus.

Loops through all ground-truth speeches (prompt files × 5 speeches)
and for each generates:
    - control
    - mild dysfluent   (severity 0)
    - moderate dysfluent (severity 1)
    - severe dysfluent   (severity 2)

Usage:
    python -m src.generate_corpus
    python -m src.generate_corpus --dry-run          # LLM only, skip synthesis
    python -m src.generate_corpus --start-gt 20      # resume from gt_idx 20
    python -m src.generate_corpus --speaker 42       # fixed speaker ID
    python -m src.generate_corpus --control-only     # extra control pass, new speakers
"""
import logging
logging.basicConfig(level=logging.WARNING)
import csv
import os
import sys
import argparse
import random

import torch
from google import genai
from dotenv import load_dotenv

from src.prompts import load_gt_seed, get_num_prompts, SPEECHES_PER_PROMPT
from src.dysfluency import generate_dysfluent_ipa
from src.control import generate_control_ipa
from src.synth_phonemes import (
    load_model, synthesize_sentences, save_wav,
    build_metadata_row, log_metadata,
)

load_dotenv()

SEVERITY_LABELS = ["mild", "moderate", "severe"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHONEME_DIR = os.path.join(BASE_DIR, "data", "phonemes")
GT_DIR = os.path.join(BASE_DIR, "data", "gt")
METADATA_CSV = os.path.join(BASE_DIR, "data", "metadata.csv")


def process_speech(client, model, hps, sr, device, ref_text, gt_idx,
                   prompt_idx, speaker_id, dry_run):
    """Generate control + all 3 severities for a single speech."""
    counts = {"control": 0, "mild": 0, "moderate": 0, "severe": 0}

    # ---- ground truth file ----
    gt_path = os.path.join(GT_DIR, f"gt_{gt_idx:03d}.txt")
    with open(gt_path, "w", encoding="utf-8") as f:
        f.write(ref_text.strip())

    # ---- control ----
    print(f"    [control] Generating...")
    ctrl_word_text, ctrl_ipa = generate_control_ipa(client, ref_text)

    ctrl_tag = f"control_gt{gt_idx}"
    with open(os.path.join(PHONEME_DIR, f"word_{ctrl_tag}.txt"), "w", encoding="utf-8") as f:
        f.write(ctrl_word_text)
    with open(os.path.join(PHONEME_DIR, f"phone_{ctrl_tag}.txt"), "w", encoding="utf-8") as f:
        f.write(ctrl_ipa + "\n")

    if not dry_run:
        print(f"    [control] Synthesizing...")
        ctrl_audio_dir = os.path.join(BASE_DIR, "data", "control")
        ctrl_results = synthesize_sentences(model, hps, ctrl_ipa, speaker_id, device)
        for r in ctrl_results:
            fname = f"ctrl_gt{gt_idx}_spk{speaker_id}_s{r['sentence_idx']:03d}.wav"
            wav_path = os.path.join(ctrl_audio_dir, fname)
            save_wav(r["audio"], wav_path, sr)
            row = build_metadata_row(
                file_path=os.path.relpath(wav_path, BASE_DIR),
                label="control",
                severity="none",
                speaker_id=speaker_id,
                result=r,
                prompt_idx=prompt_idx,
                gt_idx=gt_idx,
            )
            log_metadata(METADATA_CSV, row)
            print(f"      [s{r['sentence_idx']:03d}] {r['duration_sec']:.1f}s → {fname}")
        counts["control"] = len(ctrl_results)

    # ---- dysfluent: all 3 severities ----
    for severity in range(3):
        severity_label = SEVERITY_LABELS[severity]
        print(f"    [{severity_label}] Generating L1 + L2...")
        dys_word_text, dys_ipa = generate_dysfluent_ipa(client, ref_text, severity)

        tag = f"{severity_label}_gt{gt_idx}"
        with open(os.path.join(PHONEME_DIR, f"word_{tag}.txt"), "w", encoding="utf-8") as f:
            f.write(dys_word_text)
        with open(os.path.join(PHONEME_DIR, f"phone_{tag}.txt"), "w", encoding="utf-8") as f:
            f.write(dys_ipa + "\n")

        if not dry_run:
            print(f"    [{severity_label}] Synthesizing...")
            dys_audio_dir = os.path.join(BASE_DIR, "data", "dysfluent", severity_label)
            os.makedirs(dys_audio_dir, exist_ok=True)
            dys_results = synthesize_sentences(model, hps, dys_ipa, speaker_id, device)
            for r in dys_results:
                fname = f"dys_{severity_label}_gt{gt_idx}_spk{speaker_id}_s{r['sentence_idx']:03d}.wav"
                wav_path = os.path.join(dys_audio_dir, fname)
                save_wav(r["audio"], wav_path, sr)
                row = build_metadata_row(
                    file_path=os.path.relpath(wav_path, BASE_DIR),
                    label="dysfluent",
                    severity=severity_label,
                    speaker_id=speaker_id,
                    result=r,
                    prompt_idx=prompt_idx,
                    gt_idx=gt_idx,
                )
                log_metadata(METADATA_CSV, row)
                print(f"      [s{r['sentence_idx']:03d}] {r['duration_sec']:.1f}s → {fname}")
            counts[severity_label] = len(dys_results)

    return counts


def get_existing_speakers(metadata_csv):
    """Read metadata.csv and return {gt_idx: set of speaker_ids already used}."""
    speakers = {}
    if not os.path.isfile(metadata_csv):
        return speakers
    with open(metadata_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt = int(row["gt_idx"])
            spk = int(row["speaker_id"])
            speakers.setdefault(gt, set()).add(spk)
    return speakers


def pick_new_speaker(existing_speakers, gt_idx):
    """Pick a random VCTK speaker ID not already used for this gt_idx."""
    used = existing_speakers.get(gt_idx, set())
    available = [s for s in range(109) if s not in used]
    if not available:
        available = list(range(109))  # fallback if all used
    return random.choice(available)


def process_control_only(client, model, hps, sr, device, ref_text, gt_idx,
                         prompt_idx, speaker_id, dry_run):
    """Generate only the control layer for a single speech (no dysfluent)."""
    print(f"    [control] Generating...")
    ctrl_word_text, ctrl_ipa = generate_control_ipa(client, ref_text)

    ctrl_tag = f"control_gt{gt_idx}_v2"
    with open(os.path.join(PHONEME_DIR, f"word_{ctrl_tag}.txt"), "w", encoding="utf-8") as f:
        f.write(ctrl_word_text)
    with open(os.path.join(PHONEME_DIR, f"phone_{ctrl_tag}.txt"), "w", encoding="utf-8") as f:
        f.write(ctrl_ipa + "\n")

    count = 0
    if not dry_run:
        print(f"    [control] Synthesizing...")
        ctrl_audio_dir = os.path.join(BASE_DIR, "data", "control")
        ctrl_results = synthesize_sentences(model, hps, ctrl_ipa, speaker_id, device)
        for r in ctrl_results:
            fname = f"ctrl_gt{gt_idx}_spk{speaker_id}_s{r['sentence_idx']:03d}.wav"
            wav_path = os.path.join(ctrl_audio_dir, fname)
            save_wav(r["audio"], wav_path, sr)
            row = build_metadata_row(
                file_path=os.path.relpath(wav_path, BASE_DIR),
                label="control",
                severity="none",
                speaker_id=speaker_id,
                result=r,
                prompt_idx=prompt_idx,
                gt_idx=gt_idx,
            )
            log_metadata(METADATA_CSV, row)
            print(f"      [s{r['sentence_idx']:03d}] {r['duration_sec']:.1f}s → {fname}")
        count = len(ctrl_results)

    return count


def main():
    parser = argparse.ArgumentParser(description="Generate full test dataset corpus")
    parser.add_argument("--speaker", type=int, default=None,
                        help="VCTK speaker ID 0-108 (default: random per gt_idx)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run LLM generation only, skip synthesis")
    parser.add_argument("--start-gt", type=int, default=0,
                        help="Resume from this gt_idx (skip earlier ones)")
    parser.add_argument("--control-only", action="store_true",
                        help="Generate only control layer with new speakers (no dysfluent)")
    args = parser.parse_args()

    num_prompts = get_num_prompts()
    total_gt = num_prompts * SPEECHES_PER_PROMPT

    mode_label = "control-only (new speakers)" if args.control_only else "control + mild + moderate + severe"
    print(f"{'='*60}")
    print(f"  Corpus generation: {num_prompts} prompts × {SPEECHES_PER_PROMPT} speeches = {total_gt} GTs")
    print(f"  Mode: {mode_label}")
    print(f"  Starting from gt_idx={args.start_gt}")
    print(f"{'='*60}")

    # ---- directories ----
    os.makedirs(PHONEME_DIR, exist_ok=True)
    os.makedirs(GT_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data", "control"), exist_ok=True)
    if not args.control_only:
        for sev in SEVERITY_LABELS:
            os.makedirs(os.path.join(BASE_DIR, "data", "dysfluent", sev), exist_ok=True)

    # ---- LLM client ----
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    # ---- load existing speakers (for --control-only) ----
    existing_speakers = {}
    if args.control_only:
        existing_speakers = get_existing_speakers(METADATA_CSV)
        print(f"  Found existing speakers for {len(existing_speakers)} GTs")

    # ---- load VITS model ----
    model, hps, sr, device = None, None, None, None
    if not args.dry_run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nLoading VITS model (device={device})...")
        model, hps = load_model(device)
        sr = hps.data.sampling_rate

    # ---- main loop ----
    grand_total = {"control": 0, "mild": 0, "moderate": 0, "severe": 0}

    for prompt_idx in range(num_prompts):
        speeches = load_gt_seed(prompt_idx)

        for speech_offset, ref_text in enumerate(speeches):
            gt_idx = prompt_idx * SPEECHES_PER_PROMPT + speech_offset

            if gt_idx < args.start_gt:
                continue

            if args.control_only:
                speaker_id = args.speaker if args.speaker is not None else pick_new_speaker(existing_speakers, gt_idx)

                print(f"\n{'─'*60}")
                print(f"  GT {gt_idx}/{total_gt - 1}  (prompt={prompt_idx}, speech={speech_offset + 1})  speaker={speaker_id}")
                print(f"{'─'*60}")

                count = process_control_only(
                    client, model, hps, sr, device,
                    ref_text, gt_idx, prompt_idx, speaker_id, args.dry_run,
                )
                grand_total["control"] += count
            else:
                speaker_id = args.speaker if args.speaker is not None else random.randint(0, 108)

                print(f"\n{'─'*60}")
                print(f"  GT {gt_idx}/{total_gt - 1}  (prompt={prompt_idx}, speech={speech_offset + 1})  speaker={speaker_id}")
                print(f"{'─'*60}")

                counts = process_speech(
                    client, model, hps, sr, device,
                    ref_text, gt_idx, prompt_idx, speaker_id, args.dry_run,
                )

                for k, v in counts.items():
                    grand_total[k] += v

            done = gt_idx - args.start_gt + 1
            remaining = total_gt - gt_idx - 1
            print(f"\n  Progress: {done} done, {remaining} remaining")

    total_wavs = sum(grand_total.values())
    print(f"\n{'='*60}")
    print(f"  CORPUS COMPLETE")
    print(f"  Total WAVs: {total_wavs}")
    if args.control_only:
        print(f"    Control:  {grand_total['control']}")
    else:
        print(f"    Control:  {grand_total['control']}")
        print(f"    Mild:     {grand_total['mild']}")
        print(f"    Moderate: {grand_total['moderate']}")
        print(f"    Severe:   {grand_total['severe']}")
    print(f"  Metadata → {METADATA_CSV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
