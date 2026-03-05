"""Synthesize audio from a dysfluent IPA phoneme string using VITS (VCTK model).

Input format: space-separated IPA phones with inline dysfluency markers, e.g.:
    aɪ w oʊ k ʌ p ɜː [PRO] l ɪ ɚ | b ɪ k ʌ [DEL] z ...

Dysfluency markers handled:
    [PRO]  prolongation  — extends the preceding phoneme's duration by 0.17-0.8s
                           inside the VITS duration matrix
    [PAU]  block/pause   — splits synthesis and inserts 0.3-1.5s silence
    [DEL]  deletion      — stripped (phone was deleted, nothing to say)
    [INS]  insertion     — stripped (inserted phone already in stream)
    [REP]  repetition    — stripped (the preceding ... already signals repetition)
    ...                  — kept as … (pause / repetition ellipsis)
    |      sentence boundary — chunk split point; each chunk becomes a separate WAV

Usage (CLI, still works):
    python3 synth_phonemes.py phonemes.txt [output_dir] [--speaker N]

Programmatic usage:
    from synth_phonemes import load_model, synthesize_sentences
    model, hps = load_model(device)
    results = synthesize_sentences(model, hps, raw_ipa, speaker_id, device)
    # results: list of (audio_np_array, duration_sec, chunk_ipa, metadata_dict)
"""

import os
import sys
import re
import random
import argparse
import csv

# Add the vits/ directory to sys.path so that bare imports used by vits
# internals (e.g. `import commons`, `import modules`) resolve correctly.
_VITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vits")
sys.path.insert(0, _VITS_DIR)

import numpy as np
import torch
from scipy.io.wavfile import write as write_wav

# monotonic_align requires a Cython extension that may not be built.
# It is only used during training, not inference, so mock it if missing.
import types
try:
    import monotonic_align  # noqa: F401
except (ImportError, ModuleNotFoundError):
    _ma = types.ModuleType("monotonic_align")
    _ma.__path__ = []
    sys.modules["monotonic_align"] = _ma

from models import SynthesizerTrn
import commons
import utils

# Load symbols directly from text/symbols.py, bypassing text/__init__.py
# which pulls in cleaners → unidecode, phonemizer (not needed for inference).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("text.symbols", os.path.join(_VITS_DIR, "text", "symbols.py"))
_sym_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sym_mod)
symbols = _sym_mod.symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def text_to_sequence_phn(phonemes: str) -> list[int]:
    """Convert a cleaned IPA string to a sequence of symbol IDs."""
    return [_symbol_to_id[c] for c in phonemes if c in _symbol_to_id]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = os.path.join(_VITS_DIR, "pretrained_vctk.pth")
CONFIG_PATH     = os.path.join(_VITS_DIR, "configs", "vctk_base.json")

SILENCE_SECS        = 0.30           # silence between sentence-boundary chunks
PAU_SECS_RANGE      = (0.3, 1.5)    # silence range for [PAU] blocks
PRO_SECS_RANGE      = (0.17, 0.8)   # extra duration range for [PRO] prolongation
MAX_CHUNK_CHARS     = 500            # safety upper limit on IPA chars per inference call
MAX_DURATION_SEC    = 20.0           # if a synthesized chunk exceeds this, re-split and re-synth

PAUSE_CHAR = "…"             # U+2026, in VITS punctuation symbol set

_VALID = set(symbols)

# ---------------------------------------------------------------------------
# Marker / token normalisation
# ---------------------------------------------------------------------------

_STRIP_MARKERS = re.compile(r'\[(DEL|INS|REP|SUB)\]')
_ALL_MARKERS = re.compile(r'\[(PRO|PAU|DEL|INS|REP|SUB)\]')

def preprocess_ipa(raw: str) -> str:
    """Normalise an IPA string, keeping [PAU] and [PRO] markers for synthesis."""
    s = _STRIP_MARKERS.sub("", raw)
    s = s.replace("...", PAUSE_CHAR)
    s = re.sub(r' +', ' ', s).strip()
    return s


def filter_to_valid(s: str) -> str:
    """Drop any character not in the VITS symbol table."""
    return "".join(c for c in s if c in _VALID)


def count_markers(raw: str) -> dict:
    """Count dysfluency markers in raw IPA string."""
    markers = _ALL_MARKERS.findall(raw)
    counts = {"PRO": 0, "PAU": 0, "DEL": 0, "INS": 0, "REP": 0, "SUB": 0}
    for m in markers:
        counts[m] += 1
    return counts


def count_phonemes(ipa: str) -> int:
    """Count valid phoneme symbols in an IPA string (excluding markers)."""
    cleaned = _ALL_MARKERS.sub("", ipa)
    return sum(1 for c in cleaned if c in _symbol_to_id)

# ---------------------------------------------------------------------------
# Chunk splitting  (split on | sentence boundaries)
# ---------------------------------------------------------------------------

def split_into_chunks(ipa: str) -> list[str]:
    """Split the full IPA string at '|' sentence boundaries."""
    raw_chunks = ipa.split("|")
    chunks = []
    for raw in raw_chunks:
        chunk = raw.strip()
        if not chunk:
            continue
        chunks.append(chunk)
    return chunks

# ---------------------------------------------------------------------------
# VITS model
# ---------------------------------------------------------------------------

def load_model(device: torch.device):
    hps = utils.get_hparams_from_file(CONFIG_PATH)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    net_g.eval()
    utils.load_checkpoint(CHECKPOINT_PATH, net_g, None)
    return net_g, hps

# ---------------------------------------------------------------------------
# [PRO] marker parsing
# ---------------------------------------------------------------------------

def parse_pro_markers(chunk: str) -> tuple[str, list[int]]:
    """Extract [PRO] markers and return clean IPA with prolongation positions."""
    pro_indices = []
    clean = ""
    i = 0
    while i < len(chunk):
        if chunk[i:i+5] == "[PRO]":
            for j in range(len(clean) - 1, -1, -1):
                if clean[j] in _symbol_to_id:
                    pro_indices.append(j)
                    break
            i += 5
        else:
            clean += chunk[i]
            i += 1
    return clean, pro_indices


def text_to_sequence_with_pro(phonemes: str, pro_char_indices: list[int]) -> tuple[list[int], list[int]]:
    """Convert IPA to symbol IDs, mapping prolongation char positions to ID positions."""
    pro_set = set(pro_char_indices)
    ids = []
    pro_id_indices = []
    for i, c in enumerate(phonemes):
        if c in _symbol_to_id:
            if i in pro_set:
                pro_id_indices.append(len(ids))
            ids.append(_symbol_to_id[c])
    return ids, pro_id_indices

# ---------------------------------------------------------------------------
# Synthesis helpers
# ---------------------------------------------------------------------------

def _synthesize_ids(model, hps, ids: list[int], speaker_id: int,
                    device: torch.device, length_scale: float) -> np.ndarray:
    """Standard VITS inference from pre-computed symbol IDs (no prolongation)."""
    add_blank = getattr(hps.data, "add_blank", True)
    if add_blank:
        ids = commons.intersperse(ids, 0)

    x = torch.LongTensor(ids).unsqueeze(0).to(device)
    x_lengths = torch.LongTensor([len(ids)]).to(device)
    sid = torch.LongTensor([speaker_id]).to(device)

    with torch.no_grad():
        audio = model.infer(
            x, x_lengths, sid=sid,
            noise_scale=0.6, noise_scale_w=0.6, length_scale=length_scale,
        )[0][0, 0].cpu().float().numpy()
    return audio


def _synthesize_with_prolongation(model, hps, ids: list[int],
                                  pro_id_indices: list[int],
                                  speaker_id: int, device: torch.device,
                                  length_scale: float) -> np.ndarray:
    """VITS inference with duration extension at prolongation positions."""
    add_blank = getattr(hps.data, "add_blank", True)
    sr = hps.data.sampling_rate
    hop_length = hps.data.hop_length

    if add_blank:
        pro_interspersed = [2 * idx + 1 for idx in pro_id_indices]
        ids = commons.intersperse(ids, 0)
    else:
        pro_interspersed = list(pro_id_indices)

    x = torch.LongTensor(ids).unsqueeze(0).to(device)
    x_lengths = torch.LongTensor([len(ids)]).to(device)
    sid = torch.LongTensor([speaker_id]).to(device)

    with torch.no_grad():
        x_enc, m_p, logs_p, x_mask = model.enc_p(x, x_lengths)
        g = model.emb_g(sid).unsqueeze(-1) if model.n_speakers > 0 else None

        if model.use_sdp:
            logw = model.dp(x_enc, x_mask, g=g, reverse=True, noise_scale=0.6)
        else:
            logw = model.dp(x_enc, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)

        for idx in pro_interspersed:
            if idx < w_ceil.shape[2]:
                extra_secs = random.uniform(*PRO_SECS_RANGE)
                extra_frames = int(extra_secs * sr / hop_length)
                w_ceil[0, 0, idx] += extra_frames

        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(
            commons.sequence_mask(y_lengths, None), 1
        ).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * 0.6
        z = model.flow(z_p, y_mask, g=g, reverse=True)
        o = model.dec((z * y_mask), g=g)
        audio = o[0, 0].cpu().float().numpy()

    return audio

# ---------------------------------------------------------------------------
# Top-level chunk synthesis (handles [PAU] + [PRO])
# ---------------------------------------------------------------------------

def synthesize_chunk(model, hps, chunk: str, speaker_id: int,
                     device: torch.device) -> tuple[np.ndarray, float]:
    """Synthesize one sentence chunk, handling [PAU] and [PRO] markers.

    Returns:
        (audio_array, length_scale)
    """
    sr = hps.data.sampling_rate
    length_scale = random.uniform(1.2, 1.5)

    sub_chunks = re.split(r'\s*\[PAU\]\s*', chunk)

    audio_parts = []
    for sc in sub_chunks:
        sc = sc.strip()
        if not sc:
            continue

        clean_ipa, pro_char_indices = parse_pro_markers(sc)
        ids, pro_id_indices = text_to_sequence_with_pro(clean_ipa, pro_char_indices)
        if not ids:
            continue

        if pro_id_indices:
            audio = _synthesize_with_prolongation(
                model, hps, ids, pro_id_indices, speaker_id, device, length_scale)
        else:
            audio = _synthesize_ids(
                model, hps, ids, speaker_id, device, length_scale)

        if audio.size > 0:
            audio_parts.append(audio)

    if not audio_parts:
        return np.array([], dtype=np.float32), length_scale

    result = []
    for i, part in enumerate(audio_parts):
        result.append(part)
        if i < len(audio_parts) - 1:
            pause_secs = random.uniform(*PAU_SECS_RANGE)
            result.append(np.zeros(int(sr * pause_secs), dtype=np.float32))

    return np.concatenate(result), length_scale


# ---------------------------------------------------------------------------
# Per-sentence synthesis (main programmatic API)
# ---------------------------------------------------------------------------

def synthesize_sentences(model, hps, raw_ipa: str, speaker_id: int,
                         device: torch.device) -> list[dict]:
    """Synthesize each sentence (|-delimited) as a separate audio clip.

    Args:
        model: loaded VITS model
        hps: model hyperparameters
        raw_ipa: full IPA string with dysfluency markers and | boundaries
        speaker_id: VCTK speaker ID
        device: torch device

    Returns:
        List of dicts, one per sentence:
        {
            "audio": np.ndarray,
            "duration_sec": float,
            "chunk_ipa": str,          # raw IPA for this chunk (with markers)
            "length_scale": float,
            "marker_counts": dict,     # {PRO: n, PAU: n, DEL: n, INS: n, REP: n}
            "num_phonemes": int,
            "sentence_idx": int,
        }
    """
    sr = hps.data.sampling_rate
    ipa = preprocess_ipa(raw_ipa)
    chunks = split_into_chunks(ipa)

    results = []
    # We also want marker counts from the *raw* per-chunk IPA (before preprocessing).
    # Split raw on | to align with chunks.
    raw_chunks = raw_ipa.split("|")
    raw_chunks = [rc.strip() for rc in raw_chunks if rc.strip()]

    for i, chunk in enumerate(chunks):
        audio, length_scale = synthesize_chunk(model, hps, chunk, speaker_id, device)
        if audio.size == 0:
            continue

        # Use raw chunk for marker counting if available, else fall back to processed
        raw_chunk = raw_chunks[i] if i < len(raw_chunks) else chunk
        markers = count_markers(raw_chunk)
        n_phonemes = count_phonemes(raw_chunk)

        duration = len(audio) / sr
        results.append({
            "audio": audio,
            "duration_sec": round(duration, 3),
            "chunk_ipa": chunk,
            "length_scale": round(length_scale, 3),
            "marker_counts": markers,
            "num_phonemes": n_phonemes,
            "sentence_idx": i,
        })

    results = _split_long(results, model, hps, speaker_id, device)
    return _merge_short(results, sr)


MIN_DURATION_SEC = 5.0
MERGE_GAP_SEC = 0.15


def _bisect_ipa(ipa: str) -> tuple[str, str]:
    """Split an IPA string into two halves at the nearest comma or space to the midpoint."""
    mid = len(ipa) // 2
    # Prefer splitting at a comma near the midpoint
    best = -1
    best_dist = len(ipa)
    for i, c in enumerate(ipa):
        if c == ',' and abs(i - mid) < best_dist:
            best = i
            best_dist = abs(i - mid)
    if best != -1 and best_dist < len(ipa) // 4:
        return ipa[:best + 1].strip(), ipa[best + 1:].strip()
    # Fall back to nearest space
    best = -1
    best_dist = len(ipa)
    for i, c in enumerate(ipa):
        if c == ' ' and abs(i - mid) < best_dist:
            best = i
            best_dist = abs(i - mid)
    if best != -1:
        return ipa[:best].strip(), ipa[best + 1:].strip()
    # No split point found, return as-is
    return ipa, ""


def _split_long(results: list[dict], model, hps, speaker_id: int,
                device: torch.device) -> list[dict]:
    """Re-split and re-synthesize any chunk that exceeds MAX_DURATION_SEC."""
    sr = hps.data.sampling_rate
    out = []
    for r in results:
        if r["duration_sec"] <= MAX_DURATION_SEC:
            out.append(r)
            continue
        # Bisect the IPA and re-synthesize each half
        left_ipa, right_ipa = _bisect_ipa(r["chunk_ipa"])
        if not right_ipa:
            out.append(r)
            continue
        for half_ipa in (left_ipa, right_ipa):
            audio, length_scale = synthesize_chunk(model, hps, half_ipa, speaker_id, device)
            if audio.size == 0:
                continue
            raw_chunk = half_ipa
            markers = count_markers(raw_chunk)
            n_phonemes = count_phonemes(raw_chunk)
            duration = len(audio) / sr
            out.append({
                "audio": audio,
                "duration_sec": round(duration, 3),
                "chunk_ipa": half_ipa,
                "length_scale": round(length_scale, 3),
                "marker_counts": markers,
                "num_phonemes": n_phonemes,
                "sentence_idx": 0,
            })
    # Recurse if any half is still too long
    if any(r["duration_sec"] > MAX_DURATION_SEC for r in out):
        out = _split_long(out, model, hps, speaker_id, device)
    # Re-index
    for i, r in enumerate(out):
        r["sentence_idx"] = i
    return out


def _merge_two(a: dict, b: dict, sr: int) -> dict:
    """Merge two result dicts by concatenating audio with a small silence gap."""
    gap = np.zeros(int(sr * MERGE_GAP_SEC), dtype=np.float32)
    merged_audio = np.concatenate([a["audio"], gap, b["audio"]])
    mc = {k: a["marker_counts"][k] + b["marker_counts"][k] for k in a["marker_counts"]}
    return {
        "audio": merged_audio,
        "duration_sec": round(len(merged_audio) / sr, 3),
        "chunk_ipa": a["chunk_ipa"] + " | " + b["chunk_ipa"],
        "length_scale": round((a["length_scale"] + b["length_scale"]) / 2, 3),
        "marker_counts": mc,
        "num_phonemes": a["num_phonemes"] + b["num_phonemes"],
        "sentence_idx": a["sentence_idx"],
    }


def _merge_short(results: list[dict], sr: int) -> list[dict]:
    """Merge any result shorter than MIN_DURATION_SEC into an adjacent result."""
    if not results:
        return results

    merged = list(results)
    changed = True
    while changed:
        changed = False
        for i, r in enumerate(merged):
            if r["duration_sec"] < MIN_DURATION_SEC:
                if i + 1 < len(merged):
                    merged[i] = _merge_two(r, merged[i + 1], sr)
                    del merged[i + 1]
                elif i > 0:
                    merged[i - 1] = _merge_two(merged[i - 1], r, sr)
                    del merged[i]
                else:
                    continue
                changed = True
                break

    for i, r in enumerate(merged):
        r["sentence_idx"] = i

    return merged


def save_wav(audio: np.ndarray, path: str, sr: int):
    """Save audio array as 16-bit WAV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_wav(path, sr, (audio * 32767).astype(np.int16))


# ---------------------------------------------------------------------------
# Metadata logging
# ---------------------------------------------------------------------------

METADATA_FIELDS = [
    "file_path", "label", "severity", "speaker_id", "duration_sec",
    "sentence_idx", "prompt_idx", "num_phonemes", "num_markers_total",
    "markers_PRO", "markers_PAU", "markers_DEL", "markers_INS", "markers_REP",
    "markers_SUB", "dysfluency_rate", "length_scale", "gt_idx",
    "chunk_ipa",
]

def log_metadata(csv_path: str, row: dict):
    """Append a row to the metadata CSV. Creates file + header if needed."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_metadata_row(file_path: str, label: str, severity: str,
                       speaker_id: int, result: dict, prompt_idx: int,
                       gt_idx: int = 0) -> dict:
    """Build a metadata dict from a synthesize_sentences result entry."""
    mc = result["marker_counts"]
    total_markers = sum(mc.values())
    n_ph = result["num_phonemes"]
    return {
        "file_path": file_path,
        "label": label,
        "severity": severity,
        "speaker_id": speaker_id,
        "duration_sec": result["duration_sec"],
        "sentence_idx": result["sentence_idx"],
        "prompt_idx": prompt_idx,
        "num_phonemes": n_ph,
        "num_markers_total": total_markers,
        "markers_PRO": mc["PRO"],
        "markers_PAU": mc["PAU"],
        "markers_DEL": mc["DEL"],
        "markers_INS": mc["INS"],
        "markers_REP": mc["REP"],
        "markers_SUB": mc["SUB"],
        "dysfluency_rate": round(total_markers / max(n_ph, 1), 4),
        "length_scale": result["length_scale"],
        "gt_idx": gt_idx,
        "chunk_ipa": result["chunk_ipa"],
    }


# ---------------------------------------------------------------------------
# CLI (standalone usage still works)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Synthesize dysfluent IPA with VITS (VCTK)")
    parser.add_argument("input", help="Path to phonemes.txt, or '-' for stdin")
    parser.add_argument("output", nargs="?", default="output",
                        help="Output directory for per-sentence WAVs (default: output/)")
    parser.add_argument("--speaker", type=int, default=None,
                        help="VCTK speaker ID 0–108 (default: random)")
    parser.add_argument("--label", default="dysfluent", help="Label for metadata")
    parser.add_argument("--severity", default="unknown", help="Severity for metadata")
    parser.add_argument("--prompt-idx", type=int, default=0, help="Prompt index for metadata")
    parser.add_argument("--metadata-csv", default=None,
                        help="Path to metadata CSV (default: <output>/metadata.csv)")
    args = parser.parse_args()

    if not os.path.isfile(CHECKPOINT_PATH):
        print(f"ERROR: VCTK checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)

    if args.input == "-":
        raw = sys.stdin.read()
    else:
        if not os.path.isfile(args.input):
            print(f"ERROR: file not found: {args.input}")
            sys.exit(1)
        raw = open(args.input, encoding="utf-8").read()

    if not raw.strip():
        print("ERROR: no phoneme input found")
        sys.exit(1)

    speaker_id = args.speaker if args.speaker is not None else random.randint(0, 108)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Speaker ID: {speaker_id} | Device: {device}")

    print("Loading VITS (VCTK) model...")
    model, hps = load_model(device)
    sr = hps.data.sampling_rate

    os.makedirs(args.output, exist_ok=True)
    csv_path = args.metadata_csv or os.path.join(args.output, "metadata.csv")

    results = synthesize_sentences(model, hps, raw, speaker_id, device)
    print(f"Synthesized {len(results)} sentences")

    for r in results:
        fname = f"{args.label}_{args.severity}_p{args.prompt_idx}_s{r['sentence_idx']:03d}.wav"
        wav_path = os.path.join(args.output, fname)
        save_wav(r["audio"], wav_path, sr)

        row = build_metadata_row(
            file_path=wav_path,
            label=args.label,
            severity=args.severity,
            speaker_id=speaker_id,
            result=r,
            prompt_idx=args.prompt_idx,
        )
        log_metadata(csv_path, row)
        print(f"  [{r['sentence_idx']:03d}] {r['duration_sec']:.1f}s -> {fname}")

    print(f"\nDone. {len(results)} WAVs in {args.output}, metadata in {csv_path}")


if __name__ == "__main__":
    main()