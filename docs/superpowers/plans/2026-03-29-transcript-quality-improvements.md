# Transcript Quality Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate Whisper hallucinations from silence/hold music, make batch transcription fully unattended with deferred speaker naming, and merge filler micro-blocks to reduce visual clutter.

**Architecture:** Three independent improvements to `transcribe.py`: (1) Silero VAD preprocessing + post-processing hallucination filter, (2) new CLI flags `--skip-speakers`/`--auto`/`--name-speakers` to decouple transcription from speaker identification, (3) filler block merging. All tested via `smoke_test.py` using the existing test runner pattern.

**Tech Stack:** Python 3.13, silero-vad, torchaudio, mlx-whisper, pyannote-audio

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `transcribe.py` | Modify | Add `detect_speech_segments`, `prepare_vad_audio`, `remap_timestamps`, `filter_hallucinations`, `merge_filler_blocks`, `name_speakers_in_files`; modify `transcribe`, `process_file`, `parse_args`, `main` |
| `smoke_test.py` | Modify | Add 10 new tests for all three improvements |
| `pyproject.toml` | Modify | Add `silero-vad` dependency |
| `setup.sh` | No change | `silero-vad` installs via `uv sync` automatically |

---

## Task 1: Add `silero-vad` Dependency

**Files:**
- Modify: `pyproject.toml:5-12`

- [ ] **Step 1: Add silero-vad to pyproject.toml**

In `pyproject.toml`, add `silero-vad` to the dependencies list:

```toml
dependencies = [
    "mlx-whisper>=0.4.0",
    "google-genai>=1.0.0",
    "openai>=1.0.0",
    "pyannote-audio>=3.3.0",
    "silero-vad>=5.1",
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
]
```

- [ ] **Step 2: Install the new dependency**

Run: `uv sync`
Expected: Installs `silero-vad` successfully.

- [ ] **Step 3: Verify import works**

Run: `uv run python -c "from silero_vad import load_silero_vad; print('silero-vad OK')"`
Expected: Prints `silero-vad OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add silero-vad for voice activity detection"
```

---

## Task 2: Post-Processing Hallucination Filter

The post-processing filter is independent of VAD and simpler, so build and test it first. This catches hallucinations even without VAD.

**Files:**
- Modify: `transcribe.py` (add `filter_hallucinations` after `build_blocks`)
- Modify: `smoke_test.py` (add 2 tests)

- [ ] **Step 1: Write the failing tests in `smoke_test.py`**

Add these two test functions before the `main()` function in `smoke_test.py`, and add `filter_hallucinations` to the import block at the top:

```python
# Add to imports at top:
from transcribe import (
    # ... existing imports ...
    filter_hallucinations,
)


def test_filter_hallucinations():
    """Test that hallucination patterns are removed."""
    blocks = [
        {"speaker": "Speaker 1", "start": 0.0, "text": "Thank you."},
        {"speaker": "Speaker 1", "start": 30.0, "text": "Thank you."},
        {"speaker": "Speaker 2", "start": 60.0, "text": "Thank you."},
        {"speaker": "Speaker 1", "start": 90.0, "text": "Thank you."},
        {"speaker": "Speaker 1", "start": 120.0, "text": "Bye bye bye bye bye bye bye bye bye bye"},
        {"speaker": "Speaker 1", "start": 600.0, "text": "So I started my career at Lyft."},
        {"speaker": "Speaker 2", "start": 610.0, "text": "Tell me more about that."},
    ]
    result = filter_hallucinations(blocks)
    assert len(result) == 2, f"Expected 2 blocks, got {len(result)}: {[b['text'] for b in result]}"
    assert result[0]["text"] == "So I started my career at Lyft."
    assert result[1]["text"] == "Tell me more about that."
    print("  filter_hallucinations: OK")


def test_filter_hallucinations_preserves_real():
    """Test that real dialogue blocks are not removed."""
    blocks = [
        {"speaker": "Speaker 1", "start": 0.0, "text": "Thank you for joining us today."},
        {"speaker": "Speaker 2", "start": 5.0, "text": "Thanks for having me, I appreciate the opportunity."},
        {"speaker": "Speaker 1", "start": 10.0, "text": "Yeah."},
        {"speaker": "Speaker 2", "start": 15.0, "text": "Let me tell you about my experience."},
    ]
    result = filter_hallucinations(blocks)
    assert len(result) == 4, f"Expected 4 blocks, got {len(result)}"
    print("  filter_hallucinations (preserves real): OK")
```

Also add both tests to the `tests` list in `main()`:

```python
    tests = [
        # ... existing tests ...
        test_filter_hallucinations,
        test_filter_hallucinations_preserves_real,
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python smoke_test.py`
Expected: FAIL — `ImportError` because `filter_hallucinations` doesn't exist yet.

- [ ] **Step 3: Implement `filter_hallucinations` in `transcribe.py`**

Add this function after `build_blocks()` (around line 224):

```python
import re


KNOWN_HALLUCINATIONS = {
    "thank you.", "thanks for watching.", "please subscribe.",
    "bye.", "thank you for watching.", "thanks for watching",
    "thank you", "bye",
}


def filter_hallucinations(blocks: list[dict]) -> list[dict]:
    """Remove blocks that are Whisper hallucinations from silence.

    Detects: known hallucination phrases, consecutive repetition (3+),
    and internal repetition (same short phrase repeated 5+ times).
    """
    if not blocks:
        return blocks

    # Mark blocks for removal
    remove = [False] * len(blocks)

    for i, block in enumerate(blocks):
        text = block["text"].strip()
        text_lower = text.lower().rstrip(".")

        # Heuristic 1: Known hallucination phrases (standalone)
        if text.lower() in KNOWN_HALLUCINATIONS:
            remove[i] = True
            continue

        # Heuristic 2: Internal repetition — same short phrase repeated 5+ times
        words = text.split()
        if len(words) >= 5:
            # Check for a repeating unit of 1-3 words
            for unit_len in range(1, 4):
                if len(words) < unit_len * 5:
                    continue
                unit = " ".join(words[:unit_len]).lower().strip(".,!?")
                if not unit:
                    continue
                repetitions = 0
                for j in range(0, len(words), unit_len):
                    chunk = " ".join(words[j:j + unit_len]).lower().strip(".,!?")
                    if chunk == unit:
                        repetitions += 1
                    else:
                        break
                if repetitions >= 5:
                    remove[i] = True
                    break

    # Heuristic 3: Consecutive identical blocks (run of 3+)
    i = 0
    while i < len(blocks):
        run_start = i
        text_i = blocks[i]["text"].strip().lower()
        while i + 1 < len(blocks) and blocks[i + 1]["text"].strip().lower() == text_i:
            i += 1
        run_length = i - run_start + 1
        if run_length >= 3:
            for j in range(run_start, run_start + run_length):
                remove[j] = True
        i += 1

    result = [b for b, r in zip(blocks, remove) if not r]
    removed_count = len(blocks) - len(result)
    if removed_count:
        print(f"  Filtered {removed_count} hallucinated blocks")
    return result
```

Also add `import re` to the top of the file if not already present. (Check first — it's not currently imported, but we use it above only in the `KNOWN_HALLUCINATIONS` set. Actually `re` isn't needed for the above — remove that import. The logic uses string operations only.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python smoke_test.py`
Expected: ALL tests pass, including the two new ones.

- [ ] **Step 5: Commit**

```bash
git add transcribe.py smoke_test.py
git commit -m "feat: add post-processing hallucination filter"
```

---

## Task 3: VAD Preprocessing — `detect_speech_segments` and `prepare_vad_audio`

**Files:**
- Modify: `transcribe.py` (add `detect_speech_segments`, `prepare_vad_audio`, `remap_timestamps`)
- Modify: `smoke_test.py` (add 1 test)

- [ ] **Step 1: Write the failing test in `smoke_test.py`**

Add `detect_speech_segments` to the import block, then add the test before `main()`:

```python
# Add to imports:
from transcribe import (
    # ... existing imports ...
    detect_speech_segments,
)


def test_detect_speech_segments_mocked():
    """Test detect_speech_segments with mocked Silero VAD."""
    from unittest.mock import MagicMock, patch
    import torch

    # Mock a 16kHz mono audio tensor (10 seconds = 160000 samples)
    fake_wav = torch.zeros(160000)

    # Mock VAD returning two speech segments
    fake_timestamps = [
        {"start": 16000, "end": 80000},   # 1.0s - 5.0s
        {"start": 96000, "end": 144000},   # 6.0s - 9.0s
    ]

    with patch("transcribe.load_silero_vad") as mock_load, \
         patch("transcribe.read_audio", return_value=fake_wav) as mock_read, \
         patch("transcribe.get_speech_timestamps", return_value=fake_timestamps):
        mock_load.return_value = MagicMock()
        result = detect_speech_segments("fake_audio.wav")

    assert len(result) == 2
    assert abs(result[0]["start"] - 1.0) < 0.01
    assert abs(result[0]["end"] - 5.0) < 0.01
    assert abs(result[1]["start"] - 6.0) < 0.01
    assert abs(result[1]["end"] - 9.0) < 0.01
    print("  detect_speech_segments (mocked): OK")
```

Add to the `tests` list in `main()`:

```python
        test_detect_speech_segments_mocked,
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python smoke_test.py`
Expected: FAIL — `ImportError` because `detect_speech_segments` doesn't exist yet.

- [ ] **Step 3: Implement VAD functions in `transcribe.py`**

Add these functions after the `transcribe()` function (around line 77). The imports for silero_vad are at module level because they're lightweight and needed for mocking:

```python
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


SAMPLE_RATE = 16000  # Silero VAD requires 16kHz


def detect_speech_segments(audio_path: str) -> list[dict]:
    """Use Silero VAD to find speech regions in the audio.

    Returns list of {"start": float, "end": float} in seconds.
    """
    model = load_silero_vad()
    wav = read_audio(audio_path)
    timestamps = get_speech_timestamps(wav, model)

    # Convert sample indices to seconds
    segments = []
    for ts in timestamps:
        segments.append({
            "start": ts["start"] / SAMPLE_RATE,
            "end": ts["end"] / SAMPLE_RATE,
        })

    print(f"  VAD found {len(segments)} speech segments")
    return segments


def prepare_vad_audio(audio_path: str, speech_segments: list[dict]) -> tuple[str, list]:
    """Concatenate speech-only portions into a temp WAV file.

    Returns (temp_file_path, offset_map) where offset_map is a list of
    (trimmed_start, original_start, duration) tuples for timestamp remapping.
    """
    import torchaudio
    import tempfile

    waveform, sr = torchaudio.load(audio_path)

    # Mix to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    chunks = []
    offset_map = []
    trimmed_pos = 0.0

    for seg in speech_segments:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        chunk = waveform[:, start_sample:end_sample]
        chunks.append(chunk)

        duration = seg["end"] - seg["start"]
        offset_map.append((trimmed_pos, seg["start"], duration))
        trimmed_pos += duration

    import torch
    if chunks:
        combined = torch.cat(chunks, dim=1)
    else:
        combined = waveform  # No speech found, use original

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(tmp.name, combined, sr)
    tmp.close()

    return tmp.name, offset_map


def remap_timestamps(words: list[dict], offset_map: list) -> list[dict]:
    """Remap word timestamps from trimmed audio back to original timeline.

    offset_map: list of (trimmed_start, original_start, duration)
    """
    for word in words:
        for trimmed_start, original_start, duration in offset_map:
            trimmed_end = trimmed_start + duration
            if trimmed_start <= word["start"] < trimmed_end:
                offset = original_start - trimmed_start
                word["start"] += offset
                word["end"] += offset
                break
    return words
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python smoke_test.py`
Expected: ALL tests pass.

- [ ] **Step 5: Commit**

```bash
git add transcribe.py smoke_test.py
git commit -m "feat: add Silero VAD speech segment detection"
```

---

## Task 4: Filler/Micro-Block Cleanup

**Files:**
- Modify: `transcribe.py` (add `merge_filler_blocks`)
- Modify: `smoke_test.py` (add 3 tests)

- [ ] **Step 1: Write the failing tests in `smoke_test.py`**

Add `merge_filler_blocks` to the import block, then add tests before `main()`:

```python
# Add to imports:
from transcribe import (
    # ... existing imports ...
    merge_filler_blocks,
)


def test_merge_filler_blocks():
    """Test that filler blocks are merged into the previous block."""
    blocks = [
        {"speaker": "Speaker 1", "start": 0.0, "text": "So I started at Lyft."},
        {"speaker": "Speaker 2", "start": 5.0, "text": "Yeah."},
        {"speaker": "Speaker 1", "start": 6.0, "text": "And then I moved to Kite."},
    ]
    result = merge_filler_blocks(blocks)
    assert len(result) == 2, f"Expected 2 blocks, got {len(result)}"
    assert "Yeah." in result[0]["text"]
    assert result[0]["text"].startswith("So I started at Lyft.")
    assert result[1]["text"] == "And then I moved to Kite."
    print("  merge_filler_blocks: OK")


def test_merge_filler_blocks_first():
    """Test that filler as first block merges into the next block."""
    blocks = [
        {"speaker": "Speaker 1", "start": 0.0, "text": "Um."},
        {"speaker": "Speaker 1", "start": 1.0, "text": "So I started at Lyft."},
    ]
    result = merge_filler_blocks(blocks)
    assert len(result) == 1, f"Expected 1 block, got {len(result)}"
    assert "Um." in result[0]["text"]
    assert "Lyft" in result[0]["text"]
    print("  merge_filler_blocks (first block): OK")


def test_merge_filler_preserves_substantive():
    """Test that short non-filler blocks are kept."""
    blocks = [
        {"speaker": "Speaker 1", "start": 0.0, "text": "What do you think?"},
        {"speaker": "Speaker 2", "start": 5.0, "text": "That's correct."},
        {"speaker": "Speaker 1", "start": 10.0, "text": "Great, let's continue."},
    ]
    result = merge_filler_blocks(blocks)
    assert len(result) == 3, f"Expected 3 blocks, got {len(result)}"
    print("  merge_filler_preserves_substantive: OK")
```

Add all three to the `tests` list in `main()`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python smoke_test.py`
Expected: FAIL — `ImportError` because `merge_filler_blocks` doesn't exist yet.

- [ ] **Step 3: Implement `merge_filler_blocks` in `transcribe.py`**

Add after `filter_hallucinations()`:

```python
FILLER_WORDS = {
    "yeah", "hmm", "um", "okay", "uh", "mm-hmm", "right", "sure",
    "so", "yep", "mhm", "uh-huh", "oh", "ah", "mmm", "hm",
}


def merge_filler_blocks(blocks: list[dict]) -> list[dict]:
    """Merge short filler-only blocks into adjacent blocks.

    A block is filler if it has <= 5 words and all words (stripped of
    punctuation) are in the FILLER_WORDS set.
    """
    if not blocks:
        return blocks

    def is_filler(block: dict) -> bool:
        words = block["text"].strip().split()
        if len(words) > 5:
            return False
        stripped = [w.strip(".,!?:;'\"").lower() for w in words]
        return all(w in FILLER_WORDS for w in stripped if w)

    # Tag each block
    filler_flags = [is_filler(b) for b in blocks]

    result = []
    for i, block in enumerate(blocks):
        if filler_flags[i]:
            if result:
                # Merge into previous block
                result[-1]["text"] = result[-1]["text"].rstrip() + " " + block["text"].strip()
            else:
                # First block is filler — find next non-filler and prepend
                for j in range(i + 1, len(blocks)):
                    if not filler_flags[j]:
                        blocks[j]["text"] = block["text"].strip() + " " + blocks[j]["text"]
                        break
        else:
            result.append(block)

    merged_count = len(blocks) - len(result)
    if merged_count:
        print(f"  Merged {merged_count} filler blocks")
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python smoke_test.py`
Expected: ALL tests pass.

- [ ] **Step 5: Commit**

```bash
git add transcribe.py smoke_test.py
git commit -m "feat: merge filler micro-blocks into adjacent blocks"
```

---

## Task 5: Integrate VAD, Hallucination Filter, and Filler Merge into Pipeline

Wire VAD into `transcribe()` and add all three new steps to `process_file()`.

**Files:**
- Modify: `transcribe.py:44-76` (`transcribe` function) and `transcribe.py:626-690` (`process_file` function)

- [ ] **Step 1: Modify `transcribe()` to accept speech segments**

Update the `transcribe` function signature and body. The function should accept an optional `speech_segments` parameter. When provided, it creates trimmed audio via `prepare_vad_audio`, transcribes that, then remaps timestamps back:

```python
def transcribe(audio_path: str, speech_segments: list[dict] | None = None) -> list[dict]:
    """Run mlx-whisper on the audio file, return list of words with timestamps."""
    import mlx_whisper

    # If VAD segments provided, transcribe only speech portions
    trimmed_path = None
    offset_map = None
    if speech_segments:
        trimmed_path, offset_map = prepare_vad_audio(audio_path, speech_segments)
        audio_to_transcribe = trimmed_path
    else:
        audio_to_transcribe = audio_path

    print("Transcribing audio with mlx-whisper (Apple Silicon GPU)...")
    result = mlx_whisper.transcribe(
        audio_to_transcribe,
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        word_timestamps=True,
    )

    words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            words.append({
                "word": w["word"],
                "start": w["start"],
                "end": w["end"],
            })

    # Calculate duration from last word or last segment
    duration = 0.0
    if words:
        duration = words[-1]["end"]
    elif result.get("segments"):
        duration = result["segments"][-1]["end"]

    # Remap timestamps if we used trimmed audio
    if offset_map:
        words = remap_timestamps(words, offset_map)
        # Duration should reflect original audio length, not trimmed
        if speech_segments:
            duration = max(seg["end"] for seg in speech_segments)

    # Clean up temp file
    if trimmed_path:
        Path(trimmed_path).unlink(missing_ok=True)

    info = TranscriptionInfo(
        language=result.get("language", "unknown"),
        duration=duration,
    )

    return words, info
```

- [ ] **Step 2: Modify `process_file()` to run VAD first**

Update `process_file` to detect speech segments before the parallel transcribe+diarize step. Add `filter_hallucinations` (Task 2) and `merge_filler_blocks` (Task 4) calls after `build_blocks`:

```python
def process_file(audio_path: str, output_dir: Path, diarization_pipeline=None,
                 skip_speakers: bool = False, auto_speakers: bool = False) -> Path:
    """Run the full transcription pipeline on a single audio file.

    Returns the path to the written markdown transcript.
    """
    audio_file = Path(audio_path)
    filename = audio_file.name

    print(f"Processing: {filename}")
    print("=" * 50)

    # Step 0: VAD — detect speech segments
    speech_segments = detect_speech_segments(audio_path)

    # Steps 1 & 2: Transcribe and diarize in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        transcribe_future = executor.submit(transcribe, audio_path, speech_segments)
        diarize_future = executor.submit(diarize, audio_path, diarization_pipeline)

    words, info = transcribe_future.result()
    print(f"  Transcribed {len(words)} words, language: {info.language}")

    speaker_segments = diarize_future.result()
    unique_speakers = set(s["speaker"] for s in speaker_segments)
    print(f"  Found {len(unique_speakers)} speakers in {len(speaker_segments)} segments")

    # Step 3: Assign speakers to words
    words = assign_speakers(words, speaker_segments)

    # Step 4: Build dialogue blocks
    blocks = build_blocks(words)
    print(f"  Built {len(blocks)} dialogue blocks")

    # Step 5: Filter hallucinations
    blocks = filter_hallucinations(blocks)

    # Step 6: Merge filler blocks
    blocks = merge_filler_blocks(blocks)

    # Step 7: Polish transcript via LLM
    blocks = polish_transcript(blocks)

    # Step 8: Identify speakers (conditional)
    if skip_speakers:
        speaker_name_map = {}
    elif auto_speakers:
        print("\nIdentifying speakers (auto mode)...")
        suggestions = infer_speaker_names(blocks)
        speaker_name_map = {}
        for speaker, info_dict in suggestions.items():
            name = info_dict.get("likely_name")
            if name:
                speaker_name_map[speaker] = name
        if speaker_name_map:
            print(f"  Auto-assigned {len(speaker_name_map)} speaker names")
    else:
        speaker_name_map = identify_speakers(blocks)

    for block in blocks:
        block["speaker"] = speaker_name_map.get(block["speaker"], block["speaker"])

    # Collect final speaker names for metadata
    final_speakers = sorted(set(b["speaker"] for b in blocks))

    # Step 9: Generate summary
    summary = generate_summary(blocks)
    if summary:
        print("  Generated interview summary")

    # Step 10: Format and save
    markdown = format_markdown(
        filename=filename,
        blocks=blocks,
        duration=info.duration,
        language=info.language,
        num_speakers=len(unique_speakers),
        speakers=final_speakers,
        summary=summary,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_file.stem
    output_path = output_dir / f"{stem}.md"
    output_path.write_text(markdown)

    print("=" * 50)
    print(f"Saved transcript to {output_path}")
    return output_path
```

- [ ] **Step 3: Run existing tests to verify nothing broke**

Run: `uv run python smoke_test.py`
Expected: ALL tests pass. (The `process_file` function is only tested for importability, and the new parameters are optional with backward-compatible defaults.)

- [ ] **Step 4: Commit**

```bash
git add transcribe.py
git commit -m "feat: integrate VAD preprocessing into transcription pipeline"
```

---

## Task 6: CLI Flags — `--skip-speakers`, `--auto`, `--name-speakers`

**Files:**
- Modify: `transcribe.py:704-726` (`parse_args`) and `transcribe.py:729-801` (`main`)
- Modify: `smoke_test.py` (add 3 tests)

- [ ] **Step 1: Write the failing tests in `smoke_test.py`**

Add tests before `main()`:

```python
def test_parse_args_skip_speakers():
    """Test that --skip-speakers flag is parsed."""
    args = parse_args(["--skip-speakers"])
    assert args.skip_speakers is True
    print("  parse_args (--skip-speakers): OK")


def test_parse_args_auto():
    """Test that --auto flag is parsed."""
    args = parse_args(["--auto"])
    assert args.auto is True
    print("  parse_args (--auto): OK")


def test_parse_args_name_speakers():
    """Test that --name-speakers flag is parsed and is mutually exclusive."""
    args = parse_args(["--name-speakers"])
    assert args.name_speakers is True
    print("  parse_args (--name-speakers): OK")
```

Add all three to the `tests` list in `main()`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python smoke_test.py`
Expected: FAIL — `args.skip_speakers` attribute doesn't exist.

- [ ] **Step 3: Update `parse_args` in `transcribe.py`**

Replace the `parse_args` function:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files with speaker diarization to markdown."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Audio file(s) to transcribe. If omitted, processes all audio files in --input-dir.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing audio files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output transcripts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-speakers",
        action="store_true",
        help="Skip speaker identification (default in batch mode).",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-accept LLM speaker name suggestions without prompting.",
    )
    parser.add_argument(
        "--name-speakers",
        action="store_true",
        help="Post-hoc speaker naming mode: rename generic speakers in existing transcripts.",
    )
    args = parser.parse_args(argv)

    # Mutual exclusivity: --name-speakers is standalone
    if args.name_speakers and (args.files or args.skip_speakers or args.auto):
        parser.error("--name-speakers cannot be combined with file args, --skip-speakers, or --auto.")

    return args
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python smoke_test.py`
Expected: ALL tests pass.

- [ ] **Step 5: Commit**

```bash
git add transcribe.py smoke_test.py
git commit -m "feat: add --skip-speakers, --auto, --name-speakers CLI flags"
```

---

## Task 7: Wire Speaker Flags into `main()`

Pass the speaker-related flags through `main()` to `process_file()`. Make `--skip-speakers` the default in batch mode.

**Files:**
- Modify: `transcribe.py:729-801` (`main` function)

- [ ] **Step 1: Update `main()` to pass speaker flags**

Replace the `main` function:

```python
def main():
    args = parse_args()

    if args.name_speakers:
        # Standalone mode: rename speakers in existing transcripts
        name_speakers_in_files(args.output_dir)
        return

    if args.files:
        # Single-file mode: process specific files
        for file_path in args.files:
            audio_file = Path(file_path)
            if not audio_file.exists():
                # Try looking in input dir
                audio_file = args.input_dir / file_path
            if not audio_file.exists():
                print(f"Error: Audio file not found: {file_path}")
                sys.exit(1)
            process_file(
                str(audio_file), args.output_dir,
                skip_speakers=args.skip_speakers,
                auto_speakers=args.auto,
            )
    else:
        # Batch mode: process all audio files in input dir
        # Default to skip-speakers in batch mode
        skip_speakers = True if not args.auto else False

        if not args.input_dir.exists():
            print(f"Error: Input directory not found: {args.input_dir}")
            sys.exit(1)

        all_files = discover_audio_files(args.input_dir)
        if not all_files:
            print(f"No audio files found in {args.input_dir}")
            sys.exit(0)

        # Skip files that already have transcripts
        to_process = [
            f for f in all_files
            if not (args.output_dir / f"{f.stem}.md").exists()
        ]

        skipped = len(all_files) - len(to_process)
        print(f"Found {len(all_files)} audio files, {skipped} already transcribed, {len(to_process)} to process.")
        print()

        if not to_process:
            print("All files already transcribed. Nothing to do.")
            sys.exit(0)

        # Load models once for batch processing
        print("Loading models...")
        diarization_pipeline = load_diarization_pipeline()
        print()

        import time

        failed = []
        elapsed_times = []
        for i, audio_file in enumerate(to_process, 1):
            # ETA based on average processing time
            if elapsed_times:
                avg_time = sum(elapsed_times) / len(elapsed_times)
                remaining = avg_time * (len(to_process) - i + 1)
                eta = fmt_duration(remaining)
                print(f"\n[{i}/{len(to_process)}] {audio_file.name}  (ETA: ~{eta})")
            else:
                print(f"\n[{i}/{len(to_process)}] {audio_file.name}")

            file_start = time.time()
            try:
                process_file(
                    str(audio_file), args.output_dir,
                    diarization_pipeline=diarization_pipeline,
                    skip_speakers=skip_speakers,
                    auto_speakers=args.auto,
                )
            except Exception as e:
                print(f"\n  ERROR: Failed to process {audio_file.name}: {e}")
                failed.append((audio_file.name, str(e)))
            elapsed_times.append(time.time() - file_start)

        total_time = sum(elapsed_times)
        print(f"\nDone! Processed {len(to_process) - len(failed)}/{len(to_process)} files in {fmt_duration(total_time)}.")
        if failed:
            print(f"\nFailed files ({len(failed)}):")
            for name, error in failed:
                print(f"  - {name}: {error}")
```

- [ ] **Step 2: Run existing tests to verify nothing broke**

Run: `uv run python smoke_test.py`
Expected: ALL tests pass. (`main` is not directly tested, but the functions it calls are.)

- [ ] **Step 3: Commit**

```bash
git add transcribe.py
git commit -m "feat: wire speaker flags into main, skip-speakers default in batch"
```

---

## Task 8: `--name-speakers` Mode — Post-Hoc Speaker Naming

**Files:**
- Modify: `transcribe.py` (add `name_speakers_in_files`)
- Modify: `smoke_test.py` (add 1 test)

- [ ] **Step 1: Write the failing test in `smoke_test.py`**

Add `name_speakers_in_files` to the import block, then add the test:

```python
# Add to imports:
from transcribe import (
    # ... existing imports ...
    name_speakers_in_files,
)


def test_name_speakers_in_files():
    """Test that --name-speakers replaces generic names in body and frontmatter."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a transcript with generic speaker names
        md_content = """---
title: test-interview
date: 2026-03-29
source_file: test.wav
duration: 15s
language: en
speakers:
  - Speaker 1
  - Speaker 2
tags:
  - transcript
  - interview
---

# test-interview

## Transcript

**[00:01] Speaker 1:**
Hi, I'm the hiring manager at Acme Corp.

**[00:10] Speaker 2:**
Thanks for having me, I'm really excited about this role.

**[00:20] Speaker 1:**
Tell me about your background.
"""
        (tmpdir / "test-interview.md").write_text(md_content)

        # Simulate user input: name Speaker 1 as "Smith (Acme)", Speaker 2 as "Jones"
        with patch("builtins.input", side_effect=["Smith (Acme)", "Jones"]):
            name_speakers_in_files(tmpdir)

        result = (tmpdir / "test-interview.md").read_text()
        assert "Speaker 1" not in result
        assert "Speaker 2" not in result
        assert "Smith (Acme)" in result
        assert "Jones" in result
        # Check frontmatter too
        assert "  - Smith (Acme)" in result
        assert "  - Jones" in result
    print("  name_speakers_in_files: OK")
```

Add to the `tests` list in `main()`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python smoke_test.py`
Expected: FAIL — `ImportError` because `name_speakers_in_files` doesn't exist yet.

- [ ] **Step 3: Implement `name_speakers_in_files` in `transcribe.py`**

Add before `parse_args()`:

```python
import re


def name_speakers_in_files(output_dir: Path):
    """Scan transcripts for generic 'Speaker N' names and prompt for real names.

    Reads each .md file in output_dir, finds files with generic speaker names,
    shows distinctive passages, and prompts the user to provide real names.
    """
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)

    md_files = sorted(output_dir.glob("*.md"))
    if not md_files:
        print(f"No transcript files found in {output_dir}")
        return

    # Find files with generic speaker names
    speaker_pattern = re.compile(r"Speaker \d+")
    files_to_name = []
    for md_file in md_files:
        content = md_file.read_text()
        generic_speakers = sorted(set(speaker_pattern.findall(content)))
        if generic_speakers:
            files_to_name.append((md_file, generic_speakers))

    if not files_to_name:
        print("All transcripts already have named speakers. Nothing to do.")
        return

    print(f"Found {len(files_to_name)} transcripts with unnamed speakers.\n")

    for md_file, generic_speakers in files_to_name:
        content = md_file.read_text()
        print(f"\n{'=' * 50}")
        print(f"File: {md_file.name}")
        print(f"Unnamed speakers: {', '.join(generic_speakers)}")

        # Extract blocks for passage display
        block_pattern = re.compile(
            r"\*\*\[[\d:]+\] (.+?):\*\*\n(.+?)(?=\n\n|\Z)", re.DOTALL
        )
        blocks_by_speaker = {}
        for match in block_pattern.finditer(content):
            speaker = match.group(1)
            text = match.group(2).strip()
            if speaker in generic_speakers:
                blocks_by_speaker.setdefault(speaker, []).append(text)

        name_map = {}
        for speaker in generic_speakers:
            passages = blocks_by_speaker.get(speaker, [])
            # Show longest/most distinctive passages
            substantive = [p for p in passages if len(p) > 30]
            substantive.sort(key=len, reverse=True)

            print(f"\n  {speaker}:")
            if substantive:
                print(f"    Passages:")
                for p in substantive[:3]:
                    display = p[:150] + "..." if len(p) > 150 else p
                    print(f"      > {display}")

            answer = input(f"    Name for {speaker}? [skip]: ").strip()
            if answer and answer.lower() != "skip":
                name_map[speaker] = answer
                print(f"    \u2713 {speaker} \u2192 {answer}")
            else:
                print(f"    - Keeping \"{speaker}\"")

        # Apply replacements
        if name_map:
            for old_name, new_name in name_map.items():
                content = content.replace(old_name, new_name)
            md_file.write_text(content)
            print(f"\n  Updated {md_file.name}")

    print(f"\nDone! Named speakers in {len(files_to_name)} files.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python smoke_test.py`
Expected: ALL tests pass.

- [ ] **Step 5: Commit**

```bash
git add transcribe.py smoke_test.py
git commit -m "feat: add --name-speakers mode for post-hoc speaker naming"
```

---

## Task 9: Update CLAUDE.md and Final Verification

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run the full smoke test suite**

Run: `uv run python smoke_test.py`
Expected: ALL tests pass (should be 36 tests: 26 original + 10 new).

- [ ] **Step 2: Update CLAUDE.md**

Update the following sections in `CLAUDE.md`:

In the **Pipeline order** section, update to:
```
- Pipeline order: transcribe + diarize (parallel) → assign speakers → build blocks → filter hallucinations → merge filler blocks → LLM polish → identify speakers → generate summary → format markdown
```

In the **Key details** section, add:
```
- Silero VAD preprocesses audio to detect speech segments, preventing Whisper hallucinations from silence/hold music
- Post-processing hallucination filter removes repetitive phantom text (known phrases, consecutive identical blocks, internal repetition)
- Filler micro-blocks ("Yeah.", "Hmm.", "Um.") are merged into adjacent blocks to reduce visual noise
- Batch mode defaults to `--skip-speakers` for fully unattended processing
- `--name-speakers` mode allows post-hoc speaker naming on existing transcripts
- `--auto` flag auto-accepts LLM speaker suggestions without interactive prompts
```

In the **Capability check** section, update the test count:
```
All 36 tests must pass.
```

Update the test list to include the new tests:
```
`filter_hallucinations` (patterns + preserves real), `detect_speech_segments` (mocked), `merge_filler_blocks` (merge + first block + preserves substantive), `name_speakers_in_files`, `parse_args` (--skip-speakers/--auto/--name-speakers)
```

- [ ] **Step 3: Run smoke tests one final time**

Run: `uv run python smoke_test.py`
Expected: ALL 36 TESTS PASSED

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for hallucination filtering, deferred naming, filler cleanup"
```
