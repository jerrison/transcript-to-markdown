# Transcript Quality Improvements

Three independent improvements to the transcription pipeline addressing accuracy, workflow, and robustness issues observed during the first real batch run of 12 interview recordings.

## Problem Statement

After batch-processing 12 interview audio files, three categories of issues emerged:

1. **Whisper hallucinations**: Silence/hold music at the start of video call recordings causes Whisper to generate repetitive phantom text ("Thank you." x19, "Bye bye bye..." x110). Observed in at least 2 of 12 files, consuming the first 7-10 minutes of transcript with garbage.
2. **Painful interactive workflow**: Batch mode requires sitting through 12 speaker confirmation prompts one at a time. The user prefers to transcribe first, name speakers later at their own pace.
3. **Micro-block clutter**: Many tiny blocks with just "Yeah.", "Hmm.", "Um." create visual noise in the transcript without adding substance.

## Improvement 1: Whisper Hallucination Filtering

Two-layer defense: prevent hallucinations at the source, then catch any that slip through.

### Layer 1: VAD Preprocessing

Add Silero VAD (Voice Activity Detection) as a preprocessing step before Whisper transcription.

**New dependency:** `silero-vad` (pip package). Requires PyTorch (already a dependency via pyannote).

**New function: `detect_speech_segments(audio_path) -> list[dict]`**
- Load audio at 16kHz mono (Silero VAD requirement)
- Run `get_speech_timestamps()` to identify speech regions
- Return list of `{"start": float, "end": float}` in seconds

**New function: `prepare_vad_audio(audio_path, speech_segments) -> tuple[str, list]`**
- Concatenate only the speech portions into a temporary WAV file
- Build an offset map: `[(trimmed_start, original_start, duration), ...]`
- Return the temp file path and the offset map

**Modified: `transcribe(audio_path, speech_segments=None)`**
- If `speech_segments` is provided, call `prepare_vad_audio()` to create trimmed audio
- Transcribe the trimmed audio
- Remap all word timestamps back to original timeline using the offset map
- Clean up the temp file

**Pipeline change in `process_file()`:**
- Run VAD before the parallel transcribe+diarize step
- Pass speech segments to `transcribe()`
- VAD is fast (~2-3 seconds), so minimal impact on total processing time

### Layer 2: Post-Processing Hallucination Filter

Safety net applied after `build_blocks()`.

**New function: `filter_hallucinations(blocks) -> list[dict]`**

Detection heuristics (a block is removed if ANY match):
1. **Consecutive repetition**: Block text matches the previous block's text exactly (after strip/lower), and the run of identical blocks is >= 3
2. **Known hallucination phrases**: Standalone block text (stripped) is one of: "Thank you.", "Thanks for watching.", "Please subscribe.", "Bye.", "Thank you for watching."
3. **Internal repetition**: A single block contains the same short phrase (<=3 words) repeated 5+ times (catches "bye bye bye bye..." pattern)

Removed blocks are logged: `"Filtered N hallucinated blocks"`.

Returns the cleaned block list.

### Testing

- Smoke test: `test_filter_hallucinations` — synthetic blocks with known hallucination patterns, verify they're removed
- Smoke test: `test_filter_hallucinations_preserves_real` — verify real dialogue blocks are not removed
- Smoke test: `test_detect_speech_segments` — mock Silero VAD, verify segment format
- Manual test: re-run on `Crusoe HM Pricing PM` and `Finch Legal Recruiter Screen` files, verify hallucinations are gone and real content preserved

## Improvement 2: Deferred Speaker Naming

Separate transcription from speaker identification so batch runs are fully unattended.

### Batch Mode Changes

**New flag: `--skip-speakers`**
- Skips LLM speaker identification and interactive confirmation entirely
- Output uses generic names: "Speaker 1", "Speaker 2", etc.
- Default behavior in batch mode (no file args). Single-file mode keeps current interactive default.

**New flag: `--auto`**
- Auto-accepts LLM speaker name suggestions without prompting
- All suggestions accepted regardless of confidence (user can fix in the `.md` file)
- Useful when LLM suggestions are trusted

### New `--name-speakers` Mode

**Usage:** `uv run python transcribe.py --name-speakers [--output-dir DIR]`

Scans the output directory for `.md` transcript files that still have generic "Speaker N" names.

**For each file with unnamed speakers:**
1. Parse the markdown to extract blocks and identify generic speaker names
2. Show distinctive passages for each speaker (same UX as current `confirm_speakers`)
3. Prompt user for each speaker's real name (user types their preferred format, e.g., "Riveron (Crusoe)")
4. Replace all occurrences of the generic name in the transcript body
5. Update the YAML frontmatter `speakers:` list
6. Write the updated file

**The user can:**
- Process all unnamed files in one session
- Skip individual files or speakers (press Enter)
- Stop at any time and resume later (only files with "Speaker N" still present are shown)

### Testing

- Smoke test: `test_name_speakers_replaces` — verify speaker names are replaced in both body and frontmatter
- Smoke test: `test_name_speakers_skips_already_named` — verify files without "Speaker N" are skipped
- Smoke test: `test_skip_speakers_flag` — verify `--skip-speakers` skips identification
- Smoke test: `test_auto_flag` — verify `--auto` accepts LLM suggestions without input

## Improvement 3: Filler/Micro-Block Cleanup

Merge short filler blocks into adjacent blocks to reduce visual noise.

### Implementation

**New function: `merge_filler_blocks(blocks) -> list[dict]`**

A block is classified as "filler" if:
- It contains <= 5 words, AND
- All words (after stripping punctuation) are in the filler set: {"yeah", "hmm", "um", "okay", "uh", "mm-hmm", "right", "sure", "so", "yep", "mhm", "uh-huh", "oh", "ah"}

Filler blocks are merged:
- Append the filler text to the **previous** block's text (separated by a space)
- If the filler block is the very first block, prepend to the next block instead
- The merged block retains the original block's speaker and timestamp

**Pipeline insertion:** After `filter_hallucinations()`, before `polish_transcript()`.

Order in `process_file()`:
1. `assign_speakers()`
2. `build_blocks()`
3. `filter_hallucinations()` — remove phantom blocks
4. `merge_filler_blocks()` — clean up micro-blocks
5. `polish_transcript()` — LLM fixes on clean text
6. `identify_speakers()` / skip / auto
7. `generate_summary()`
8. `format_markdown()`

### Testing

- Smoke test: `test_merge_filler_blocks` — verify filler blocks are merged into previous
- Smoke test: `test_merge_filler_blocks_first` — verify first-block filler merges into next
- Smoke test: `test_merge_filler_preserves_substantive` — verify short non-filler blocks ("That's correct.") are kept

## Pipeline Order (Updated)

```
Audio file
  |
  v
detect_speech_segments()        -- NEW: Silero VAD
  |
  v
transcribe() + diarize()        -- parallel, transcribe uses VAD segments
  |
  v
assign_speakers()
  |
  v
build_blocks()
  |
  v
filter_hallucinations()         -- NEW: remove phantom blocks
  |
  v
merge_filler_blocks()           -- NEW: clean up micro-blocks
  |
  v
polish_transcript()             -- LLM fixes
  |
  v
identify_speakers()             -- CHANGED: skip in batch, --name-speakers later
  |
  v
generate_summary()
  |
  v
format_markdown()
  |
  v
.md file
```

## Dependencies

- `silero-vad` — new pip dependency for VAD preprocessing
- `soundfile` or `torchaudio` — for loading/saving audio at 16kHz (torchaudio already available via pyannote)

## Flags Summary

| Flag | Behavior |
|------|----------|
| (no args) | Batch mode, `--skip-speakers` implied |
| `file.wav` | Single-file mode, interactive speaker naming |
| `--skip-speakers` | Skip speaker identification (default in batch) |
| `--auto` | Auto-accept LLM speaker suggestions |
| `--name-speakers` | Post-hoc speaker naming for existing transcripts |
| `--input-dir` | Override input directory |
| `--output-dir` | Override output directory |

**Mutual exclusivity:** `--name-speakers` is a standalone mode — it does not transcribe. It is incompatible with file args, `--skip-speakers`, and `--auto`. The script should error if these are combined.
