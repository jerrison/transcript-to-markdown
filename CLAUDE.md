# Transcript-to-Markdown

Audio transcription pipeline: Whisper (speech-to-text) + pyannote (speaker diarization) + Gemini Flash (speaker identification) → markdown.

## Project structure

- `transcribe.py` — main script (transcribe, diarize, assign speakers, format markdown)
- `smoke_test.py` — fast end-to-end smoke test using synthetic data (no models needed)
- `setup.sh` — one-time setup (installs deps, downloads models, configures HF token)

## Capability check

After making any code changes to `transcribe.py`, always run the smoke test before declaring done:

```bash
uv run python smoke_test.py
```

All 19 tests must pass. The smoke test covers: `fmt_timestamp`, `fmt_duration`, `assign_speakers`, `normalize_speaker_names`, `build_blocks`, `format_markdown` (with YAML frontmatter), file I/O, `infer_speaker_names` (mocked LLM + no-key fallback), `confirm_speakers` (accept/custom/skip), speaker name replacement, `process_file` importability, `discover_audio_files`, batch skip logic, and `parse_args` (defaults/flags/files) — using synthetic data so it runs in under a second with no model downloads.

If the change involves model loading, diarization, or transcription logic that the smoke test can't cover, also do a quick manual test with a real audio file:

```bash
uv run python transcribe.py 00-unprocessed-audio-recordings/<some-file>.wav
```

Clean up any test artifacts (temp files, test transcripts) after verification.

## Key details

- Whisper runs on CPU via faster-whisper (CTranslate2 backend, no MPS support)
- pyannote diarization uses MPS on Apple Silicon when available, falls back to CPU
- pyannote's `SpeakerDiarization` pipeline returns `DiarizeOutput` dataclass (not raw `Annotation`) — use `output.speaker_diarization` to get the `Annotation`
- HF_TOKEN is loaded from `.env` or environment variable
- `ProgressHook` from pyannote shows tqdm progress bars during diarization
- After diarization, Gemini Flash infers speaker names from transcript context, then prompts user to confirm/correct each name interactively
- GOOGLE_API_KEY loaded from `.env` or environment variable (optional — gracefully degrades to manual naming if missing)
- Speaker identification happens between `build_blocks()` and `format_markdown()` in the pipeline
- Default input/output directories point to Google Drive folders; override with `--input-dir` / `--output-dir`
- Batch mode (no file args) discovers all `.wav/.mp3/.m4a/.flac/.ogg/.wma` files and skips those with existing `.md` transcripts
- Single-file mode: `uv run python transcribe.py path/to/file.wav`
- Batch mode: `uv run python transcribe.py` (processes all unprocessed files in input dir)
- Output format uses YAML frontmatter (title, date, source_file, duration, language, speakers, tags) for Obsidian compatibility and LLM parseability
