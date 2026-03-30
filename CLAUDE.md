# Transcript-to-Markdown

Audio transcription pipeline: mlx-whisper (speech-to-text, Apple Silicon GPU) + pyannote (speaker diarization) + OpenAI GPT-5.4 / Gemini Flash (speaker identification) → markdown.

## Project structure

- `transcribe.py` — main script (transcribe, diarize, assign speakers, format markdown)
- `smoke_test.py` — fast end-to-end smoke test using synthetic data (no models needed)
- `setup.sh` — one-time setup (installs deps, downloads models, configures HF token)

## Capability check

After making any code changes to `transcribe.py`, always run the smoke test before declaring done:

```bash
uv run python smoke_test.py
```

All 20 tests must pass. The smoke test covers: `fmt_timestamp`, `fmt_duration`, `assign_speakers`, `normalize_speaker_names`, `build_blocks`, `format_markdown` (with YAML frontmatter), file I/O, `infer_speaker_names` (mocked OpenAI + no-key fallback + Gemini fallback), `confirm_speakers` (accept/custom/skip), speaker name replacement, `process_file` importability, `discover_audio_files`, batch skip logic, and `parse_args` (defaults/flags/files) — using synthetic data so it runs in under a second with no model downloads.

If the change involves model loading, diarization, or transcription logic that the smoke test can't cover, also do a quick manual test with a real audio file:

```bash
uv run python transcribe.py 00-unprocessed-audio-recordings/<some-file>.wav
```

Clean up any test artifacts (temp files, test transcripts) after verification.

## Key details

- Whisper runs on Apple Silicon GPU via mlx-whisper (MLX backend, ~1.8x faster than whisper-cpp)
- pyannote diarization uses MPS on Apple Silicon when available, falls back to CPU
- pyannote's `SpeakerDiarization` pipeline returns `DiarizeOutput` dataclass (not raw `Annotation`) — use `output.speaker_diarization` to get the `Annotation`
- HF_TOKEN is loaded from `.env` or environment variable
- `ProgressHook` from pyannote shows tqdm progress bars during diarization
- After diarization, LLM infers speaker names from transcript context, then prompts user to confirm/correct each name interactively
- OPENAI_API_KEY (GPT-5.4) is the default for speaker identification; falls back to GOOGLE_API_KEY (Gemini Flash) if unavailable
- If neither key is set, prompts user for an OpenAI key or skips to manual naming
- Speaker identification happens between `build_blocks()` and `format_markdown()` in the pipeline
- Default input/output directories point to Google Drive folders; override with `--input-dir` / `--output-dir`
- Batch mode (no file args) discovers all `.wav/.mp3/.m4a/.flac/.ogg/.wma` files and skips those with existing `.md` transcripts
- Single-file mode: `uv run python transcribe.py path/to/file.wav`
- Batch mode: `uv run python transcribe.py` (processes all unprocessed files in input dir)
- Output format uses YAML frontmatter (title, date, source_file, duration, language, speakers, tags) for Obsidian compatibility and LLM parseability
