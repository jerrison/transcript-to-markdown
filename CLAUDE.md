# Transcript-to-Markdown

Audio transcription pipeline: Whisper (speech-to-text) + pyannote (speaker diarization) → markdown.

## Project structure

- `transcribe.py` — main script (transcribe, diarize, assign speakers, format markdown)
- `smoke_test.py` — fast end-to-end smoke test using synthetic data (no models needed)
- `setup.sh` — one-time setup (installs deps, downloads models, configures HF token)
- `00-unprocessed-audio-recordings/` — input audio files
- `01-transcripts/` — output markdown transcripts

## Capability check

After making any code changes to `transcribe.py`, always run the smoke test before declaring done:

```bash
uv run python smoke_test.py
```

All 7 tests must pass. The smoke test covers: `fmt_timestamp`, `fmt_duration`, `assign_speakers`, `normalize_speaker_names`, `build_blocks`, `format_markdown`, and file I/O — using synthetic data so it runs in under a second with no model downloads.

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
