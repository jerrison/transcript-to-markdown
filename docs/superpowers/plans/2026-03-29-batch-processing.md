# Batch Processing with Configurable Directories

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable processing all audio files in a directory in one run, with configurable input/output paths and skip-if-already-transcribed logic.

**Architecture:** Extract the single-file processing logic from `main()` into a `process_file()` function. Replace `sys.argv` parsing with `argparse` supporting `--input-dir`, `--output-dir`, and optional positional file args. Batch mode discovers `.wav`/`.mp3`/`.m4a` files, skips those with existing `.md` output, and processes each sequentially (preserving interactive speaker identification).

**Tech Stack:** Python stdlib (`argparse`, `pathlib`), no new dependencies.

---

### Task 1: Extract `process_file()` from `main()`

**Files:**
- Modify: `transcribe.py:369-435` (the `main()` function)
- Test: `smoke_test.py`

- [ ] **Step 1: Write the failing test**

Add to `smoke_test.py`:

```python
def test_process_file_exists():
    """Test that process_file function is importable."""
    from transcribe import process_file
    assert callable(process_file)
    print("  process_file exists: OK")
```

Add `test_process_file_exists` to the `tests` list in `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python smoke_test.py`
Expected: FAIL with `ImportError: cannot import name 'process_file'`

- [ ] **Step 3: Implement `process_file()`**

In `transcribe.py`, extract the processing logic from `main()` into a standalone function. The new function takes an audio file path and output directory, runs the full pipeline (transcribe → diarize → assign → build blocks → identify speakers → format → save), and returns the output path.

Replace lines 369-435 with:

```python
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma"}


def process_file(audio_path: str, output_dir: Path) -> Path:
    """Run the full transcription pipeline on a single audio file.

    Returns the path to the written markdown transcript.
    """
    audio_file = Path(audio_path)
    filename = audio_file.name

    print(f"Processing: {filename}")
    print("=" * 50)

    # Step 1: Transcribe
    words, info = transcribe(audio_path)
    print(f"  Transcribed {len(words)} words, language: {info.language}")

    # Step 2: Diarize
    speaker_segments = diarize(audio_path)
    unique_speakers = set(s["speaker"] for s in speaker_segments)
    print(f"  Found {len(unique_speakers)} speakers in {len(speaker_segments)} segments")

    # Step 3: Assign speakers to words
    words = assign_speakers(words, speaker_segments)

    # Step 4: Build dialogue blocks
    blocks = build_blocks(words)
    print(f"  Built {len(blocks)} dialogue blocks")

    # Step 5: Identify speakers
    speaker_name_map = identify_speakers(blocks)
    for block in blocks:
        block["speaker"] = speaker_name_map.get(block["speaker"], block["speaker"])

    # Step 6: Format and save
    markdown = format_markdown(
        filename=filename,
        blocks=blocks,
        duration=info.duration,
        language=info.language,
        num_speakers=len(unique_speakers),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_file.stem
    output_path = output_dir / f"{stem}.md"
    output_path.write_text(markdown)

    print("=" * 50)
    print(f"Saved transcript to {output_path}")
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python transcribe.py <audio-file>")
        print("Example: uv run python transcribe.py recording.wav")
        sys.exit(1)

    input_path = sys.argv[1]
    audio_dir = Path("00-unprocessed-audio-recordings")
    output_dir = Path("01-transcripts")

    # Resolve audio file path
    audio_file = Path(input_path)
    if not audio_file.exists():
        audio_file = audio_dir / input_path
    if not audio_file.exists():
        print(f"Error: Audio file not found: {input_path}")
        print(f"Place audio files in {audio_dir}/")
        sys.exit(1)

    process_file(str(audio_file), output_dir)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python smoke_test.py`
Expected: ALL 14 TESTS PASSED

- [ ] **Step 5: Commit**

```bash
git add transcribe.py smoke_test.py
git commit -m "refactor: extract process_file() from main()"
```

---

### Task 2: Add `argparse` with `--input-dir` and `--output-dir`

**Files:**
- Modify: `transcribe.py` (the `main()` function)
- Test: `smoke_test.py`

- [ ] **Step 1: Write the failing test**

Add to `smoke_test.py`:

```python
def test_parse_args_defaults():
    """Test that parse_args returns correct defaults."""
    from transcribe import parse_args
    args = parse_args([])
    assert args.input_dir == Path("/Users/jerrison/My Drive (jerrisonli@gmail.com)/00. Top Folder/02. Recruiting/04-interviews-audio-transcripts")
    assert args.output_dir == Path("/Users/jerrison/My Drive (jerrisonli@gmail.com)/00. Top Folder/02. Recruiting/05-interviews-transcripts")
    assert args.files == []
    print("  parse_args (defaults): OK")


def test_parse_args_with_flags():
    """Test that parse_args handles --input-dir and --output-dir."""
    from transcribe import parse_args
    args = parse_args(["--input-dir", "/tmp/in", "--output-dir", "/tmp/out"])
    assert args.input_dir == Path("/tmp/in")
    assert args.output_dir == Path("/tmp/out")
    print("  parse_args (flags): OK")


def test_parse_args_with_files():
    """Test that parse_args handles positional file arguments."""
    from transcribe import parse_args
    args = parse_args(["file1.wav", "file2.wav"])
    assert args.files == ["file1.wav", "file2.wav"]
    print("  parse_args (files): OK")
```

Add all three to the `tests` list in `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python smoke_test.py`
Expected: FAIL with `ImportError: cannot import name 'parse_args'`

- [ ] **Step 3: Implement `parse_args()` and update `main()`**

Add `import argparse` at the top of `transcribe.py` (with the existing stdlib imports).

Add `parse_args()` just before `main()`:

```python
DEFAULT_INPUT_DIR = Path(
    "/Users/jerrison/My Drive (jerrisonli@gmail.com)"
    "/00. Top Folder/02. Recruiting/04-interviews-audio-transcripts"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/jerrison/My Drive (jerrisonli@gmail.com)"
    "/00. Top Folder/02. Recruiting/05-interviews-transcripts"
)


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
    return parser.parse_args(argv)
```

Replace `main()` with:

```python
def main():
    args = parse_args()

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
            process_file(str(audio_file), args.output_dir)
    else:
        print("Usage: uv run python transcribe.py [files...] [--input-dir DIR] [--output-dir DIR]")
        print("  No files specified — batch mode coming in next task.")
        sys.exit(1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python smoke_test.py`
Expected: ALL 17 TESTS PASSED

- [ ] **Step 5: Commit**

```bash
git add transcribe.py smoke_test.py
git commit -m "feat: add argparse with --input-dir and --output-dir flags"
```

---

### Task 3: Add batch mode with skip-if-transcribed

**Files:**
- Modify: `transcribe.py` (the `main()` function)
- Test: `smoke_test.py`

- [ ] **Step 1: Write the failing test for `discover_audio_files()`**

Add to `smoke_test.py`:

```python
def test_discover_audio_files():
    """Test that discover_audio_files finds audio files and skips non-audio."""
    from transcribe import discover_audio_files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # Create test files
        (tmpdir / "interview1.wav").touch()
        (tmpdir / "interview2.mp3").touch()
        (tmpdir / "interview3.m4a").touch()
        (tmpdir / "notes.txt").touch()
        (tmpdir / ".DS_Store").touch()
        (tmpdir / "Icon\r").touch()

        files = discover_audio_files(tmpdir)
        names = [f.name for f in files]
        assert len(files) == 3, f"Expected 3 audio files, got {len(files)}: {names}"
        assert "interview1.wav" in names
        assert "interview2.mp3" in names
        assert "interview3.m4a" in names
        assert "notes.txt" not in names
    print("  discover_audio_files: OK")
```

Add to the `tests` list.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python smoke_test.py`
Expected: FAIL with `ImportError: cannot import name 'discover_audio_files'`

- [ ] **Step 3: Implement `discover_audio_files()`**

Add to `transcribe.py`, just after `AUDIO_EXTENSIONS`:

```python
def discover_audio_files(input_dir: Path) -> list[Path]:
    """Find all audio files in the input directory, sorted by name."""
    files = [
        f for f in sorted(input_dir.iterdir())
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return files
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python smoke_test.py`
Expected: ALL 18 TESTS PASSED

- [ ] **Step 5: Write the failing test for skip logic**

Add to `smoke_test.py`:

```python
def test_batch_skips_existing_transcripts():
    """Test that batch mode skips files that already have transcripts."""
    from transcribe import discover_audio_files
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "audio"
        output_dir = Path(tmpdir) / "transcripts"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create audio files
        (input_dir / "done.wav").touch()
        (input_dir / "pending.wav").touch()

        # Create existing transcript for "done"
        (output_dir / "done.md").write_text("# already done")

        all_files = discover_audio_files(input_dir)
        to_process = [
            f for f in all_files
            if not (output_dir / f"{f.stem}.md").exists()
        ]

        assert len(to_process) == 1
        assert to_process[0].name == "pending.wav"
    print("  batch_skips_existing: OK")
```

Add to the `tests` list.

- [ ] **Step 6: Run test to verify it passes**

This test doesn't require new code (it tests the filtering logic inline). Run to confirm:

Run: `uv run python smoke_test.py`
Expected: ALL 19 TESTS PASSED

- [ ] **Step 7: Wire batch mode into `main()`**

Update the `else` branch in `main()`:

```python
def main():
    args = parse_args()

    if args.files:
        # Single-file mode: process specific files
        for file_path in args.files:
            audio_file = Path(file_path)
            if not audio_file.exists():
                audio_file = args.input_dir / file_path
            if not audio_file.exists():
                print(f"Error: Audio file not found: {file_path}")
                sys.exit(1)
            process_file(str(audio_file), args.output_dir)
    else:
        # Batch mode: process all audio files in input dir
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

        for i, audio_file in enumerate(to_process, 1):
            print(f"\n[{i}/{len(to_process)}] {audio_file.name}")
            process_file(str(audio_file), args.output_dir)

        print(f"\nDone! Processed {len(to_process)} files.")
```

- [ ] **Step 8: Run all tests**

Run: `uv run python smoke_test.py`
Expected: ALL 19 TESTS PASSED

- [ ] **Step 9: Commit**

```bash
git add transcribe.py smoke_test.py
git commit -m "feat: add batch mode with auto-discovery and skip-if-transcribed"
```

---

### Task 4: Update `smoke_test.py` imports and `CLAUDE.md`

**Files:**
- Modify: `smoke_test.py` (imports at top)
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update `smoke_test.py` imports**

Update the import block at the top of `smoke_test.py` to include the new functions:

```python
from transcribe import (
    assign_speakers,
    build_blocks,
    confirm_speakers,
    discover_audio_files,
    fmt_duration,
    fmt_timestamp,
    format_markdown,
    infer_speaker_names,
    normalize_speaker_names,
    parse_args,
    process_file,
)
```

- [ ] **Step 2: Run all tests**

Run: `uv run python smoke_test.py`
Expected: ALL 19 TESTS PASSED

- [ ] **Step 3: Update CLAUDE.md**

Add to the "Key details" section in `CLAUDE.md`:

```markdown
- Default input/output directories point to Google Drive interview folders; override with `--input-dir` / `--output-dir`
- Batch mode (no file args) discovers all `.wav/.mp3/.m4a/.flac/.ogg/.wma` files and skips those with existing `.md` transcripts
- Single-file mode: `uv run python transcribe.py path/to/file.wav`
- Batch mode: `uv run python transcribe.py` (processes all unprocessed files in input dir)
```

Update the "Capability check" test count from 13 to 19.

- [ ] **Step 4: Commit**

```bash
git add smoke_test.py CLAUDE.md
git commit -m "docs: update CLAUDE.md and imports for batch processing"
```
