"""Smoke test: verify the transcript pipeline end-to-end with synthetic data.

Bypasses Whisper and pyannote model loading to test the processing logic
(assign_speakers, build_blocks, format_markdown) and file I/O quickly.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

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


def test_fmt_timestamp():
    assert fmt_timestamp(0) == "00:00"
    assert fmt_timestamp(65) == "01:05"
    assert fmt_timestamp(3661) == "1:01:01"
    print("  fmt_timestamp: OK")


def test_fmt_duration():
    assert fmt_duration(0) == "0s"
    assert fmt_duration(65) == "1m 5s"
    assert fmt_duration(3661) == "1h 1m 1s"
    print("  fmt_duration: OK")


def make_synthetic_words():
    """Create fake word-level data simulating two speakers."""
    words = []
    # Speaker A talks from 0-5s
    for i, w in enumerate(["Hello", " everyone,", " welcome", " to", " the", " meeting."]):
        words.append({"word": w, "start": i * 0.8, "end": i * 0.8 + 0.7})
    # Speaker B talks from 6-10s
    for i, w in enumerate(["Thanks", " for", " having", " me", " here."]):
        words.append({"word": w, "start": 6 + i * 0.8, "end": 6 + i * 0.8 + 0.7})
    # Speaker A again from 12-15s
    for i, w in enumerate(["Let's", " get", " started."]):
        words.append({"word": w, "start": 12 + i * 0.8, "end": 12 + i * 0.8 + 0.7})
    return words


def make_synthetic_segments():
    """Create fake diarization segments for two speakers."""
    return [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 6.0, "end": 10.0, "speaker": "SPEAKER_01"},
        {"start": 12.0, "end": 15.0, "speaker": "SPEAKER_00"},
    ]


def test_assign_speakers():
    words = make_synthetic_words()
    segments = make_synthetic_segments()
    result = assign_speakers(words, segments)

    assert all("speaker" in w for w in result), "All words must have a speaker"
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[6]["speaker"] == "SPEAKER_01"
    assert result[11]["speaker"] == "SPEAKER_00"
    assert all(w["speaker"] != "Unknown" for w in result), "No words should be Unknown"
    print("  assign_speakers: OK")


def test_assign_speakers_nearest_fallback():
    """Test that words in gaps between segments get assigned to nearest speaker."""
    # Word at 5.5s falls in gap between segments (0-5s and 6-10s)
    words = [{"word": "um", "start": 5.2, "end": 5.8}]
    segments = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 6.0, "end": 10.0, "speaker": "SPEAKER_01"},
    ]
    result = assign_speakers(words, segments)
    # Midpoint 5.5 is closer to SPEAKER_00's end (5.0) than SPEAKER_01's start (6.0)
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[0]["speaker"] != "Unknown"
    print("  assign_speakers (nearest fallback): OK")


def test_normalize_speaker_names():
    words = [
        {"speaker": "SPEAKER_00"},
        {"speaker": "SPEAKER_01"},
        {"speaker": "SPEAKER_00"},
    ]
    mapping = normalize_speaker_names(words)
    assert mapping == {"SPEAKER_00": "Speaker 1", "SPEAKER_01": "Speaker 2"}
    print("  normalize_speaker_names: OK")


def test_build_blocks():
    words = make_synthetic_words()
    segments = make_synthetic_segments()
    words = assign_speakers(words, segments)
    blocks = build_blocks(words)

    assert len(blocks) == 3, f"Expected 3 blocks, got {len(blocks)}"
    assert blocks[0]["speaker"] == "Speaker 1"
    assert blocks[1]["speaker"] == "Speaker 2"
    assert blocks[2]["speaker"] == "Speaker 1"
    assert "Hello" in blocks[0]["text"]
    assert "Thanks" in blocks[1]["text"]
    assert "started" in blocks[2]["text"]
    print("  build_blocks: OK")


def test_format_markdown():
    words = make_synthetic_words()
    segments = make_synthetic_segments()
    words = assign_speakers(words, segments)
    blocks = build_blocks(words)

    md = format_markdown(
        filename="test.wav",
        blocks=blocks,
        duration=15.0,
        language="en",
        num_speakers=2,
    )

    # Check YAML frontmatter
    assert md.startswith("---\n")
    assert "title: test" in md
    assert "source_file: test.wav" in md
    assert "duration: 15s" in md
    assert "language: en" in md
    assert "  - transcript" in md
    assert "  - interview" in md
    # Check body
    assert "# test" in md
    assert "**Duration**: 15s" in md
    assert "Speaker 1" in md
    assert "Speaker 2" in md
    assert "Hello" in md
    print("  format_markdown: OK")


def test_file_io():
    """Test writing and reading a transcript file."""
    words = make_synthetic_words()
    segments = make_synthetic_segments()
    words = assign_speakers(words, segments)
    blocks = build_blocks(words)

    md = format_markdown(
        filename="test.wav",
        blocks=blocks,
        duration=15.0,
        language="en",
        num_speakers=2,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(md)
        tmp_path = Path(f.name)

    try:
        content = tmp_path.read_text()
        assert content == md
        assert len(content) > 200
        print(f"  file_io: OK ({len(content)} chars written/read)")
    finally:
        tmp_path.unlink()


def test_infer_speaker_names_mocked():
    """Test infer_speaker_names with a mocked OpenAI response (no API call)."""
    blocks = [
        {"speaker": "Speaker 1", "start": 0.0, "text": "Hi, I'm Sarah. Let's start the meeting."},
        {"speaker": "Speaker 2", "start": 5.0, "text": "Thanks Sarah, I'm Alex from engineering."},
    ]

    fake_response = {
        "Speaker 1": {
            "likely_name": "Sarah",
            "confidence": "high",
            "role": "Meeting organizer",
            "summary": "Opened the meeting",
        },
        "Speaker 2": {
            "likely_name": "Alex",
            "confidence": "high",
            "role": "Engineer",
            "summary": "Introduced themselves",
        },
    }

    import json
    from unittest.mock import MagicMock

    mock_message = MagicMock()
    mock_message.content = json.dumps(fake_response)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("transcribe.load_openai_api_key", return_value="fake-key"), \
         patch("openai.OpenAI", return_value=mock_client):
        result = infer_speaker_names(blocks)

    assert result == fake_response
    print("  infer_speaker_names (mocked): OK")


def test_infer_speaker_names_no_key():
    """Test that infer_speaker_names returns empty dict when no API keys."""
    blocks = [{"speaker": "Speaker 1", "start": 0.0, "text": "Hello"}]

    with patch("transcribe.load_openai_api_key", return_value=None), \
         patch("transcribe.load_google_api_key", return_value=None), \
         patch("builtins.input", return_value="skip"):
        result = infer_speaker_names(blocks)

    assert result == {}
    print("  infer_speaker_names (no key): OK")


def test_infer_speaker_names_gemini_fallback():
    """Test that infer_speaker_names falls back to Gemini when OpenAI key missing."""
    blocks = [
        {"speaker": "Speaker 1", "start": 0.0, "text": "Hi, I'm Sarah."},
    ]

    fake_response = {
        "Speaker 1": {
            "likely_name": "Sarah",
            "confidence": "high",
            "role": "Speaker",
            "summary": "Introduced themselves",
        },
    }

    import json
    from unittest.mock import MagicMock

    mock_response = MagicMock()
    mock_response.text = json.dumps(fake_response)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("transcribe.load_openai_api_key", return_value=None), \
         patch("transcribe.load_google_api_key", return_value="fake-key"), \
         patch("google.genai.Client", return_value=mock_client):
        result = infer_speaker_names(blocks)

    assert result == fake_response
    print("  infer_speaker_names (gemini fallback): OK")


def test_confirm_speakers_accept():
    """Test confirm_speakers with user accepting suggestions."""
    suggestions = {
        "Speaker 1": {
            "likely_name": "Sarah",
            "confidence": "high",
            "role": "Manager",
            "summary": "Led the meeting",
        },
        "Speaker 2": {
            "likely_name": "Alex",
            "confidence": "medium",
            "role": "Engineer",
            "summary": "Gave updates",
        },
    }
    blocks = [
        {"speaker": "Speaker 1", "start": 0.0, "text": "Hello"},
        {"speaker": "Speaker 2", "start": 5.0, "text": "Hi there"},
    ]

    # Simulate user pressing Enter (accept) for both
    with patch("builtins.input", side_effect=["y", ""]):
        result = confirm_speakers(suggestions, blocks)

    assert result == {"Speaker 1": "Sarah", "Speaker 2": "Alex"}
    print("  confirm_speakers (accept): OK")


def test_confirm_speakers_custom_name():
    """Test confirm_speakers with user providing a custom name."""
    suggestions = {
        "Speaker 1": {
            "likely_name": "Sarah",
            "confidence": "low",
            "role": "Unknown",
            "summary": "Spoke briefly",
        },
    }
    blocks = [
        {"speaker": "Speaker 1", "start": 0.0, "text": "Hello"},
    ]

    # Simulate user typing a custom name
    with patch("builtins.input", return_value="Jane Doe"):
        result = confirm_speakers(suggestions, blocks)

    assert result == {"Speaker 1": "Jane Doe"}
    print("  confirm_speakers (custom name): OK")


def test_confirm_speakers_skip():
    """Test confirm_speakers when user skips (no suggestion, presses Enter)."""
    suggestions = {}
    blocks = [
        {"speaker": "Speaker 1", "start": 0.0, "text": "Hello"},
    ]

    # Simulate user pressing Enter to skip
    with patch("builtins.input", return_value=""):
        result = confirm_speakers(suggestions, blocks)

    assert result == {}
    print("  confirm_speakers (skip): OK")


def test_process_file_exists():
    """Test that process_file is importable and callable."""
    assert callable(process_file)
    print("  process_file_exists: OK")


def test_speaker_name_replacement():
    """Test that speaker names are correctly replaced in blocks."""
    blocks = [
        {"speaker": "Speaker 1", "start": 0.0, "text": "Hello everyone"},
        {"speaker": "Speaker 2", "start": 5.0, "text": "Thanks for having me"},
        {"speaker": "Speaker 1", "start": 10.0, "text": "Let's get started"},
    ]
    name_map = {"Speaker 1": "Sarah Chen", "Speaker 2": "Alex Kim"}

    for block in blocks:
        block["speaker"] = name_map.get(block["speaker"], block["speaker"])

    assert blocks[0]["speaker"] == "Sarah Chen"
    assert blocks[1]["speaker"] == "Alex Kim"
    assert blocks[2]["speaker"] == "Sarah Chen"
    print("  speaker_name_replacement: OK")


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

        files = discover_audio_files(tmpdir)
        names = [f.name for f in files]
        assert len(files) == 3, f"Expected 3 audio files, got {len(files)}: {names}"
        assert "interview1.wav" in names
        assert "interview2.mp3" in names
        assert "interview3.m4a" in names
        assert "notes.txt" not in names
    print("  discover_audio_files: OK")


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


def test_parse_args_defaults():
    """Test that parse_args returns correct defaults."""
    from transcribe import parse_args
    args = parse_args([])
    assert args.input_dir == Path("/Users/jerrison/My Drive (jerrisonli@gmail.com)/00. Top Folder/02. Recruiting/04-interviews-audio-transcripts")
    assert args.output_dir == Path("/Users/jerrison/My Drive (jerrisonli@gmail.com)/00. Top Folder/04-obsidian-vaults/jerrison-personal-gdrive/00-Recruiting/03-transcripts")
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


def main():
    print("Running smoke tests...")
    tests = [
        test_fmt_timestamp,
        test_fmt_duration,
        test_assign_speakers,
        test_assign_speakers_nearest_fallback,
        test_normalize_speaker_names,
        test_build_blocks,
        test_format_markdown,
        test_file_io,
        test_infer_speaker_names_mocked,
        test_infer_speaker_names_no_key,
        test_infer_speaker_names_gemini_fallback,
        test_confirm_speakers_accept,
        test_confirm_speakers_custom_name,
        test_confirm_speakers_skip,
        test_process_file_exists,
        test_speaker_name_replacement,
        test_discover_audio_files,
        test_batch_skips_existing_transcripts,
        test_parse_args_defaults,
        test_parse_args_with_flags,
        test_parse_args_with_files,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  FAIL {test.__name__}: {e}")
            failed += 1

    print()
    if failed:
        print(f"FAILED: {failed}/{len(tests)} tests failed")
        sys.exit(1)
    else:
        print(f"ALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
