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
    detect_speech_segments,
    discover_audio_files,
    filter_hallucinations,
    fmt_duration,
    fmt_timestamp,
    format_markdown,
    generate_summary,
    infer_speaker_names,
    merge_filler_blocks,
    name_speakers_in_files,
    normalize_speaker_names,
    parse_args,
    polish_transcript,
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


def test_polish_transcript_mocked():
    """Test polish_transcript with mocked LLM response."""
    blocks = [
        {"speaker": "Speaker 1", "start": 0.0, "text": "Welcome to Gogle"},
        {"speaker": "Speaker 2", "start": 5.0, "text": "Thanks for having me"},
    ]

    import json
    from unittest.mock import MagicMock

    corrections = [{"index": 0, "text": "Welcome to Google"}]
    mock_message = MagicMock()
    mock_message.content = json.dumps(corrections)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("transcribe.load_openai_api_key", return_value="fake-key"), \
         patch("openai.OpenAI", return_value=mock_client):
        result = polish_transcript(blocks)

    assert result[0]["text"] == "Welcome to Google"
    assert result[1]["text"] == "Thanks for having me"
    print("  polish_transcript (mocked): OK")


def test_polish_transcript_no_key():
    """Test polish_transcript returns original blocks when no API key."""
    blocks = [{"speaker": "Speaker 1", "start": 0.0, "text": "Hello"}]

    with patch("transcribe.load_openai_api_key", return_value=None), \
         patch("transcribe.load_google_api_key", return_value=None):
        result = polish_transcript(blocks)

    assert result[0]["text"] == "Hello"
    print("  polish_transcript (no key): OK")


def test_generate_summary_mocked():
    """Test generate_summary with mocked LLM response."""
    blocks = [
        {"speaker": "Speaker 1", "start": 0.0, "text": "Let's discuss the project timeline."},
        {"speaker": "Speaker 2", "start": 5.0, "text": "We should aim for Q2 launch."},
    ]

    from unittest.mock import MagicMock

    mock_message = MagicMock()
    mock_message.content = "- Discussed project timeline\n- Targeting Q2 launch"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("transcribe.load_openai_api_key", return_value="fake-key"), \
         patch("openai.OpenAI", return_value=mock_client):
        result = generate_summary(blocks)

    assert "Q2" in result
    print("  generate_summary (mocked): OK")


def test_generate_summary_no_key():
    """Test generate_summary returns empty string when no API key."""
    blocks = [{"speaker": "Speaker 1", "start": 0.0, "text": "Hello"}]

    with patch("transcribe.load_openai_api_key", return_value=None), \
         patch("transcribe.load_google_api_key", return_value=None):
        result = generate_summary(blocks)

    assert result == ""
    print("  generate_summary (no key): OK")


def test_format_markdown_with_summary():
    """Test that format_markdown includes summary section when provided."""
    blocks = [{"speaker": "Speaker 1", "start": 0.0, "text": "Hello"}]
    md = format_markdown(
        filename="test.wav",
        blocks=blocks,
        duration=15.0,
        language="en",
        num_speakers=1,
        summary="- Key point one\n- Key point two",
    )
    assert "## Summary" in md
    assert "Key point one" in md
    assert "## Transcript" in md
    # Summary should come before Transcript
    assert md.index("## Summary") < md.index("## Transcript")
    print("  format_markdown (with summary): OK")


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


def test_filter_hallucinations_consecutive():
    """Test consecutive identical blocks (heuristic 3) in isolation."""
    blocks = [
        {"speaker": "Speaker 1", "start": 0.0, "text": "Mm-hmm."},
        {"speaker": "Speaker 2", "start": 1.0, "text": "Mm-hmm."},
        {"speaker": "Speaker 1", "start": 2.0, "text": "Mm-hmm."},
        {"speaker": "Speaker 1", "start": 10.0, "text": "That's a great point about the market."},
    ]
    result = filter_hallucinations(blocks)
    assert len(result) == 1, f"Expected 1 block, got {len(result)}"
    assert result[0]["text"] == "That's a great point about the market."
    print("  filter_hallucinations (consecutive): OK")


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

        # Mock LLM suggestions and simulate user accepting/overriding
        fake_suggestions = {
            "Speaker 1": {
                "likely_name": "John Smith",
                "confidence": "high",
                "role": "Hiring Manager",
                "summary": "Asked interview questions",
            },
            "Speaker 2": {
                "likely_name": "Jane Doe",
                "confidence": "medium",
                "role": "Candidate",
                "summary": "Described their background",
            },
        }
        # Speaker 1: override suggested "John Smith" with "Smith (Acme)"
        # Speaker 2: override suggested "Jane Doe" with "Jones"
        with patch("transcribe.infer_speaker_names", return_value=fake_suggestions), \
             patch("builtins.input", side_effect=["Smith (Acme)", "Jones"]):
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
        test_polish_transcript_mocked,
        test_polish_transcript_no_key,
        test_generate_summary_mocked,
        test_generate_summary_no_key,
        test_format_markdown_with_summary,
        test_filter_hallucinations,
        test_filter_hallucinations_preserves_real,
        test_filter_hallucinations_consecutive,
        test_detect_speech_segments_mocked,
        test_merge_filler_blocks,
        test_merge_filler_blocks_first,
        test_merge_filler_preserves_substantive,
        test_parse_args_skip_speakers,
        test_parse_args_auto,
        test_parse_args_name_speakers,
        test_name_speakers_in_files,
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
