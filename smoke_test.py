"""Smoke test: verify the transcript pipeline end-to-end with synthetic data.

Bypasses Whisper and pyannote model loading to test the processing logic
(assign_speakers, build_blocks, format_markdown) and file I/O quickly.
"""

import sys
import tempfile
from pathlib import Path

from transcribe import (
    assign_speakers,
    build_blocks,
    fmt_duration,
    fmt_timestamp,
    format_markdown,
    normalize_speaker_names,
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
    print("  assign_speakers: OK")


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

    assert "# Transcript: test.wav" in md
    assert "**Duration**: 15s" in md
    assert "**Speakers**: 2" in md
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
        assert len(content) > 100
        print(f"  file_io: OK ({len(content)} chars written/read)")
    finally:
        tmp_path.unlink()


def main():
    print("Running smoke tests...")
    tests = [
        test_fmt_timestamp,
        test_fmt_duration,
        test_assign_speakers,
        test_normalize_speaker_names,
        test_build_blocks,
        test_format_markdown,
        test_file_io,
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
