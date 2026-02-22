"""Transcribe an audio file with speaker diarization to markdown."""

import os
import sys
from datetime import date
from pathlib import Path


def fmt_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def fmt_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    parts = []
    if h > 0:
        parts.append(f"{h}h")
    if m > 0:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def transcribe(audio_path: str) -> list[dict]:
    """Run faster-whisper on the audio file, return list of words with timestamps."""
    from faster_whisper import WhisperModel

    print("Loading Whisper large-v3-turbo model...")
    model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")

    print("Transcribing audio...")
    segments, info = model.transcribe(audio_path, word_timestamps=True)

    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append({
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                })

    return words, info


def load_hf_token() -> str:
    """Load HuggingFace token from environment or .env file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()

    print("Error: HF_TOKEN not found.")
    print("Run ./setup.sh or set HF_TOKEN environment variable.")
    sys.exit(1)


def diarize(audio_path: str) -> list[dict]:
    """Run pyannote speaker diarization, return list of speaker segments."""
    import torch
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    hf_token = load_hf_token()

    print("Loading pyannote diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU) acceleration")
        pipeline.to(torch.device("mps"))
    else:
        print("Using CPU (MPS not available)")

    print("Diarizing audio...")
    with ProgressHook() as hook:
        output = pipeline(audio_path, hook=hook)

    # pipeline returns DiarizeOutput dataclass; extract the Annotation
    diarization = getattr(output, "speaker_diarization", output)

    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    return speaker_segments


def assign_speakers(words: list[dict], speaker_segments: list[dict]) -> list[dict]:
    """Assign a speaker to each word by maximum timestamp overlap."""
    for word in words:
        best_speaker = "Unknown"
        best_overlap = 0.0

        for seg in speaker_segments:
            overlap_start = max(word["start"], seg["start"])
            overlap_end = min(word["end"], seg["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg["speaker"]

        word["speaker"] = best_speaker

    return words


def normalize_speaker_names(words: list[dict]) -> dict[str, str]:
    """Map raw speaker IDs (SPEAKER_00) to sequential names (Speaker 1)."""
    raw_ids = []
    for w in words:
        if w["speaker"] not in raw_ids and w["speaker"] != "Unknown":
            raw_ids.append(w["speaker"])

    return {raw: f"Speaker {i + 1}" for i, raw in enumerate(raw_ids)}


def build_blocks(words: list[dict], pause_threshold: float = 1.5) -> list[dict]:
    """Group consecutive same-speaker words into dialogue blocks.

    A new block starts on speaker change or a pause > pause_threshold seconds.
    """
    if not words:
        return []

    speaker_map = normalize_speaker_names(words)

    blocks = []
    current_speaker = words[0]["speaker"]
    current_text = words[0]["word"]
    block_start = words[0]["start"]
    prev_end = words[0]["end"]

    for w in words[1:]:
        pause = w["start"] - prev_end
        if w["speaker"] != current_speaker or pause > pause_threshold:
            blocks.append({
                "speaker": speaker_map.get(current_speaker, current_speaker),
                "start": block_start,
                "text": current_text.strip(),
            })
            current_speaker = w["speaker"]
            current_text = w["word"]
            block_start = w["start"]
        else:
            current_text += w["word"]
        prev_end = w["end"]

    # Final block
    blocks.append({
        "speaker": speaker_map.get(current_speaker, current_speaker),
        "start": block_start,
        "text": current_text.strip(),
    })

    return blocks


def format_markdown(
    filename: str,
    blocks: list[dict],
    duration: float,
    language: str,
    num_speakers: int,
) -> str:
    """Format dialogue blocks as markdown."""
    lines = [
        f"# Transcript: {filename}",
        "",
        f"- **Date**: {date.today().isoformat()}",
        f"- **Duration**: {fmt_duration(duration)}",
        f"- **Language**: {language}",
        f"- **Speakers**: {num_speakers}",
        "",
        "---",
        "",
        "## Transcript",
        "",
    ]

    for block in blocks:
        ts = fmt_timestamp(block["start"])
        lines.append(f"**[{ts}] {block['speaker']}:**")
        lines.append(block["text"])
        lines.append("")

    return "\n".join(lines)


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
        # Try looking in the audio directory
        audio_file = audio_dir / input_path
    if not audio_file.exists():
        print(f"Error: Audio file not found: {input_path}")
        print(f"Place audio files in {audio_dir}/")
        sys.exit(1)

    audio_path = str(audio_file)
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

    # Step 5: Format and save
    markdown = format_markdown(
        filename=filename,
        blocks=blocks,
        duration=info.duration,
        language=info.language,
        num_speakers=len(unique_speakers),
    )

    output_dir.mkdir(exist_ok=True)
    stem = audio_file.stem
    output_path = output_dir / f"{stem}.md"
    output_path.write_text(markdown)

    print("=" * 50)
    print(f"Saved transcript to {output_path}")


if __name__ == "__main__":
    main()
