"""Transcribe an audio file with speaker diarization to markdown."""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

SAMPLE_RATE = 16000  # Silero VAD requires 16kHz


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


class TranscriptionInfo:
    """Minimal info object matching the fields used from faster-whisper's TranscriptionInfo."""

    def __init__(self, language: str, duration: float):
        self.language = language
        self.duration = duration


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
    import torch

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

    if chunks:
        import torch
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


def load_diarization_pipeline():
    """Load and return the pyannote diarization pipeline for reuse."""
    import torch
    from pyannote.audio import Pipeline

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

    return pipeline


def diarize(audio_path: str, pipeline=None) -> list[dict]:
    """Run pyannote speaker diarization, return list of speaker segments."""
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    if pipeline is None:
        pipeline = load_diarization_pipeline()

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
    """Assign a speaker to each word by maximum timestamp overlap.

    Falls back to nearest speaker segment when no overlap exists,
    eliminating "Unknown" assignments from diarization gaps.
    """
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

        # Fall back to nearest segment if no overlap found
        if best_overlap == 0.0 and speaker_segments:
            word_mid = (word["start"] + word["end"]) / 2
            nearest = min(
                speaker_segments,
                key=lambda seg: min(abs(seg["start"] - word_mid), abs(seg["end"] - word_mid)),
            )
            best_speaker = nearest["speaker"]

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

        # Heuristic 1: Known hallucination phrases (standalone)
        if text.lower() in KNOWN_HALLUCINATIONS:
            remove[i] = True
            continue

        # Heuristic 2: Internal repetition — same short phrase repeated 5+ times
        words = text.split()
        if len(words) >= 5:
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


def load_openai_api_key() -> str | None:
    """Load OpenAI API key from environment or .env file."""
    token = os.environ.get("OPENAI_API_KEY")
    if token:
        return token

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip()

    return None


def load_google_api_key() -> str | None:
    """Load Google API key from environment or .env file."""
    token = os.environ.get("GOOGLE_API_KEY")
    if token:
        return token

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("GOOGLE_API_KEY="):
                return line.split("=", 1)[1].strip()

    return None


def infer_speaker_names(blocks: list[dict]) -> dict:
    """Identify speaker names from transcript context using LLM.

    Tries OpenAI (GPT-5.4) first, falls back to Gemini Flash.
    Returns a dict mapping speaker labels to suggestion dicts:
    {"Speaker 1": {"likely_name": "Sarah", "confidence": "high", "role": "...", "summary": "..."}}

    Returns empty dict if no API key or on failure.
    """
    # Build prompt (shared by both providers)
    snippet_blocks = blocks[:30]
    snippet_lines = []
    for b in snippet_blocks:
        snippet_lines.append(f"{b['speaker']}: {b['text']}")
    snippet = "\n".join(snippet_lines)

    speakers = sorted(set(b["speaker"] for b in blocks))

    prompt = f"""Analyze this transcript and identify the real names of each speaker.
Return ONLY valid JSON (no markdown fencing) with this structure:
{{
  "Speaker 1": {{
    "likely_name": "First Last" or null if unknown,
    "confidence": "high" | "medium" | "low",
    "role": "brief role description",
    "summary": "one sentence about what they said"
  }},
  ...
}}

Speakers to identify: {', '.join(speakers)}

Transcript:
{snippet}"""

    # Try OpenAI first
    openai_key = load_openai_api_key()
    if openai_key:
        try:
            result = _infer_with_openai(prompt, openai_key)
            if result:
                return result
        except Exception as e:
            print(f"  Warning: OpenAI speaker identification failed: {e}")

    # Fall back to Gemini
    google_key = load_google_api_key()
    if google_key:
        try:
            result = _infer_with_gemini(prompt, google_key)
            if result:
                return result
        except Exception as e:
            print(f"  Warning: Gemini speaker identification failed: {e}")

    # No keys available — prompt user
    print("\n  No OPENAI_API_KEY or GOOGLE_API_KEY found (needed for speaker identification).")
    print("  Set one in .env or environment variable, or enter one now.")
    key = input("  Enter your OpenAI API key [skip]: ").strip()
    if key and key.lower() != "skip":
        # Save to .env
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            content = env_path.read_text()
            if not content.endswith("\n"):
                content += "\n"
            content += f"OPENAI_API_KEY={key}\n"
        else:
            content = f"OPENAI_API_KEY={key}\n"
        env_path.write_text(content)
        print("  Saved to .env — you won't be asked again.")
        try:
            result = _infer_with_openai(prompt, key)
            if result:
                return result
        except Exception as e:
            print(f"  Warning: OpenAI speaker identification failed: {e}")

    return {}


def _infer_with_openai(prompt: str, api_key: str) -> dict:
    """Call OpenAI GPT-5.4 for speaker identification."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()
    return json.loads(text)


def _infer_with_gemini(prompt: str, api_key: str) -> dict:
    """Call Gemini Flash for speaker identification."""
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
    )
    text = response.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()
    return json.loads(text)


def polish_transcript(blocks: list[dict]) -> list[dict]:
    """Use LLM to fix transcription errors in block text.

    Fixes garbled names, technical terms, and obvious mistakes.
    Returns blocks with corrected text. Falls back to original on failure.
    """
    openai_key = load_openai_api_key()
    if not openai_key:
        google_key = load_google_api_key()
        if not google_key:
            return blocks

    # Build transcript for correction
    lines = []
    for i, b in enumerate(blocks):
        lines.append(f"[{i}] {b['text']}")
    transcript = "\n".join(lines)

    prompt = f"""Fix transcription errors in this interview transcript. Fix:
- Garbled or misspelled proper nouns (people, companies, products)
- Technical terms that were misheard
- Obviously wrong words from speech-to-text errors

Return ONLY valid JSON: a list of corrections as objects with "index" (block number) and "text" (corrected text).
If a block needs no changes, omit it. Return [] if nothing needs fixing.

Transcript:
{transcript}"""

    try:
        if openai_key:
            corrections = _infer_with_openai(prompt, openai_key)
        else:
            corrections = _infer_with_gemini(prompt, google_key)

        if isinstance(corrections, list):
            for fix in corrections:
                idx = fix.get("index")
                text = fix.get("text")
                if idx is not None and text and 0 <= idx < len(blocks):
                    blocks[idx]["text"] = text
            if corrections:
                print(f"  Polished {len(corrections)} blocks via LLM")
    except Exception as e:
        print(f"  Warning: LLM transcript polishing failed: {e}")

    return blocks


def generate_summary(blocks: list[dict]) -> str:
    """Use LLM to generate a brief interview summary.

    Returns a markdown-formatted summary string, or empty string on failure.
    """
    openai_key = load_openai_api_key()
    if not openai_key:
        google_key = load_google_api_key()
        if not google_key:
            return ""

    # Build transcript for summarization
    lines = []
    for b in blocks:
        lines.append(f"{b['speaker']}: {b['text']}")
    transcript = "\n".join(lines)

    prompt = f"""Summarize this interview transcript concisely. Include:
- Key topics discussed
- Important decisions or action items
- Notable quotes or insights

Keep it to 3-5 bullet points. Use plain text (no markdown formatting beyond bullet points).

Transcript:
{transcript}"""

    try:
        if openai_key:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-5.4",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        else:
            from google import genai
            client = genai.Client(api_key=google_key)
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
            )
            return response.text.strip()
    except Exception as e:
        print(f"  Warning: LLM summary generation failed: {e}")
        return ""


def confirm_speakers(suggestions: dict, blocks: list[dict]) -> dict[str, str]:
    """Interactive CLI flow to confirm or correct speaker name suggestions.

    Returns a mapping of original labels to confirmed names, e.g.:
    {"Speaker 1": "Sarah Chen", "Speaker 2": "Alex Kim"}
    """
    speakers = sorted(set(b["speaker"] for b in blocks))
    name_map = {}

    for i, speaker in enumerate(speakers, 1):
        info = suggestions.get(speaker, {})
        likely_name = info.get("likely_name")
        role = info.get("role", "")
        summary = info.get("summary", "")

        # Collect distinctive passages — longest blocks have the most substance
        speaker_blocks = [b for b in blocks if b["speaker"] == speaker]
        substantive = [b for b in speaker_blocks if len(b["text"].strip()) > 30]
        substantive.sort(key=lambda b: len(b["text"]), reverse=True)
        excerpts = []
        for b in substantive[:3]:
            text = b["text"].strip()
            if len(text) > 150:
                text = text[:147] + "..."
            excerpts.append(text)

        print(f"\n  Speaker {i} of {len(speakers)}:")

        if role:
            print(f"    Role: {role}")
        if summary:
            print(f"    Said: {summary}")
        if excerpts:
            print(f"    Passages:")
            for excerpt in excerpts:
                print(f"      > {excerpt}")

        if likely_name:
            print(f'    Suggested name: {likely_name}')
            answer = input(f'\n    Accept "{likely_name}"? [Y/name]: ').strip()
            if answer == "" or answer.lower() == "y":
                name_map[speaker] = likely_name
                print(f"    \u2713 {speaker} \u2192 {likely_name}")
            elif answer.lower() == "skip":
                print(f"    - Keeping \"{speaker}\"")
            else:
                name_map[speaker] = answer
                print(f"    \u2713 {speaker} \u2192 {answer}")
        else:
            answer = input(f"    Name for {speaker}? [skip]: ").strip()
            if answer and answer.lower() != "skip":
                name_map[speaker] = answer
                print(f"    \u2713 {speaker} \u2192 {answer}")
            else:
                print(f"    - Keeping \"{speaker}\"")

    return name_map


def identify_speakers(blocks: list[dict]) -> dict[str, str]:
    """Infer speaker names via LLM, then confirm interactively.

    Returns a mapping of original speaker labels to confirmed names.
    """
    print("\nIdentifying speakers...")
    suggestions = infer_speaker_names(blocks)

    if suggestions:
        print("  LLM provided speaker suggestions.")
    else:
        print("  No LLM suggestions available (no API key or call failed).")

    return confirm_speakers(suggestions, blocks)


def format_markdown(
    filename: str,
    blocks: list[dict],
    duration: float,
    language: str,
    num_speakers: int,
    speakers: list[str] | None = None,
    summary: str = "",
) -> str:
    """Format dialogue blocks as Obsidian-friendly markdown with YAML frontmatter."""
    stem = Path(filename).stem
    speaker_list = speakers or sorted(set(b["speaker"] for b in blocks))

    # YAML frontmatter
    lines = [
        "---",
        f"title: {stem}",
        f"date: {date.today().isoformat()}",
        f"source_file: {filename}",
        f"duration: {fmt_duration(duration)}",
        f"language: {language}",
        "speakers:",
    ]
    for s in speaker_list:
        lines.append(f"  - {s}")
    lines += [
        "tags:",
        "  - transcript",
        "  - interview",
        "---",
        "",
        f"# {stem}",
        "",
        "## Metadata",
        f"- **Duration**: {fmt_duration(duration)}",
        f"- **Language**: {language}",
        f"- **Speakers**: {', '.join(speaker_list)}",
        "",
        "---",
        "",
    ]

    if summary:
        lines += [
            "## Summary",
            "",
            summary,
            "",
            "---",
            "",
        ]

    lines += [
        "## Transcript",
        "",
    ]

    for block in blocks:
        ts = fmt_timestamp(block["start"])
        lines.append(f"**[{ts}] {block['speaker']}:**")
        lines.append(block["text"])
        lines.append("")

    return "\n".join(lines)


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma"}


def discover_audio_files(input_dir: Path) -> list[Path]:
    """Find all audio files in the input directory, sorted by name."""
    return [
        f for f in sorted(input_dir.iterdir())
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ]


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


DEFAULT_INPUT_DIR = Path(
    "/Users/jerrison/My Drive (jerrisonli@gmail.com)"
    "/00. Top Folder/02. Recruiting/04-interviews-audio-transcripts"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/jerrison/My Drive (jerrisonli@gmail.com)"
    "/00. Top Folder/04-obsidian-vaults/jerrison-personal-gdrive"
    "/00-Recruiting/03-transcripts"
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


if __name__ == "__main__":
    main()
