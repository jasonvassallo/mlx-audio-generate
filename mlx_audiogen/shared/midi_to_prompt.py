"""MIDI-to-prompt: analyze MIDI data and generate a text description.

Reads MIDI file bytes and produces a natural language description
suitable as a generation prompt. Analyzes note range, density,
rhythm patterns, and key signature.
"""

import struct

# Note names
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Common key profiles (Krumhansl-Kessler)
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def midi_to_prompt(midi_bytes: bytes) -> str:
    """Convert MIDI file bytes to a descriptive prompt.

    Args:
        midi_bytes: Standard MIDI File bytes.

    Returns:
        Natural language description of the MIDI content.
    """
    notes = _parse_midi_notes(midi_bytes)

    if not notes:
        return "a simple musical passage"

    # Analyze
    pitches = [n[1] for n in notes]
    velocities = [n[2] for n in notes]
    times = [n[0] for n in notes]

    min_pitch = min(pitches)
    max_pitch = max(pitches)
    avg_velocity = sum(velocities) / len(velocities)

    # Duration
    total_duration = max(times) - min(times) if len(times) > 1 else 1.0

    # Note density (notes per second)
    density = len(notes) / max(total_duration, 0.1)

    # Pitch range description
    range_desc = _describe_range(min_pitch, max_pitch)

    # Velocity description
    vel_desc = _describe_velocity(avg_velocity)

    # Estimate key
    key = _estimate_key(pitches)

    # Density description
    if density > 8:
        tempo_desc = "fast, dense"
    elif density > 4:
        tempo_desc = "moderate tempo"
    elif density > 2:
        tempo_desc = "slow, spacious"
    else:
        tempo_desc = "very sparse"

    # Build prompt
    parts = []
    parts.append(f"{vel_desc} {range_desc} musical passage")
    parts.append(f"in {key}")
    parts.append(tempo_desc)

    if max_pitch - min_pitch <= 12:
        parts.append("with narrow melodic range")
    elif max_pitch - min_pitch >= 36:
        parts.append("with wide melodic range spanning multiple octaves")

    return ", ".join(parts)


def _parse_midi_notes(
    midi_bytes: bytes,
) -> list[tuple[float, int, int]]:
    """Parse MIDI bytes and extract note events.

    Returns list of (time_seconds, pitch, velocity).
    """
    if len(midi_bytes) < 14 or midi_bytes[:4] != b"MThd":
        return []

    # Read header
    _fmt, _ntracks, tpb = struct.unpack(">HHh", midi_bytes[8:14])
    if tpb <= 0:
        tpb = 480

    # Find first track
    pos = 14
    notes = []
    current_tick = 0
    tempo = 500000  # Default 120 BPM

    while pos < len(midi_bytes) - 8:
        if midi_bytes[pos : pos + 4] == b"MTrk":
            track_len = struct.unpack(">I", midi_bytes[pos + 4 : pos + 8])[0]
            track_end = pos + 8 + track_len
            pos += 8

            while pos < track_end:
                # Read variable-length delta
                delta = 0
                while pos < track_end:
                    b = midi_bytes[pos]
                    pos += 1
                    delta = (delta << 7) | (b & 0x7F)
                    if not (b & 0x80):
                        break

                current_tick += delta

                if pos >= track_end:
                    break

                status = midi_bytes[pos]

                # Meta event
                if status == 0xFF:
                    pos += 1
                    if pos >= track_end:
                        break
                    meta_type = midi_bytes[pos]
                    pos += 1
                    # Read length
                    meta_len = 0
                    while pos < track_end:
                        b = midi_bytes[pos]
                        pos += 1
                        meta_len = (meta_len << 7) | (b & 0x7F)
                        if not (b & 0x80):
                            break

                    if meta_type == 0x51 and meta_len == 3 and pos + 3 <= track_end:
                        tempo = (
                            midi_bytes[pos] << 16
                            | midi_bytes[pos + 1] << 8
                            | midi_bytes[pos + 2]
                        )
                    pos += meta_len
                    continue

                # Note on (0x90-0x9F)
                if (status & 0xF0) == 0x90 and pos + 2 <= track_end:
                    pos += 1
                    pitch = midi_bytes[pos]
                    pos += 1
                    velocity = midi_bytes[pos]
                    pos += 1
                    if velocity > 0:
                        time_s = current_tick * tempo / (tpb * 1_000_000)
                        notes.append((time_s, pitch, velocity))
                    continue

                # Note off or other 2-byte messages
                if (status & 0xF0) in (0x80, 0xA0, 0xB0, 0xE0):
                    pos += 3
                    continue

                # Program change, channel pressure (1-byte)
                if (status & 0xF0) in (0xC0, 0xD0):
                    pos += 2
                    continue

                # Running status or unknown — skip
                pos += 1

            break  # Only process first track
        else:
            pos += 1

    return notes


def _describe_range(min_pitch: int, max_pitch: int) -> str:
    """Describe the pitch range."""
    avg = (min_pitch + max_pitch) / 2
    if avg < 48:
        return "deep bass"
    elif avg < 60:
        return "bass"
    elif avg < 72:
        return "mid-range"
    elif avg < 84:
        return "high"
    else:
        return "very high, bright"


def _describe_velocity(avg_vel: float) -> str:
    """Describe the average velocity."""
    if avg_vel > 110:
        return "aggressive, loud"
    elif avg_vel > 85:
        return "strong"
    elif avg_vel > 60:
        return "moderate"
    elif avg_vel > 35:
        return "gentle, soft"
    else:
        return "very quiet, delicate"


def _estimate_key(pitches: list[int]) -> str:
    """Estimate the musical key using pitch class histogram."""
    if not pitches:
        return "C major"

    # Build pitch class histogram
    histogram = [0.0] * 12
    for p in pitches:
        histogram[p % 12] += 1

    total = sum(histogram)
    if total == 0:
        return "C major"

    histogram = [h / total for h in histogram]

    # Correlate with major and minor profiles for each root
    best_key = "C major"
    best_corr = -1.0

    for root in range(12):
        rotated = histogram[root:] + histogram[:root]

        # Major correlation
        corr = sum(a * b for a, b in zip(rotated, MAJOR_PROFILE))
        if corr > best_corr:
            best_corr = corr
            best_key = f"{NOTE_NAMES[root]} major"

        # Minor correlation
        corr = sum(a * b for a, b in zip(rotated, MINOR_PROFILE))
        if corr > best_corr:
            best_corr = corr
            best_key = f"{NOTE_NAMES[root]} minor"

    return best_key
