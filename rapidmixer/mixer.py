import numpy as np
import librosa
import soundfile as sf


# ----------------------------
# SEGÉDFÜGGVÉNYEK
# ----------------------------

def get_beat_times(y, sr):
    """
    Beat pontok meghatározása (időben, másodpercben)
    """
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    return librosa.frames_to_time(beat_frames, sr=sr)


def get_first_mix_beat(beat_times, min_start=1.0):
    """
    Az első használható beat (ne a track elején)
    """
    candidates = beat_times[beat_times >= min_start]
    if len(candidates) == 0:
        return min_start
    return float(candidates[0])


def get_last_mix_beat(beat_times, track_duration, fade_seconds):
    """
    Az utolsó beat, ami még alkalmas fade-re
    """
    target_time = max(track_duration - fade_seconds, 0)
    candidates = beat_times[beat_times <= target_time]
    if len(candidates) == 0:
        return target_time
    return float(candidates[-1])


def rms_normalize(y, target_rms=0.1):
    rms = np.sqrt(np.mean(y**2) + 1e-12)
    gain = target_rms / rms
    return y * gain


def beat_aligned_crossfade(a, b, sr, fade_seconds, b_start_sample):
    """
    Beathez igazított crossfade
    """
    fade_len = int(fade_seconds * sr)

    if b_start_sample < 0:
        b_start_sample = 0

    b = b[b_start_sample:]

    fade_len = min(fade_len, len(a), len(b))

    if fade_len <= 0:
        raise ValueError("Nincs elég audio a fade-hez")

    a_main = a[:-fade_len]
    a_tail = a[-fade_len:]

    b_head = b[:fade_len]
    b_main = b[fade_len:]

    fade_out = np.linspace(1.0, 0.0, fade_len)
    fade_in = np.linspace(0.0, 1.0, fade_len)

    mixed = a_tail * fade_out + b_head * fade_in
    return np.concatenate([a_main, mixed, b_main])


# ----------------------------
# FŐ MIXER
# ----------------------------

def mix_tracklist_to_target_bpm(
    track_paths,
    track_bpms,
    target_bpm,
    fade_seconds,
    out_path="mixed.wav",
    sr=22050,
    target_rms=0.1,
    progress_callback=None,
):
    if not track_paths or len(track_paths) < 2:
        raise ValueError("Legalább 2 track kell")

    total_steps = len(track_paths) * 3 + 1
    current_step = 0

    def update_progress():
        nonlocal current_step
        current_step += 1
        progress = int((current_step / total_steps) * 100)
        if progress_callback:
            progress_callback(min(progress, 100))

    # ----------------------------
    # ELSŐ TRACK
    # ----------------------------

    y_mix, _ = librosa.load(track_paths[0], sr=sr, mono=True)
    update_progress()

    bpm0 = float(track_bpms[0])
    rate0 = (target_bpm / bpm0) if bpm0 > 0 else 1.0

    y_mix = librosa.effects.time_stretch(y_mix, rate=rate0)
    y_mix = rms_normalize(y_mix, target_rms)
    update_progress()

    update_progress()

    # ----------------------------
    # TÖBBI TRACK
    # ----------------------------

    for i, path in enumerate(track_paths[1:], start=1):

        # --- betöltés ---
        y, _ = librosa.load(path, sr=sr, mono=True)
        update_progress()

        # --- BPM igazítás ---
        bpm = float(track_bpms[i])
        rate = (target_bpm / bpm) if bpm > 0 else 1.0

        y = librosa.effects.time_stretch(y, rate=rate)
        y = rms_normalize(y, target_rms)
        update_progress()

        # ----------------------------
        # BEAT ANALÍZIS (CSAK EZ!)
        # ----------------------------

        beat_times_a = get_beat_times(y_mix, sr)
        beat_times_b = get_beat_times(y, sr)

        # --- A track vége ---
        a_duration = len(y_mix) / sr
        a_exit_time = get_last_mix_beat(beat_times_a, a_duration, fade_seconds)
        a_exit_sample = int(a_exit_time * sr)

        # --- B track eleje ---
        b_entry_time = get_first_mix_beat(beat_times_b, min_start=1.0)
        b_start_sample = int(b_entry_time * sr)

        # --- A levágása ---
        fade_len = int(fade_seconds * sr)
        cut_a = y_mix[:a_exit_sample + fade_len]

        # --- CROSSFADE ---
        y_mix = beat_aligned_crossfade(
            cut_a,
            y,
            sr,
            fade_seconds,
            b_start_sample
        )

        update_progress()

    # ----------------------------
    # MENTÉS
    # ----------------------------

    sf.write(out_path, y_mix, sr)

    if progress_callback:
        progress_callback(100)

    return out_path