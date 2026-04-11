import numpy as np
import librosa
import soundfile as sf

# ----------------------------
# DSP segédek
# ----------------------------

def estimate_bpm(y, sr):
    """
    BPM becslés.
    Visszaad: (tempo, beat_times)
    """
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return float(tempo), beat_times


def rms_normalize(y, target_rms=0.1):
    """
    Egyszerű RMS normalizálás.
    """
    rms = np.sqrt(np.mean(y**2) + 1e-12)
    gain = target_rms / rms
    return y * gain


def crossfade(a, b, sr, fade_seconds=8.0):
    """
    Két jel összefűzése fade-del.
    - a végéből és b elejéből vesz fade_len mintát
    - lineáris fade (később lehet cosine)
    """
    fade_len = int(fade_seconds * sr)
    fade_len = min(fade_len, len(a), len(b))

    a_main = a[:-fade_len]
    a_tail = a[-fade_len:]

    b_head = b[:fade_len]
    b_main = b[fade_len:]

    fade_out = np.linspace(1.0, 0.0, fade_len)
    fade_in  = np.linspace(0.0, 1.0, fade_len)

    mixed = a_tail * fade_out + b_head * fade_in
    return np.concatenate([a_main, mixed, b_main])


def safe_trim_for_fade(y, sr, fade_seconds, min_after_fade_seconds=2.0):
    """
    Biztonság: ha a track túl rövid, ne omoljon össze a mix.
    """
    fade_len = int(fade_seconds * sr)
    min_len = fade_len + int(min_after_fade_seconds * sr)
    if len(y) < min_len:
        raise ValueError("A track túl rövid a megadott fade-hez.")
    return y


# ----------------------------
# Fő pipeline: N track mix
# ----------------------------

def mix_tracklist_to_target_bpm(
    track_paths: list[str],
    track_bpms,
    target_bpm: int,
    fade_seconds: float,
    out_path: str = "mixed.wav",
    sr: int = 22050,
    target_rms: float = 0.1,
    progress_callback=None, #folyamtjelző függvény hosszú ideig futó folyamatokhoz
):
    if not track_paths or len(track_paths) < 2:
        raise ValueError("Legalább 2 track kell a mixhez.")

    total_steps = len(track_paths) * 3 + 1
    current_step = 0

    def update_progress():
        nonlocal current_step
        current_step += 1
        progress = int((current_step / total_steps) * 100)
        if progress_callback:
            progress_callback(min(progress, 100))

    y_mix, _ = librosa.load(track_paths[0], sr=sr, mono=True)
    update_progress()

    #bpm0, _ = estimate_bpm(y_mix, sr) - ez kellett a BPM számításhoz, de nagyon lassú
    bpm0 = float(track_bpms[0])
    rate0 = (target_bpm / bpm0) if bpm0 > 0 else 1.0
    y_mix = librosa.effects.time_stretch(y_mix, rate=rate0)
    y_mix = rms_normalize(y_mix, target_rms=target_rms)
    y_mix = safe_trim_for_fade(y_mix, sr, fade_seconds)
    update_progress()

    for i, path in enumerate(track_paths[1:], start=1):
        y, _ = librosa.load(path, sr=sr, mono=True)
        update_progress()

        #bpm, _ = estimate_bpm(y, sr)
        bpm = float(track_bpms[i])
        rate = (target_bpm / bpm) if bpm > 0 else 1.0
        y = librosa.effects.time_stretch(y, rate=rate)
        y = rms_normalize(y, target_rms=target_rms)
        y = safe_trim_for_fade(y, sr, fade_seconds)
        update_progress()

        y_mix = crossfade(y_mix, y, sr, fade_seconds=fade_seconds)
        update_progress()

    sf.write(out_path, y_mix, sr)

    if progress_callback:
        progress_callback(100)

    return out_path