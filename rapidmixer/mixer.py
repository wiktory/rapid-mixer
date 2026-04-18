import numpy as np
import librosa
import soundfile as sf


def get_num_samples(y):
    if y.ndim == 1:
        return y.shape[0]
    return y.shape[1]


def ensure_stereo(y):
    if y.ndim == 1:
        return np.vstack([y, y])
    return y


def to_mono_for_analysis(y):
    if y.ndim == 1:
        return y
    return librosa.to_mono(y)


def get_beat_times(y, sr):
    """
    Beat pontok meghatározása mono analízis alapján.
    """
    y_mono = to_mono_for_analysis(y)
    _, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr)
    return librosa.frames_to_time(beat_frames, sr=sr)


def get_first_mix_beat(beat_times, min_start=1.0):
    candidates = beat_times[beat_times >= min_start]
    if len(candidates) == 0:
        return min_start
    return float(candidates[0])


def get_last_mix_beat(beat_times, track_duration, fade_seconds):
    target_time = max(track_duration - fade_seconds, 0)
    candidates = beat_times[beat_times <= target_time]
    if len(candidates) == 0:
        return target_time
    return float(candidates[-1])


def rms_normalize(y, target_rms=0.1):
    rms = np.sqrt(np.mean(y**2) + 1e-12)
    gain = target_rms / rms
    return y * gain


def time_stretch_stereo(y, rate):
    """
    Stereo kompatibilis time-stretch.
    """
    if y.ndim == 1:
        return librosa.effects.time_stretch(y, rate=rate)

    left = librosa.effects.time_stretch(y[0], rate=rate)
    right = librosa.effects.time_stretch(y[1], rate=rate)

    min_len = min(len(left), len(right))
    left = left[:min_len]
    right = right[:min_len]

    return np.vstack([left, right])


def beat_aligned_crossfade(a, b, sr, fade_seconds, b_start_sample):
    """
    Beathez igazított stereo crossfade.
    A és B shape-je: (channels, samples)
    """
    a = ensure_stereo(a)
    b = ensure_stereo(b)

    fade_len = int(fade_seconds * sr)

    if b_start_sample < 0:
        b_start_sample = 0

    b = b[:, b_start_sample:]

    fade_len = min(fade_len, a.shape[1], b.shape[1])

    if fade_len <= 0:
        raise ValueError("Nincs elég audio a fade-hez")

    a_main = a[:, :-fade_len]
    a_tail = a[:, -fade_len:]

    b_head = b[:, :fade_len]
    b_main = b[:, fade_len:]

    fade_out = np.linspace(1.0, 0.0, fade_len)[np.newaxis, :]
    fade_in = np.linspace(0.0, 1.0, fade_len)[np.newaxis, :]

    mixed = a_tail * fade_out + b_head * fade_in

    out = np.concatenate([a_main, mixed, b_main], axis=1)

    peak = np.max(np.abs(out))
    if peak > 0.999:
        out = out / peak * 0.999

    return out


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

    y_mix, _ = librosa.load(track_paths[0], sr=sr, mono=False)
    y_mix = ensure_stereo(y_mix)
    update_progress()

    bpm0 = float(track_bpms[0])
    rate0 = (target_bpm / bpm0) if bpm0 > 0 else 1.0

    # Kis eltérésnél nem stretch-elünk
    if abs(rate0 - 1.0) > 0.04:
        y_mix = time_stretch_stereo(y_mix, rate=rate0)

    y_mix = rms_normalize(y_mix, target_rms)
    update_progress()
    update_progress()

    # ----------------------------
    # TÖBBI TRACK
    # ----------------------------

    for i, path in enumerate(track_paths[1:], start=1):
        y, _ = librosa.load(path, sr=sr, mono=False)
        y = ensure_stereo(y)
        update_progress()

        bpm = float(track_bpms[i])
        rate = (target_bpm / bpm) if bpm > 0 else 1.0

        # Kis eltérésnél nem stretch-elünk
        if abs(rate - 1.0) > 0.04:
            y = time_stretch_stereo(y, rate=rate)

        y = rms_normalize(y, target_rms)
        update_progress()

        # ----------------------------
        # GYORSÍTOTT BEAT ANALÍZIS
        # ----------------------------

        # Az aktuális mixből csak a végét elemezzük
        analysis_tail_seconds = 30
        tail_len = int(analysis_tail_seconds * sr)

        y_mix_tail = y_mix[:, -tail_len:] if y_mix.shape[1] > tail_len else y_mix
        beat_times_a = get_beat_times(y_mix_tail, sr)

        tail_offset = max((y_mix.shape[1] - y_mix_tail.shape[1]) / sr, 0)
        beat_times_a = beat_times_a + tail_offset

        # Az új trackből csak az elejét elemezzük
        analysis_head_seconds = 20
        head_len = int(analysis_head_seconds * sr)

        y_head = y[:, :head_len] if y.shape[1] > head_len else y
        beat_times_b = get_beat_times(y_head, sr)

        a_duration = y_mix.shape[1] / sr
        a_exit_time = get_last_mix_beat(beat_times_a, a_duration, fade_seconds)
        a_exit_sample = int(a_exit_time * sr)

        b_entry_time = get_first_mix_beat(beat_times_b, min_start=1.0)
        b_start_sample = int(b_entry_time * sr)

        fade_len = int(fade_seconds * sr)
        cut_end = min(a_exit_sample + fade_len, y_mix.shape[1])
        cut_a = y_mix[:, :cut_end]

        y_mix = beat_aligned_crossfade(
            cut_a,
            y,
            sr,
            fade_seconds,
            b_start_sample
        )

        update_progress()

    # ----------------------------
    # VÉGSŐ PEAK LIMIT
    # ----------------------------

    peak = np.max(np.abs(y_mix))
    if peak > 0.999:
        y_mix = y_mix / peak * 0.999

    # soundfile stereo mentéshez (samples, channels) alak kell
    sf.write(out_path, y_mix.T, sr)

    if progress_callback:
        progress_callback(100)

    return out_path