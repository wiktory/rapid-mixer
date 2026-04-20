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
    HPSS verzió: a percussive komponensen történik a beat tracking,
    mert ez általában stabilabb ritmikai pontokat ad.
    """
    y_mono = to_mono_for_analysis(y)
    _, y_percussive = librosa.effects.hpss(y_mono)
    _, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
    return librosa.frames_to_time(beat_frames, sr=sr)


def get_first_mix_beat(beat_times, min_start=1.0, beat_index=4):
    """
    Nem az első használható beatet vesszük,
    hanem egy kicsit későbbit, mert az intro eleje
    gyakran bizonytalanabb ritmikailag.
    """
    candidates = beat_times[beat_times >= min_start]
    if len(candidates) == 0:
        return min_start
    idx = min(beat_index, len(candidates) - 1)
    return float(candidates[idx])


def get_last_mix_beat(beat_times, track_duration, fade_seconds, beats_before_end=2):
    """
    Nem a legutolsó beatet használjuk kilépésre,
    hanem néhány beattel korábbit, hogy stabilabb
    legyen az átmeneti zóna.
    """
    target_time = max(track_duration - fade_seconds, 0)
    candidates = beat_times[beat_times <= target_time]
    if len(candidates) == 0:
        return target_time
    idx = max(0, len(candidates) - 1 - beats_before_end)
    return float(candidates[idx])


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

    # Equal-power fade a lineáris helyett
    fade_out = np.cos(np.linspace(0, np.pi / 2, fade_len))[np.newaxis, :]
    fade_in = np.sin(np.linspace(0, np.pi / 2, fade_len))[np.newaxis, :]

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

    # Biztonsági fade limit
    fade_seconds = min(float(fade_seconds), 15.0)

    total_steps = len(track_paths) * 3 + 1
    current_step = 0

    def update_progress():
        nonlocal current_step
        current_step += 1
        progress = int((current_step / total_steps) * 100)
        if progress_callback:
            progress_callback(min(progress, 100))

    outfile = sf.SoundFile(
        out_path,
        mode="w",
        samplerate=sr,
        channels=2,
        subtype="PCM_16"
    )

    # Ennyi maradjon RAM-ban a mix végéből
    tail_keep_seconds = 30
    tail_keep_samples = int(tail_keep_seconds * sr)

    try:
        # ----------------------------
        # ELSŐ TRACK
        # ----------------------------
        y_tail, _ = librosa.load(track_paths[0], sr=sr, mono=False)
        y_tail = ensure_stereo(y_tail)
        update_progress()

        bpm0 = float(track_bpms[0])
        rate0 = (target_bpm / bpm0) if bpm0 > 0 else 1.0

        if abs(rate0 - 1.0) > 0.04:
            y_tail = time_stretch_stereo(y_tail, rate=rate0)

        y_tail = rms_normalize(y_tail, target_rms)
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

            if abs(rate - 1.0) > 0.04:
                y = time_stretch_stereo(y, rate=rate)

            y = rms_normalize(y, target_rms)
            update_progress()

            # ----------------------------
            # BEAT ANALÍZIS A TAIL-EN
            # ----------------------------
            analysis_tail_seconds = 30
            analysis_tail_len = int(analysis_tail_seconds * sr)

            y_tail_analysis = (
                y_tail[:, -analysis_tail_len:]
                if y_tail.shape[1] > analysis_tail_len
                else y_tail
            )
            beat_times_a = get_beat_times(y_tail_analysis, sr)

            tail_offset = max((y_tail.shape[1] - y_tail_analysis.shape[1]) / sr, 0)
            beat_times_a = beat_times_a + tail_offset

            # ----------------------------
            # BEAT ANALÍZIS AZ ÚJ TRACK ELEJÉN
            # ----------------------------
            analysis_head_seconds = 20
            analysis_head_len = int(analysis_head_seconds * sr)

            y_head = y[:, :analysis_head_len] if y.shape[1] > analysis_head_len else y
            beat_times_b = get_beat_times(y_head, sr)

            a_duration = y_tail.shape[1] / sr
            a_exit_time = get_last_mix_beat(
                beat_times_a,
                a_duration,
                fade_seconds,
                beats_before_end=2
            )
            a_exit_sample = int(a_exit_time * sr)

            b_entry_time = get_first_mix_beat(
                beat_times_b,
                min_start=1.0,
                beat_index=4
            )
            b_start_sample = int(b_entry_time * sr)

            fade_len = int(fade_seconds * sr)
            cut_end = min(a_exit_sample + fade_len, y_tail.shape[1])
            cut_a = y_tail[:, :cut_end]

            combined = beat_aligned_crossfade(
                cut_a,
                y,
                sr,
                fade_seconds,
                b_start_sample
            )

            # ----------------------------
            # SPLIT: FILE + TAIL
            # ----------------------------
            if combined.shape[1] > tail_keep_samples:
                final_chunk = combined[:, :-tail_keep_samples]
                y_tail = combined[:, -tail_keep_samples:]
                outfile.write(final_chunk.T)
            else:
                y_tail = combined

            update_progress()

        # ----------------------------
        # VÉGSŐ TAIL KIÍRÁS
        # ----------------------------
        peak = np.max(np.abs(y_tail))
        if peak > 0.999:
            y_tail = y_tail / peak * 0.999

        outfile.write(y_tail.T)

        if progress_callback:
            progress_callback(100)

        return out_path

    finally:
        outfile.close()