from django.shortcuts import render, redirect
from django.db.models import Q
from django.http import JsonResponse, FileResponse
from django.utils import timezone
from django.conf import settings

from .models import Music, MixGeneration
from .mixer import mix_tracklist_to_target_bpm

import os
import threading
import uuid
import time


def index(request):
    musics = Music.objects.all().order_by("performer", "title")

    playlist_ids = request.session.get("playlist", [])
    playlist_qs = Music.objects.filter(id__in=playlist_ids)
    music_map = {music.id: music for music in playlist_qs}
    playlist = [music_map[m_id] for m_id in playlist_ids if m_id in music_map]

    track = request.session.get("track", None)
    query = request.GET.get("music_search", "")
    playlist_modal_message = request.session.pop("playlist_modal_message", None)

    if query:
        tracks = Music.objects.filter(
            Q(performer__icontains=query) | Q(title__icontains=query)
        )

        context = {
            "musics": musics,
            "tracks": tracks,
            "playlist": playlist,
            "query": query,
            "track": track,
            "playlist_modal_message": playlist_modal_message,
        }
    else:
        context = {
            "playlist": playlist,
            "musics": musics,
            "track": track,
            "playlist_modal_message": playlist_modal_message,
        }

    return render(request, "rapidmixer/main.html", context)


def add_to_playlist(request, id):
    playlist = [int(x) for x in request.session.get("playlist", [])]
    id = int(id)

    if id in playlist:
        request.session["playlist_modal_message"] = "Ez a zene már szerepel a playlistben."
    elif len(playlist) >= 5:
        request.session["playlist_modal_message"] = "Elérted a maximum zeneszámot."
    else:
        playlist.append(id)
        request.session["playlist"] = playlist

    request.session.modified = True

    query = request.GET.get("q", "")
    if query:
        return redirect(f"/?music_search={query}")

    return redirect("index")


def update_playlist_order(request):
    order = request.GET.get("order", "")

    if not order:
        return JsonResponse(
            {"status": "error", "message": "Nem érkezett sorrend."},
            status=400,
        )

    try:
        new_order = [int(x) for x in order.split(",") if x.strip()]
    except ValueError:
        return JsonResponse(
            {"status": "error", "message": "Érvénytelen sorrend."},
            status=400,
        )

    current_playlist = [int(x) for x in request.session.get("playlist", [])]

    if sorted(new_order) != sorted(current_playlist):
        return JsonResponse(
            {"status": "error", "message": "A playlist elemei nem egyeznek."},
            status=400,
        )

    request.session["playlist"] = new_order
    request.session.modified = True

    return JsonResponse(
        {"status": "success", "message": "A playlist sorrendje frissítve."}
    )


def delete_from_playlist(request, id):
    playlist = [int(x) for x in request.session.get("playlist", [])]
    id = int(id)

    if id in playlist:
        playlist.remove(id)

    request.session["playlist"] = playlist
    request.session.modified = True

    query = request.GET.get("q", "")
    if query:
        return redirect(f"/?music_search={query}")

    return redirect("index")


def delete_all_from_playlist(request):
    request.session["playlist"] = []
    request.session.modified = True

    query = request.GET.get("q", "")
    if query:
        return redirect(f"/?music_search={query}")

    return redirect("index")


def start_mix(request, bpm, fade):
    playlist_ids = request.session.get("playlist", [])

    if len(playlist_ids) < 2:
        return JsonResponse({"error": "Minimum 2 zenét kell tartalmaznia a playlist-nek, hogy elindítsd a mix generálást!"}, status=400)

    musics = Music.objects.filter(id__in=playlist_ids)
    music_map = {music.id: music for music in musics}
    ordered_musics = [music_map[m_id] for m_id in playlist_ids if m_id in music_map]

    track_paths = [
        os.path.join(settings.BASE_DIR, "rapidmixer", "static", music.path)
        for music in ordered_musics
    ]

    track_bpms = [float(music.bpm) for music in ordered_musics]

    job_id = str(uuid.uuid4())

    temp_dir = os.path.join(settings.BASE_DIR, "temp_mixes")
    os.makedirs(temp_dir, exist_ok=True)
    out_path = os.path.join(temp_dir, f"{job_id}.wav")

    mix_record = MixGeneration.objects.create(
        job_id=job_id,
        status="processing",
        progress=0,
        bpm=int(bpm),
        fade=int(fade),
        track_count=len(ordered_musics),
        playlist_snapshot=",".join(str(m.id) for m in ordered_musics),
        output_filename=os.path.basename(out_path),
    )

    def set_progress(value):
        MixGeneration.objects.filter(job_id=job_id).update(
            progress=int(value),
            status="processing",
        )

    def worker():
        start_time = time.time()

        try:
            mix_tracklist_to_target_bpm(
                track_paths=track_paths,
                track_bpms=track_bpms,
                target_bpm=int(bpm),
                fade_seconds=float(fade),
                out_path=out_path,
                progress_callback=set_progress,
            )

            elapsed_seconds = time.time() - start_time

            mix_record = MixGeneration.objects.get(job_id=job_id)
            mix_record.status = "done"
            mix_record.progress = 100
            mix_record.duration_seconds = elapsed_seconds
            mix_record.save()

        except Exception as e:
            mix_record = MixGeneration.objects.get(job_id=job_id)
            mix_record.status = "error"
            mix_record.progress = 0
            mix_record.error_message = str(e)
            mix_record.save()

    threading.Thread(target=worker, daemon=True).start()

    return JsonResponse({"job_id": job_id})


def mix_progress(request, job_id):
    try:
        mix_record = MixGeneration.objects.get(job_id=job_id)
    except MixGeneration.DoesNotExist:
        return JsonResponse({"error": "Nincs ilyen feldolgozás."}, status=404)

    response = {
        "progress": mix_record.progress,
        "status": mix_record.status,
    }

    if mix_record.status == "done":
        response["progress"] = 100
        response["download_url"] = f"/playlist/download_mix/{job_id}/"

    if mix_record.status == "error":
        response["error"] = mix_record.error_message or "Ismeretlen hiba"

    return JsonResponse(response)


def download_mix(request, job_id):
    try:
        mix_record = MixGeneration.objects.get(job_id=job_id)
    except MixGeneration.DoesNotExist:
        return JsonResponse({"error": "Nincs ilyen feldolgozás."}, status=404)

    if mix_record.status not in ["done", "downloaded"]:
        return JsonResponse({"error": "A mix még nem tölthető le."}, status=400)

    file_path = os.path.join(
        settings.BASE_DIR,
        "temp_mixes",
        mix_record.output_filename,
    )

    if not os.path.exists(file_path):
        return JsonResponse({"error": "Fájl nem található."}, status=404)

    mix_record.status = "downloaded"
    mix_record.downloaded_at = timezone.now()
    mix_record.save()

    file_handle = open(file_path, "rb")

    response = FileResponse(
        file_handle,
        as_attachment=True,
        filename=f"mix_{job_id}.wav",
    )

    original_close = response.close

    def custom_close():
        try:
            file_handle.close()
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

        original_close()

    response.close = custom_close

    return response