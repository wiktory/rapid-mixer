from django.shortcuts import render, redirect
from .models import Music # Music model importállása
from django.db.models import Q #query kezeléshez
from .models import MixGeneration
import librosa


import os
import tempfile
import threading
import uuid
import time
from django.utils import timezone
from django.conf import settings
from django.http import JsonResponse, FileResponse
from .mixer import mix_tracklist_to_target_bpm


# from django.http import JsonResponse --> ez a JS megoldáshoz kellene




def index(request): # ezt nézzem majd át - kell a track? kell ennyi mindent átadni context-nek?
    
    musics = Music.objects.all().order_by('performer', 'title')

    playlist_ids = request.session.get("playlist", [])
    playlist = Music.objects.filter(id__in=playlist_ids)

    track = request.session.get("track", None)
    
    query = request.GET.get("music_search", "")

    playlist_modal_message = request.session.pop("playlist_modal_message", None)
    #playlist_count = len(playlist_ids)   

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
            #"playlist_count": playlist_count,
        }
    else:
        # tracks és queryt nem adjuk át, vagy üres a lista, tehát nincs keresési eredmény

        context = {
            "playlist": playlist,
            "musics": musics,
            "track": track,
            "playlist_modal_message": playlist_modal_message,
            #"playlist_count": playlist_count,
        }

    return render(request, "rapidmixer/main.html", context)

# ez a session megoldás, hogy tároljuk a tracklistet a session-ben, hogy ne vesszen el. Böngészőbezárásig él

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
        #request.session["playlist_modal_message"] = "A zene hozzáadva a playlisthez."

    request.session.modified = True

    query = request.GET.get("q", "")
    if query:
        return redirect(f"/?music_search={query}")

    return redirect("index")

from django.http import JsonResponse

def update_playlist_order(request):
    print("order update");
    order = request.GET.get("order", "")

    if not order:
        return JsonResponse({
            "status": "error",
            "message": "Nem érkezett sorrend."
        }, status=400)

    try:
        new_order = [int(x) for x in order.split(",") if x.strip()]
    except ValueError:
        return JsonResponse({
            "status": "error",
            "message": "Érvénytelen sorrend."
        }, status=400)

    current_playlist = [int(x) for x in request.session.get("playlist", [])]

    # Biztonsági ellenőrzés: ugyanazok az elemek legyenek, csak más sorrendben
    if sorted(new_order) != sorted(current_playlist):
        return JsonResponse({
            "status": "error",
            "message": "A playlist elemei nem egyeznek."
        }, status=400)

    request.session["playlist"] = new_order
    request.session.modified = True

    return JsonResponse({
        "status": "success",
        "message": "A playlist sorrendje frissítve."
    })

def delete_from_playlist(request, id):
    playlist = request.session.get("playlist", [])

    if id in playlist:
        playlist.remove(id)

    request.session["playlist"] = playlist

    query = request.GET.get("q", "")
    if query:
        return redirect(f"/?music_search={query}")

    return redirect("index")

def delete_all_from_playlist(request):

    request.session["playlist"] = []

    query = request.GET.get("q", "")
    if query:
        return redirect(f"/?music_search={query}")

    return redirect("index")


def start_mix(request, bpm, fade):
    playlist_ids = request.session.get("playlist", [])

    if len(playlist_ids) < 2:
        return JsonResponse({"error": "Legalább 2 zene kell."}, status=400)

    musics = Music.objects.filter(id__in=playlist_ids)
    music_map = {music.id: music for music in musics}
    ordered_musics = [music_map[m_id] for m_id in playlist_ids if m_id in music_map]

    track_paths = [
        os.path.join(settings.BASE_DIR, "rapidmixer", "static", music.path)
        for music in ordered_musics
    ]

    print (track_paths);

    job_id = str(uuid.uuid4())
    
    temp_dir = os.path.join(settings.BASE_DIR, "temp_mixes")
    os.makedirs(temp_dir, exist_ok=True)
    out_path = os.path.join(temp_dir, f"{job_id}.wav")

    MixGeneration.objects.create(
        job_id=job_id,
        status="processing",
        bpm=int(bpm),
        fade=int(fade),
        track_count=len(ordered_musics),
        playlist_snapshot=",".join(str(m.id) for m in ordered_musics),
        output_filename=os.path.basename(out_path),
    )

    # session kezdeti állapot
    request.session[f"mix_status_{job_id}"] = {
        "progress": 0,
        "status": "processing",
        "file_path": out_path,
    }
    request.session.modified = True

    def set_progress(value):
        # FONTOS: szálból a session írás nem ideális productionben,
        # de prototípushoz elmehet. Később Redis/DB jobb.
        from django.contrib.sessions.models import Session
        session_key = request.session.session_key
        session = Session.objects.get(session_key=session_key)
        data = session.get_decoded()
        status = data.get(f"mix_status_{job_id}", {})
        status["progress"] = value
        data[f"mix_status_{job_id}"] = status
        session.session_data = Session.objects.encode(data)
        session.save()

    def worker():
        
        start_time = time.time()

        try:
            mix_tracklist_to_target_bpm(
                track_paths=track_paths,
                target_bpm=int(bpm),
                fade_seconds=float(fade),
                out_path=out_path,
                progress_callback=set_progress
            )

            from django.contrib.sessions.models import Session
            session_key = request.session.session_key
            session = Session.objects.get(session_key=session_key)
            data = session.get_decoded()
            data[f"mix_status_{job_id}"] = {
                "progress": 100,
                "status": "done",
                "file_path": out_path,
            }
            session.session_data = Session.objects.encode(data)
            session.save()

            elapsed_seconds = time.time() - start_time

            mix_record = MixGeneration.objects.get(job_id=job_id)
            mix_record.status = "done"
            mix_record.duration_seconds = elapsed_seconds
            mix_record.save()            

        except Exception as e:
            from django.contrib.sessions.models import Session
            session_key = request.session.session_key
            session = Session.objects.get(session_key=session_key)
            data = session.get_decoded()
            data[f"mix_status_{job_id}"] = {
                "progress": 0,
                "status": "error",
                "error": str(e),
            }
            session.session_data = Session.objects.encode(data)
            session.save()

            mix_record = MixGeneration.objects.get(job_id=job_id)
            mix_record.status = "error"
            mix_record.error_message = str(e)
            mix_record.save()            

    threading.Thread(target=worker, daemon=True).start()

    return JsonResponse({"job_id": job_id})

def mix_progress(request, job_id):
    data = request.session.get(f"mix_status_{job_id}")

    if not data:
        return JsonResponse({"error": "Nincs ilyen feldolgozás."}, status=404)

    response = {
        "progress": data.get("progress", 0),
        "status": data.get("status", "processing"),
    }

    if data.get("status") == "done":
        response["download_url"] = f"/playlist/download_mix/{job_id}/"

    if data.get("status") == "error":
        response["error"] = data.get("error", "Ismeretlen hiba")

    return JsonResponse(response)

def download_mix(request, job_id):

    data = request.session.get(f"mix_status_{job_id}")

    if not data:
        return JsonResponse({"error": "Nincs ilyen feldolgozás."}, status=404)

    file_path = data.get("file_path")

    if not os.path.exists(file_path):
        return JsonResponse({"error": "Fájl nem található."}, status=404)
    
    # DB rekord frissítése letöltéskor
    mix_record = MixGeneration.objects.get(job_id=job_id)
    mix_record.status = "downloaded"
    mix_record.downloaded_at = timezone.now()
    mix_record.save()    

    file_handle = open(file_path, "rb")

    response = FileResponse(
        file_handle,
        as_attachment=True,
        filename=f"mix_{job_id}.wav"
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











'''
def mix_playlist_musics(request, bpm, fade):
    # A session-ben a "playlist" kulcs alatt tárolod a zene ID-kat
    playlist_ids = request.session.get("playlist", [])

    # Ellenőrizzük, hogy legalább 2 zene van-e
    if len(playlist_ids) < 2:
        return JsonResponse(
            {"error": "Legalább 2 zene kell a mix elkészítéséhez."},
            status=400
        )

    # Lekérjük az adatbázisból az összes érintett Music objektumot
    musics = Music.objects.filter(id__in=playlist_ids)

    # Készítünk egy szótárat, hogy ID alapján gyorsan elérjük az objektumokat
    music_map = {music.id: music for music in musics}

    # Visszaállítjuk az eredeti playlist sorrendet
    ordered_musics = [
        music_map[music_id]
        for music_id in playlist_ids
        if music_id in music_map
    ]

    # Itt kell megadni azt a mezőt, amelyik a hangfájlt tárolja.
    # Példa: ha a modelben FileField neve "file", akkor music.file.path
    track_paths = [
        os.path.join(settings.BASE_DIR, "rapidmixer", "static", music.path)
        for music in ordered_musics
    ]


    print(track_paths)

    # Ha nálad nem "file" a mező neve, hanem pl. "audio_file" vagy "music_file",
    # akkor ezt a sort annak megfelelően módosítsd.

    # Kimeneti fájl neve
    out_name = f"mix_{bpm}_{fade}.wav"
    out_dir = os.path.join(settings.MEDIA_ROOT, "mixes")
    out_path = os.path.join(out_dir, out_name)

    # Létrehozzuk a mappát, ha még nem létezik
    os.makedirs(out_dir, exist_ok=True)

    # Meghívjuk a mixelő függvényt
    mix_tracklist_to_target_bpm(
        track_paths=track_paths,
        target_bpm=int(bpm),
        fade_seconds=float(fade),
        out_path=out_path
    )

    # Visszaadjuk az elkészült fájl elérési útját
    return JsonResponse({
        "ok": True,
        "file": f"{settings.MEDIA_URL}mixes/{out_name}"
    })

'''

'''
def mix_playlist_musics(request, bpm, fade):
    print(bpm, fade)
    
    test_tracks = [
        "/assets/audio/Eminem-Without_Me.mp3",
        "/assets/audio/Eminem-Lose_Yourself.mp3",
    ]

    print(test_tracks)
    


    musics = Music.objects.all()

    playlist_ids = request.session.get("playlist", [])
    playlist = Music.objects.filter(id__in=playlist_ids)

    for zene in playlist:
        print(zene.path)

    #playlist = request.session.get("playlist", [])

    return redirect("index")
'''














'''
def audio_play(request, id):
    playlist = request.session.get("playlist", [])
    track = Music.objects.get(id=id)
    if id in playlist:
        
        y, sr = librosa.load(track.path, sr=None)
        sd.play(y, sr)  # a sr kell, hogy a hang normál sebességgel szólaljon meg
        sd.wait()

    request.session["playlist"] = playlist

    query = request.GET.get("q", "")
    if query:
        return redirect(f"/?music_search={query}")

    return redirect("index")

   

def audio_play(request, id):
     
    track = Music.objects.get(id=id)

    request.session["track"] = track
     
    query = request.GET.get("q", "")
    if query:
        return redirect(f"/?music_search={query}")

    return redirect("index")

ez a JS playlist, de ez nem jó ugyebár, mert elveszik műveletnél (pl. új keresnél)
def add_to_playlist(request, id):
    music = Music.objects.get(id=id)
    
    return JsonResponse({
        "title": music.title,
        "performer": music.performer,
    })
'''