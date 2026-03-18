"""
URL configuration for RMsite project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from rapidmixer import views

urlpatterns = [
    path('', views.index, name="index"),
    path("admin/", admin.site.urls),
    path("playlist/add/<int:id>/", views.add_to_playlist, name="add_to_playlist"), # ez kell a playlist átadáshoz és meghívja a views-ben az add_to_playlist függvényt
    path("playlist/delete/<int:id>/", views.delete_from_playlist, name="delete_from_playlist"), # ez kell a playlistből való törléshez  és meghívja a views-ben az delete_from_playlist függvényt
    #path("playlist/play/<int:id>/", views.audio_play, name="audio_play"),
    path("playlist/delete_all/", views.delete_all_from_playlist, name="delete_all_from_playlist"),
    #path("playlist/mix_musics/<int:bpm>/<int:fade>", views.mix_playlist_musics, name="mix_playlist_musics"),
    path("playlist/start_mix/<int:bpm>/<int:fade>/", views.start_mix, name="start_mix"),
    path("playlist/mix_progress/<str:job_id>/", views.mix_progress, name="mix_progress"),
    path("playlist/download_mix/<str:job_id>/", views.download_mix, name="download_mix")
]