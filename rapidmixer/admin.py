from django.contrib import admin
from .models import Music
from .models import MixGeneration

# Register your models here.

#admin.site.register(Music)

@admin.register(Music)
class MusicAdmin(admin.ModelAdmin):
    list_display = ("id","performer", "title","bpm","start_mix_point",'end_mix_point')
    search_fields = ("performer", "title")
    ordering = ("performer", "title")    

@admin.register(MixGeneration)
class MixGenerationAdmin(admin.ModelAdmin):
    list_display = ("job_id","created_at","status","downloaded_at","bpm","fade","track_count", "playlist_snapshot", "output_filename","duration_seconds","error_message",)
    list_filter = ("status",)
    ordering = ("-created_at",)