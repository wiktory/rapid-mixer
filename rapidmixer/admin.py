from django.contrib import admin
from .models import Music

# Register your models here.

#admin.site.register(Music)

@admin.register(Music)
class MusicAdmin(admin.ModelAdmin):
    list_display = ("id","performer", "title","path",)
