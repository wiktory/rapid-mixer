from django.db import models

# Create your models here.
class Music(models.Model):
    performer = models.CharField(max_length=50)
    title = models.CharField(max_length=50)
    bpm = models.FloatField()
    path = models.CharField(max_length=50)

    def __str__(self) -> str:
        return self.performer
    
class MixGeneration(models.Model):
    job_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default="processing")
    bpm = models.IntegerField()
    fade = models.IntegerField()
    track_count = models.IntegerField(default=0)
    playlist_snapshot = models.TextField(blank=True, null=True)
    output_filename = models.CharField(max_length=255, blank=True, null=True)
    duration_seconds = models.FloatField(blank=True, null=True)
    error_message = models.TextField(blank=True, null=True)
    downloaded_at = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return f"{self.job_id} - {self.status}"