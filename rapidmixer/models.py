from django.db import models

# Create your models here.
class Music(models.Model):
    performer = models.CharField(max_length=50)
    title = models.CharField(max_length=50)
    length = models.FloatField()
    path = models.CharField(max_length=50)

    def __str__(self) -> str:
        return self.performer
    