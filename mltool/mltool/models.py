
from django.db import models

class file(models.Model):

   picture = models.FileField(upload_to = 'static/')


