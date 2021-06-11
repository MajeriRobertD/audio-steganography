
from django.db import models
class Upload(models.Model):
    upload_file = models.FileField()    
    upload_date = models.DateTimeField(auto_now_add =True)
    encode_type = models.TextField(help_text='Encoding type', default='LSB')

    def delete(self, *args, **kwargs):
        self.upload_file.delete()
        super().delete(*args, **kwargs)


