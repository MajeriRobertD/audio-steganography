# Generated by Django 3.2.2 on 2021-05-27 12:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('uploader', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='upload',
            name='encode_type',
            field=models.TextField(default='LSB', help_text='Encoding type'),
        ),
    ]
