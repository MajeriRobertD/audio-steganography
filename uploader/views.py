from typing import List
from django.db.models.base import Model
from django.shortcuts import redirect, render

from django.views.generic.edit import CreateView
from django.views.generic import ListView
from django.urls import reverse_lazy
from .models import Upload
from .utils import get_plot, get_coords, encode, decode
from .forms import UploadForm
from django.core.files import File


class UploadView(CreateView):
    model = Upload
    fields = ['upload_file', ]
    success_url = reverse_lazy('upload_list')
    template_name = 'uploader/upload_file.html'
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
    
        return context

def delete_upload(request, pk):
    if request.method == 'POST':
        upload = Upload.objects.get(pk = pk)
        upload.delete()
    return redirect('upload_list')


def encode_upload(request, pk):
    if request.method == 'POST':
        upload = Upload.objects.get(pk = pk)
        message = request.POST.get("message")
        key = request.POST.get("key")
        encode('./' +upload.upload_file.url, message, key)
        the_name = "encoded_" + upload.upload_file.name 
        instance = Upload.objects.create(upload_file=File(file=open('/home/robert/licenta/app/licenta/media/song_embedded.wav', 'rb'), name= the_name)) 

    return redirect('upload_list')

def decode_upload(request, pk):
    if request.method == 'POST':
        upload = Upload.objects.get(pk = pk)
        key = request.POST.get("key")
        decoded = decode('./' +upload.upload_file.url, key)
    return render(request, 'uploader/upload_list.html', {
        'decoded': decoded,
        'uploads': Upload.objects.all(),
    } )

def draw_graph(request, pk):
    if request.method == 'POST':
        upload = Upload.objects.get(pk = pk)
        graph = get_coords('./' +upload.upload_file.url)
        x = graph[0]
        y = graph[1]
        chart = get_plot(x,y)
        name = upload.upload_file.name
       
    return render(request, 'uploader/graph.html', {
        'chart': chart,
        'name': name
    } )
    


class UploadList(ListView):
    model = Upload
    template_name = 'uploader/upload-list.html'
    context_object_name = 'uploads'

    

def upload_file(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('upload_list')

    else:
        form = UploadForm()
    return render(request, 'uploader/upload_file.html', {
        'form':form
    })