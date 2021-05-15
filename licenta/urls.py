"""licenta URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
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

from django.conf import settings
from django.conf.urls.static import static
from uploader import views as uploader_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', uploader_views.UploadView.as_view(), name='fileupload'),
    path('upload_list/upload_file', uploader_views.UploadView.as_view() , name='upload_file'),
    # path('upload_list', uploader_views.upload_list , name='upload_list'),
    path('upload_list', uploader_views.UploadList.as_view() , name='upload_list'),
    path('upload_list/<int:pk>/', uploader_views.delete_upload , name='delete_upload'),
    path('upload_list/encoded/<int:pk>/', uploader_views.encode_upload , name='encode_upload'),
    path('upload_list/decoded/<int:pk>/', uploader_views.decode_upload , name='decode_upload'),
    path('upload_list/graph/<int:pk>/', uploader_views.draw_graph , name='draw_graph'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

