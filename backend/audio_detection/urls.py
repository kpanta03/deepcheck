# audio_detection/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('detect_audio/', views.detect_audio, name='detect_audio'),
]
