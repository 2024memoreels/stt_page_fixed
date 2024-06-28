from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from .views import summarize_text
from .views import select_number_of_people, voice_separation, select_speaker, query_view

urlpatterns = [
    path('', select_number_of_people, name='select_number_of_people'),
    path('query_view/', query_view, name='query_view'),
    path('voice_separation/', voice_separation, name='voice_separation'),
    path('select_speaker/', select_speaker, name='select_speaker'),
    path('api/summarize/', summarize_text, name='summarize_text'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
