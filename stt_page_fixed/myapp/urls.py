from django.conf import settings
from django.conf.urls.static import static
from . import views
from django.urls import path
from .views import summarize_text
from .views import select_number_of_people, voice_separation, select_speaker, query_view, home

urlpatterns = [
    path('', home, name='home'),
    path('choose/', views.choose, name='choose'),
    path('make/', views.make, name='make'),
    path('make_ch1/', views.make_ch1, name='make_ch1'),
    path('choose_ch1/', views.choose_ch1, name='choose_ch1'),
    path('chat/', views.chat, name='chat'),
    path('select_number_of_people/', select_number_of_people, name='select_number_of_people'),
    path('query_view/', query_view, name='query_view'),
    path('voice_separation/', voice_separation, name='voice_separation'),
    path('select_speaker/', select_speaker, name='select_speaker'),
    path('api/summarize/', summarize_text, name='summarize_text'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
