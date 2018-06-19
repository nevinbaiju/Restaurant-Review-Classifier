from django.urls import path

from . import views

app_name = 'review_classifier'

urlpatterns = [
    path('', views.index, name='index'),
    path('classify/', views.classify, name='classify'),
]