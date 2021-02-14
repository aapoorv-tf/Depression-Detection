from django.urls import path, include
from . import views

urlpatterns = [
   path("", views.homepage, name='home'),
   path("search", views.search, name='search'),
   path("api", views.tf_processing.as_view(), name='api'),
]