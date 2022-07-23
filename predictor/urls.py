from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('predictions/', views.predictions, name='predictions'),
    path('dataVisualization/', views.dataVisualization, name='dataVisualization'),
    path('predictionResults/', views.predictionResults, name='predictionResults'),
    path('',views.homePage, name='homePage')
]