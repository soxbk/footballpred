from django import forms
from .models import Predictions
from django.forms import ModelForm


class PredictionsForm(ModelForm):
    class Meta:
        model = Predictions
        # fields = ['passingYards', 'rushingYards', 'passingYardsAllowedDefense', 'rushingYardsAllowedDefense', 'turnoversCreated', 'thirdDownsConverted', 'sacks']
        fields = '__all__' #  pass in all model fields as form fields

