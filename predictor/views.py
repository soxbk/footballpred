from re import template
from django.urls import reverse
from django.shortcuts import redirect, render
from django.http import HttpResponseRedirect
from .ANN import annML

from .models import models
# from ... import ANN
# from myFootballSite import predictor
from .forms import PredictionsForm
# Create your views here.
def predictions(request):

    #Post request --> form contents --> 
    if request.method == 'POST':
        
        form = PredictionsForm(request.POST)
        if form.is_valid():
            form.save()
            form = form.cleaned_data
            return render(request, 'predictor/predictionResults.html', context = form)

    else:
      form = PredictionsForm()
    return render(request, 'predictor/predictions.html', context={'form':form})

def dataVisualization(request):
    return render(request, 'predictor/dataVisualization.html')

def predictionResults(request):

    print('success')
    if request.method == 'POST':
        post_data = request.POST

        print(post_data)
        team_one = []
        team_two = []

        for key, value in post_data.items():
            # print(key, "===>", value)
            if "One" in str(key):
                if value != post_data.get('Team_One'):
                    team_one.append(float(value))
            elif"Two" in str(key):
                if value != post_data.get('Team_Two'):
                    team_two.append(float(value))

        print("Team ONe => ", team_one, "\n","Team Two => ", team_two)
 


    team1_win_percentage = float(annML(team_one))
    team2_win_percentage = float(annML(team_two))

    print(team1_win_percentage)

    data = {
        'Team_One': post_data.get('Team_One'),
        'Team_One_Win_Percentage': team1_win_percentage,
        'Team_Two': post_data.get('Team_Two'),
        'Team_two_Win_Percentage': team2_win_percentage
    }
    return render(request, 'predictor/predictionResults.html', context=data) 


def homePage(request):
    return render (request, 'predictor/homePage.html')