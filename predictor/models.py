from cProfile import label
from django.db import models
from django.core.validators import MinValueValidator

# Create your models here.
class Predictions(models.Model):
    
    Team_One = models.CharField(max_length=50)
    Team_One_Passing_Yards = models.FloatField(validators=[MinValueValidator(0)])
    Team_One_Rushing_Yards = models.FloatField(validators=[MinValueValidator(0)])
    Team_One_Passing_Yards_Allowed_On_Defense = models.FloatField(validators=[MinValueValidator(0)])
    Team_One_Rushing_Yards_Allowed_On_Defense = models.FloatField(validators=[MinValueValidator(0)])
    Team_One_Turnovers_Created = models.FloatField(validators=[MinValueValidator(0)])
    Team_One_Third_Downs_Converted = models.FloatField(validators=[MinValueValidator(0)])
    Team_One_Sacks = models.FloatField(validators=[MinValueValidator(0)])

    Team_Two = models.CharField(max_length=50)
    Team_Two_Passing_Yards = models.FloatField(validators=[MinValueValidator(0)])
    Team_Two_Rushing_Yards = models.FloatField(validators=[MinValueValidator(0)])
    Team_Two_Passing_Yards_Allowed_On_Defense = models.FloatField(validators=[MinValueValidator(0)])
    Team_Two_Rushing_Yards_Allowed_On_Defense = models.FloatField(validators=[MinValueValidator(0)])
    Team_Two_Turnovers_Created = models.FloatField(validators=[MinValueValidator(0)])
    Team_Two_Third_Downs_Converted = models.FloatField(validators=[MinValueValidator(0)])
    Team_Two_Sacks = models.FloatField(validators=[MinValueValidator(0)])