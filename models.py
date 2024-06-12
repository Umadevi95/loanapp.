# Create your models here.
from django.db import models

# Create your models here.
class loanModel(models.Model):

    no_of_dependents=models.IntegerField()
    Loan_amount=models.FloatField()
    Loan_term=models.IntegerField()
    Cibil_score=models.FloatField()
