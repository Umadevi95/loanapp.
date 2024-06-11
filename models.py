# Create your models here.
from django.db import models

# Create your models here.
class loanModel(models.Model):

    No_of_dependents=models.IntegerField()
    Loan_amount=models.FloatField()
    Loan_term=models.FloatField()
    Cibil_score=models.FloatField()
