from django import forms
from .models import *


class loanForm(forms.ModelForm):
    class Meta():
        model=loanModel
        fields=['no_of_dependents','Loan_amount','Loan_term','Cibil_score']
