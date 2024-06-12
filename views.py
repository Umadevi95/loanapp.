from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpRequest
from django.urls import reverse_lazy
from django.views.generic import View
from .forms import loanForm
from django.core.files.storage import FileSystemStorage
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
import sklearn
import joblib
import pickle
import matplotlib.pyplot as plt

np.random.seed(123)  # Ensure reproducibility

class dataUploadView(View):
    form_class = loanForm
    success_url = reverse_lazy('success')
    template_name = 'create.html'
    failure_url = reverse_lazy('fail')
    filenot_url = reverse_lazy('filenot')

    def get(self, request, *args, **kwargs):
        form = self.form_class()
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            data_No_of_dependents = request.POST.get('No_of_dependents')
            data_Loan_amount = request.POST.get('Loan_amount')
            data_Loan_term = request.POST.get('Loan_term')
            data_Cibil_score = request.POST.get('Cibil_score')


            # Load and preprocess the dataset
            data = pd.read_csv("preprocessedloan_data.csv")
            data = pd.get_dummies(data, drop_first=True)
            X = data.drop(columns=[' loan_status_ Rejected', 'loan_id'])
            y = data[' loan_status_ Rejected']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a RandomForestClassifier model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy:", accuracy)

            # Select top 4 features
            selector = SelectKBest(f_classif, k=4)
            selector.fit(X, y)
            feature_names = X.columns
            selected_feature_names = [feature_names[i] for i in selector.get_support(indices=True)]
            print(f"Selected {4} features:")
            print(selected_feature_names)

            # Reduce the feature set
            X_reduced = X[[' No_of_dependents', ' loan_amount', ' loan_term', ' cibil_score']]
            X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

            # Train a new model on the reduced feature set
            model_reduced = RandomForestClassifier()
            model_reduced.fit(X_train, y_train)

            filename_reduced = "reduced_model.sav"
            loaded_model_reduced = pickle.load(open(filename_reduced, 'rb'))

            # Prepare user input for prediction
            def get_user_input():
                return np.array([[int(data_No_of_dependents), float(data_Loan_amount), int(data_Loan_term), float(data_Cibil_score)]])

            selected_features = get_user_input()
            result = loaded_model_reduced.predict(selected_features)
            print("Prediction result:", result[0])

            dicc = {'yes': 1, 'no': 0}
            data = np.array([data_No_of_dependents, data_Loan_amount, data_Loan_term, data_Cibil_score])
            out = loaded_model_reduced.predict(data.reshape(1, -1))

            return render(request, "succ_msg.html", {
                'data_No_of_dependents': data_No_of_dependents,
                'data_Loan_amount': data_Loan_amount,
                'data_Loan_term': data_Loan_term,
                'data_Cibil_score': data_Cibil_score,
                'out': out
            })

        else:
            return redirect(self.failure_url)
