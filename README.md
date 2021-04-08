# Credit-Card-Defaulters

## This project has the below major parts:

1. CreditCardDefaulters.ipynb: This file contains code for the Machine Learning model to predict the sales rate.
2. app.py - This file contains Flask APIs that receives employee details through GUI or API calls, computes the precited value based on the model and returns it.
3. templates - This folder contains the HTML template to allow user to enter the input details and display the predicted sales.
4. static - This folder contains the css template to style the webpage.

## Running the project

1. Ensure that you are in the project home directory. Make sure the serialized model.pkl file also exists in this directory.

2. Now run the below command to start Flask API

python app.py

3. By default, flask will run on port 5000, navigate to the below URL.

URL: http://localhost:5000
