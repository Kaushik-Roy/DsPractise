import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style



# read the data
data = pd.read_csv("data/student-mat.csv", sep=";")

# print(data.head())

# select required attributes
data = data[
    ["G1", "G2", "G3", "studytime", "failures", "absences"]
]

# predict label
predict = "G3"

# Remove the predicting/ Outcome from x label
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# print(x)
# print(y)

#   Split the x and y data into train and test data, test_size=0.1 means 10%
#   of the data will be for testing and rest for training
#   TODO: Need to explore about sklearn.model_selection.train_test_split() method
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

"""
# loop the training model for a certain number of time to get a best score
best_score = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    # TRAINING Model
    # Create Linear Regression Model
    # Fit and find the accuracy
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test) 
    
    print(f"accuracy {accuracy}")

    # if the accuracy is better than the best score than save the model
    if(accuracy > best_score):
        best_score = accuracy
        # save the model using pickle
        with open("model/stundentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
"""

# Load model from pickle file
pickle_file = open("model/stundentmodel.pickle", "rb")
linear = pickle.load(pickle_file)

# print('Coefficients: \n', linear.coef_)
# print('intercept: \n', linear.intercept_)


# Predicting from the model
predictions = linear.predict(x_test)

# print all the predictions
for x in range(len(predictions)):
    print(f"predict {predictions[x]}, {x_test[x]}, {y_test[x]}")