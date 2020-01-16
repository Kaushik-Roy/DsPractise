import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


data = pd.read_csv("car.data")
# printing the starting line
# print(data.head())

pre = preprocessing.LabelEncoder()

# fields
# This returns a numpy array which trans forms all the field values to integer values
buying = pre.fit_transform(list(data["buying"]))
maint = pre.fit_transform(list(data["maint"]))
doors = pre.fit_transform(list(data["doors"]))
persons = pre.fit_transform(list(data["persons"]))
lug_boot = pre.fit_transform(list(data["lug_boot"]))
safety = pre.fit_transform(list(data["safety"]))
cls = pre.fit_transform(list(data["class"]))

# print(buying)

# predition
predict = "class"

# pre-pare labels
x = list(zip(buying, maint, doors, persons, lug_boot, safety))
y = list(cls)

# print(x)

# Training splits
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# print(x_train)
# print(y_test)

# prepare Model
model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)

print(accuracy)

predicted_data = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted_data)):
    print(f"Predicted: {names[predicted_data[x]]}, Data: {x_test[x]}, Actual: {names[y_test[x]]}")
