'''A critter classification script'''

import numpy
import pandas

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

# retrieving the data from the spreadsheet TR
creatures_file = "data_samples\\creatures_data_set.xlsx"
creatures_extract = pandas.read_excel(creatures_file)
creatures_df = pandas.DataFrame(creatures_extract)

# we do not need this initial column, contains no useful data TR
del creatures_df['Creature']
# print(creatures_df)

# prep the second test set
test_file = "data_samples\\creatures_test_set.xlsx"
test_extract = pandas.read_excel(test_file)
test_df = pandas.DataFrame(test_extract)
del test_df['Creature']


# need to adjust final column to be numbers only TR
def adjust_data(my_dataframe):
    data_encoder = preprocessing.LabelEncoder()
    my_dataframe.iloc[:, -1] = data_encoder.fit_transform(my_dataframe.iloc[:, -1])

    # print(my_dataframe)
    return my_dataframe


def classify_data(data_frame):
    # first create the training data for the model  TR
    X = data_frame.iloc[:, 0: -1]
    y = data_frame.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

    # print(X_train, X_test, y_train, y_test)

    # create the classification model   TR
    the_model = GaussianNB().fit(X_train, y_train)

    # test out the model
    predicted = the_model.predict(X_test)

    # print(y_test)
    # print(predicted)
    print(numpy.mean(predicted == y_test))
    print("Total points: ", (X_test.shape[0], "Mislabelled: ", (y_test != predicted).sum()))

    return the_model


def test_data(data_frame, model):
    X = data_frame.iloc[:, 0: -1]
    y = data_frame.iloc[:, -1]
    predicted2 = model.predict(X)

    # print(numpy.mean(predicted2 == y))
    print("Total points: ", (X.shape[0], "Mislabelled: ", (y != predicted2).sum()))


# create initial model
creatures_fitted_df = adjust_data(creatures_df)
my_model = classify_data(creatures_fitted_df)

# test it out
adjusted_test_data = adjust_data(test_df)
test_data(adjusted_test_data, my_model)
