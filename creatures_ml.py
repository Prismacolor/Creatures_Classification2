'''A critter classification script'''

import numpy
import pandas

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# retrieving the data from the spreadsheet TR
creatures_file = "C:\\Users\\Tay042919\\PycharmProjects\\ML_Playground\\Creatures.xlsx"
creatures_extract = pandas.read_excel(creatures_file)
creatures_df = pandas.DataFrame(creatures_extract)

# we do not need this initial column, contains no useful data TR
del creatures_df['Creature']
# print(creatures_df)


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
    the_model = MultinomialNB().fit(X_train, y_train)

    # test out the model
    predicted = the_model.predict(X_test)

    print(y_test)
    print(predicted)
    print(numpy.mean(predicted == y_test))


creatures_fitted_df = adjust_data(creatures_df)
classify_data(creatures_fitted_df)

'''C:\Python\Python37-32\python.exe C:/Users/Tay042919/PycharmProjects/ML_Playground/creatures_ml.py
success
[3 2 0 0 2 0 2 3 1 2 0 3 3 3 1 1 3 2 0 2 3 1 2 3 2 2 2 2 0 0 2 1 3 2 0 2 0
 1 3 3 2 3]
1.0

Process finished with exit code 0'''
