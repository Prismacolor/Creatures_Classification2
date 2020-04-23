import matplotlib.pyplot as plt
import numpy
import pandas

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

creatures_file = r'data_samples\creatures_data_set.csv'
creatures_extract = pandas.read_csv(creatures_file)
# you can also read from json and excel files
creatures_df = pandas.DataFrame(creatures_extract)

creatures_test_file = r'data_samples\creatures_test_set.csv'
creatures_test_extract = pandas.read_csv(creatures_test_file)
creatures_test_df = pandas.DataFrame(creatures_test_extract)


def clean_data(some_dataframe):
    print(some_dataframe.head())
    # snapshot of the dataframe, can see easily if there are columns you don't need
    del some_dataframe['Number']

    # you need all numerical data for this classification model
    environment_mappings = {'air': 0, 'cave': 1, 'desert': 2, 'swamp': 3, 'volcanoes': 4,
                            'water': 5, 'woods': 6, 'mountain': 7}
    creature_mappings = {'Dragon': 0, 'Drake': 1, 'Flying Serpent': 2, 'Serpent': 3, 'Wyrm': 4, 'Wyvern': 5}

    some_dataframe.Environment = some_dataframe.Environment.map(environment_mappings)
    some_dataframe.Classification = some_dataframe.Classification.map(creature_mappings)

    return some_dataframe


def make_tree_model(some_dataframe):
    """"Supervised (meaning you have some test examples that have the answers attached.)
	Give high level example: is this a dragon or a snake?
	First branch/node: does it have wings
	Then maybe second node is does it have legs?
	Results in a decision about what the item is
	Can handle both numerical and categorical data
	Be careful, they can get overly complex and really touchy with each subtle change in data
	You can adjust by "pruning" meaning you specify the number of samples needed before there's a split (not supported)
    """

    X = some_dataframe.iloc[:, 0: -1]
    y = some_dataframe.iloc[:, -1]
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    creatures_model = tree.DecisionTreeClassifier().fit(X_train, y_train)

    plt.figure()
    tree.plot_tree(creatures_model)
    plt.show()

    y_pred = creatures_model.predict(x_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return creatures_model


def make_random_forest(some_dataframe):
    """	Gini impurity: What this means is the decision tree tries to form nodes containing a high proportion of samples
    (data points) from a single class by finding values in the features that cleanly divide the data into classes.
    Leads to 1 - sum of (pi) squared
    It's 1 - the sum over all classes of the fraction of examples in each class
    You want the smallest amount of impurity, it can go until each node only contains samples from one class
    Collection of decision trees, known as ensemble learning
    Improve results by combining a number of different learners together
    Also helps prevent over fitting to one model, this is the one of the more common ones
    (*n_estimators (number of trees)
    max_depth, opt, lets you run until all leaves contain less than min samples split samples
    min_samples_split =, opt the minimal number of samples required to split an internal node
    min_samples_leaf, opt minimum number of samples required to be at a leave (leaf needs at least min samples leaf training examples in both branches)
    *Max_features = number of features to consider when looking for best split,
    auto will use sqrt (n_features) or you can use sqrt, if none, use all features/
    using sqrt will only use 4 random features before splitting a node, more random, better trained
    *Bootstrap= use a sample of data or whole data
    n_jobs: how many jobs are to run in parallel
    Random_state = initial state, controls the randomness of samples
    Warm_start= False (if true reuse solution from previous training)
    """

    X = some_dataframe.iloc[:, 0: -1]
    y = some_dataframe.iloc[:, -1]
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    forest_model = RandomForestClassifier(n_estimators=100, random_state=33)
    forest_model.fit(X_train, y_train)

    forest_model.predict(x_test)
    forest_score = forest_model.score(x_test, y_test)

    print(forest_score)

    return forest_model


def test_model(model, dataframe):
    X = dataframe.iloc[:, 0: -1]
    y = dataframe.iloc[:, -1]

    pred_y = model.predict(X)

    print(classification_report(y, pred_y))
    print(accuracy_score(y, pred_y))


cleaned_dataframe = clean_data(creatures_df)
tree_model_1 = make_tree_model(cleaned_dataframe)
# optional
forest_model_1 = make_random_forest(cleaned_dataframe)

cleaned_dataframe2 = clean_data(creatures_test_df)
test_model(tree_model_1, cleaned_dataframe2)
test_model(forest_model_1, cleaned_dataframe2)


