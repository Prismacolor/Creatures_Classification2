import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlrd
import openpyxl

# sklearn libraries
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# retrieving the data from the spreadsheet TR
creatures_file = r'data_samples\creatures_data_set.csv'
creatures_extract = pd.read_csv(creatures_file)
creatures_df = pd.DataFrame(creatures_extract)

# prep the test set
test_file = r'data_samples\creatures_test_set.csv'
test_extract = pd.read_csv(test_file)
test_df = pd.DataFrame(test_extract)

# prep the second test set
test_file2 = r'data_samples\creatures_test_set2.csv'
test_extract2 = pd.read_csv(test_file2)
test_df2 = pd.DataFrame(test_extract2)
del test_df2['Number']


# ***encode the data***
# todo: make a functions script that contains this function so i can import into both scripts rather than copying code
def prepare_data(critter_df):
    # encode the data with custom mappings
    environment_mappings = {'air': 0, 'cave': 1, 'desert': 2, 'swamp': 3, 'volcanoes': 4, 'water': 5, 'woods': 6,
                            'mountain': 7}
    creature_mappings = {'Dragon': 0, 'Drake': 1, 'Flying Serpent': 2, 'Serpent': 3, 'Wyrm': 4, 'Wyvern': 5}

    # first column is irrelevant
    del critter_df['Number']
    critter_df.Environment = critter_df.Environment.map(environment_mappings)
    critter_df.Classification = critter_df.Classification.map(creature_mappings)

    return critter_df


# ***compare features and determine relationships***
def make_plots(critter_df):
    critter_df.plot(kind='scatter', x='Legs', y='Classification', color='purple')
    plt.savefig('plots\\legs_class.png')

    critter_df.plot(kind='scatter', x='Wings', y='Classification', color='purple')
    plt.savefig('plots\\wings_class.png')

    critter_df.plot(kind='scatter', x='Length', y='Classification', color='blue')
    plt.savefig('plots\\length_class.png')

    critter_df.plot(kind='scatter', x='Height', y='Classification', color='blue')
    plt.savefig('plots\\height_class.png')

    critter_df.plot(kind='scatter', x='Environment', y='Classification', color='green')
    plt.savefig('plots\\env_class.png')


# ***create the clustering model***
def create_clusters(critter_df):
    # find the ideal K
    X = np.array(critter_df.drop(['Classification'], 1).astype(float))
    distortions = []

    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
        km.fit(X)
        distortions.append(km.inertia_)

    plt.figure(0)
    plt.plot(range(1, 11), distortions, marker=0)
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.savefig('plots\\ideal_k.png')
    plt.show()
    # the result of this plot shows that 'three' is perhaps the ideal number of clusters for this set


def create_final_model(X_train):
    scaler = MinMaxScaler()
    final_k = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)

    X = np.array(X_train.drop(['Classification'], 1).astype(float))
    X_scaled = scaler.fit_transform(X)

    y = X_train['Classification']
    pred_y = final_k.fit(X_scaled)

    # check to see how the model performed
    correct = 0
    for i in range(len(X)):
        predict_me = np.array(X[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = final_k.predict(predict_me)
        if prediction[0] == y[i]:
            correct += 1

    print(correct / len(X))

    # plot the data
    plt.figure(1)
    plt.scatter(X[:, 0], X[:, 1], label='True Position')
    plt.xlabel('Features')
    plt.ylabel('Classification')
    plt.savefig('plots\\X_train.png')

    # plot the clusters
    plt.scatter(X[pred_y == 0, 0], X[pred_y == 0, 1], s=50, c='green', marker='o', edgecolor='black', label='cluster 1')
    plt.scatter(X[pred_y == 1, 0], X[pred_y == 1, 1], s=50, c='blue', marker='o', edgecolor='black', label='cluster 2')
    plt.scatter(X[pred_y == 2, 0], X[pred_y == 2, 1], s=50, c='purple', marker='o', edgecolor='black', label='cluster 3')

    # plot the centroids
    plt.scatter(final_k.cluster_centers_[:, 0], final_k.cluster_centers_[:, 1], s=250, marker='*', c='black',
                edgecolor='black', label='centroids')

    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()


cleaned_creature_df = prepare_data(creatures_df)
# make_plots(cleaned_creature_df)
create_clusters(cleaned_creature_df)
create_final_model(cleaned_creature_df)

print('Process finished.')
