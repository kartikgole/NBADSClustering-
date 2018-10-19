import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})
import unicodecsv
import random
import operator
import math
import csv


def myKmeans(X, k):
    # Visualize the cluster
    fig = plt.figure(num=None, figsize=(6, 6), edgecolor='k')
    ax = fig.add_subplot(1, 1, 1, facecolor="1.0")
    ax.scatter(X['FG'], X['FGA'], alpha=0.5, c='y', edgecolors='g', s=150)
    plt.title("475 values for FG * FGA")
    plt.xlabel('Field Goals')
    plt.ylabel('Field Goals Attempted')
    plt.show()

    # Define number of clusters


    # Generating random points so that centroids can be picked from them
    random_initial_pts = np.random.choice(X.index, size=k)

    # take ncluster random points as centroids
    centroids = X.loc[random_initial_pts]

    def centroids_to_dict(centroids):
        dictionary = dict()
        # iterating counter we use to generate a cluster_id
        counter = 0

        # iterate a pandas data frame row-wise using .iterrows()
        for index, row in centroids.iterrows():
            coordinates = [row['FG'], row['FGA']]
            dictionary[counter] = coordinates
            counter += 1

        return dictionary

    centroids_dict = centroids_to_dict(centroids)

    # Calculating Eucledian distance (like pythagoras thm)
    def calculate_distance(centroid, player_values):
        root_distance = 0

        for x in range(0, len(centroid)):
            difference = centroid[x] - player_values[x]
            squared_difference = difference ** 2
            root_distance += squared_difference

        euclid_distance = math.sqrt(root_distance)
        return euclid_distance
# check1
    #print(calculate_distance([0, 0], [3, 4]))

    def assign_to_cluster(row):
        player_vals = [row['FG'], row['FGA']]
        dist_prev = -1
        cluster_id = None

        for centroid_id, centroid_vals in centroids_dict.items():
            dist = calculate_distance(centroid_vals, player_vals)
            if dist_prev == -1:
                cluster_id = centroid_id
                dist_prev = dist
            elif dist < dist_prev:
                cluster_id = centroid_id
                dist_prev = dist
        return cluster_id

    # Apply to each row in normalised point_guards
    X['cluster'] = X.apply(lambda row: assign_to_cluster(row), axis=1)

    def recalculate_centroids(df):
        new_centroids_dict = dict()

        for cluster_id in range(0, k):
            df_cluster_id = df[df['cluster'] == cluster_id]

            xmean = df_cluster_id['FG'].mean()
            ymean = df_cluster_id['FGA'].mean()
            new_centroids_dict[cluster_id] = [xmean, ymean]

            # Finish the logic
        return new_centroids_dict

    centroids_dict = recalculate_centroids(X)

    def visualize_clusters(df, num_clusters, iteration):
        colors = ['b', 'y', 'r', 'c', 'y', 'm']
        fig = plt.figure(num=None, figsize=(6, 6), edgecolor='k')
        ax = fig.add_subplot(1, 1, 1, facecolor="1.0")
        for n in range(num_clusters):
            clustered_df = df[df['cluster'] == n]
            ax.scatter(clustered_df['FG'], clustered_df['FGA'], c=colors[n - 1], edgecolors='g', alpha=0.5, s=150)
            plt.xlabel('Field Goals')
            plt.ylabel('Field Goals Attempted')
            plt.title('Clustering standardized FG * FA, Iteration %s' % (iteration))
        plt.show()

    iteration = 0
    visualize_clusters(X, 5, iteration)

    def recalculate_centroids(df):
        new_centroids_dict = dict()

        for cluster_id in range(0, k):
            df_cluster_id = df[df['cluster'] == cluster_id]

            xmean = df_cluster_id['FG'].mean()
            ymean = df_cluster_id['FGA'].mean()
            new_centroids_dict[cluster_id] = [xmean, ymean]

            # Finish the logic
        return new_centroids_dict

    # Iteration 1

    centroids_dict = recalculate_centroids(X)
    X['cluster'] = X.apply(lambda row: assign_to_cluster(row), axis=1)
    visualize_clusters(X, k, 1)

    # Iteration 2

    centroids_dict = recalculate_centroids(X)
    X['cluster'] = X.apply(lambda row: assign_to_cluster(row), axis=1)
    visualize_clusters(X, k, 2)

    # Iteration 3

    centroids_dict = recalculate_centroids(X)
    X['cluster'] = X.apply(lambda row: assign_to_cluster(row), axis=1)
    visualize_clusters(X, k, 3)

def euclideanDist(x, xi):
    d = 0.0
    for i in range(len(x)-1):
        d += pow((float(x[i])-float(xi[i])),2)  #euclidean distance
    d = math.sqrt(d)
    return d

def knn_predict(test_data, train_data, k_value):
    for i in test_data:
        eu_Distance =[]
        knn = []
        good = 0

        bad = 0
        for j in train_data:
            eu_dist = euclideanDist(i, j)
            eu_Distance.append((j[5], eu_dist))
            eu_Distance.sort(key = operator.itemgetter(1))
            knn = eu_Distance[:k_value]
            for k in knn:
                if k[0] =='g':
                    good += 1
                else:
                    bad +=1
        if good > bad:
            i.append('g')
        elif good < bad:
            i.append('b')
        else:
            i.append('NaN')

#Accuracy calculation function
def accuracy(test_data):
    correct = 0
    for i in test_data:
        if i[5] == i[6]:
            correct += 1
    accuracy = float(correct)/len(test_data) *100  #accuracy
    return accuracy

def main():
    nbaff = pd.read_csv("NBAStats.csv")
    #p_guards = nbaff[nbaff['Pos'] == 'PG']
    nbaff.head(3)
    # print(p_guards)
    nbaff_numeric = nbaff.drop(columns=['Player', 'Pos', 'Tm'])
    #print(nbaff_numeric)
    data_normalised = (nbaff_numeric - nbaff_numeric.mean()) / (nbaff_numeric.max() - nbaff_numeric.min())
    #print(data_normalised)

    nclusters = 3
    #myKmeans(data_normalised , nclusters)

    nclusters = 5
    #myKmeans(data_normalised, nclusters)

    #7 Required columns
    data_normalised1 = data_normalised.values[:, [12, 9, 16, 19, 20, 21, 22]]
    print(data_normalised1)

    trainSet = data_normalised.values[:375, :]
    trainSet = np.array(trainSet).tolist()
    testSet = nbaff_numeric.values[375:, :]
    testSet = np.array(testSet).tolist()

    print(trainSet)
    print(testSet)

    K = 5  # Assumed K value
    knn_predict(testSet, trainSet, K)
    print("Accuracy : ", accuracy(testSet))


if __name__ == '__main__':
    main()