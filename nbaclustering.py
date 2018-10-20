"Data Mining CSE 5334: Assignment 1"
"Karteek Gole, UTA ID: 1001553522"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})
import operator
import math


def myKmeans(X, k):

    # Just checking all data points in the NBA dataset by using features FG and FGA

    fig = plt.figure(num=None, figsize=(6, 6), edgecolor='k')
    ax = fig.add_subplot(1, 1, 1, facecolor="1.0")
    ax.scatter(X['FG'], X['FGA'], alpha=0.5, c='y', edgecolors='g', s=150)
    plt.title("475 values for FG * FGA")
    plt.xlabel('Field Goals')
    plt.ylabel('Field Goals Attempted')
    plt.show()

    # Generating random points so that centroids can be picked from them (i.e acc to no of clusters value k)
    random_initial_pts = np.random.choice(X.index, size=k)

    # take ncluster random points as centroids
    centroids = X.loc[random_initial_pts]

    #This function takes centroid dataframe obj as input and converts the feature attributes for that centroid into coordinates
    def centroids_data(centroids):
        dictionary = dict()
        # iterating counter, need this to generate cluster_id
        counter = 0

        # traversing the data frame using iterrows
        for index, row in centroids.iterrows():
            coordinates = [row['FG'], row['FGA']]
            dictionary[counter] = coordinates
            counter += 1

        return dictionary

    centroids_dict = centroids_data(centroids)

    # Calculating Eucledian distance (like pythagoras thm) between centroids and players
    def calc_dist(centroid, player_nos):
        root_distance = 0

        for x in range(0, len(centroid)):
            diff = centroid[x] - player_nos[x]
            sq_diff = diff ** 2
            root_distance += sq_diff

        euc_dis = math.sqrt(root_distance)
        return euc_dis

    #we know the distances now we'll determine which point belongs to which cluster
    def assign_clust(row):
        player_vals = [row['FG'], row['FGA']]
        dist_prev = -1
        cluster_id = None

        for centroid_id, centroid_vals in centroids_dict.items():
            dist = calc_dist(centroid_vals, player_vals)
            if dist_prev == -1:
                cluster_id = centroid_id
                dist_prev = dist
            elif dist < dist_prev:
                cluster_id = centroid_id
                dist_prev = dist
        return cluster_id

    # Apply to each row in normalised players
    X['cluster'] = X.apply(lambda row: assign_clust(row), axis=1)

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

    def vis_clust(df, num_clusters, iteration):
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
    vis_clust(X, 5, iteration)

    #3 Iterations for each k, to find the right cluster density
    # Iteration 1

    centroids_dict = recalculate_centroids(X)
    X['cluster'] = X.apply(lambda row: assign_clust(row), axis=1)
    vis_clust(X, k, 1)

    # Iteration 2

    centroids_dict = recalculate_centroids(X)
    X['cluster'] = X.apply(lambda row: assign_clust(row), axis=1)
    vis_clust(X, k, 2)

    # Iteration 3

    centroids_dict = recalculate_centroids(X)
    X['cluster'] = X.apply(lambda row: assign_clust(row), axis=1)
    vis_clust(X, k, 3)

def euclideanDist(x, xi):
    d = 0.0
    for i in range(len(x)-1):
        d += pow((float(x[i])-float(xi[i])),2)  #euclidean distance
    d = math.sqrt(d)
    return d

def myKNN(test_data, train_data, k_value):
    for i in test_data:
        eu_Distance =[]
        knn = []
        good = 0
        bad = 0
        for j in train_data:
            eu_dist = euclideanDist(i, j)
            #print(eu_dist)
            eu_Distance.append((j[5], eu_dist))
            eu_Distance.sort(key=operator.itemgetter(1))
            knn = eu_Distance[:k_value]
            #print(knn)
            for k in knn:
                #print(k[0])
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
    #reading CSV input
    nbaff = pd.read_csv("NBAStats.csv")

    #dropping columns with strings
    nbaff_numeric = nbaff.drop(columns=['Player', 'Pos', 'Tm'])
    #standardizing data
    data_normalised = (nbaff_numeric - nbaff_numeric.mean()) / (nbaff_numeric.max() - nbaff_numeric.min())

    #K Means for 3 clusters
    nclusters = 3
    myKmeans(data_normalised , nclusters)

    #K Means for 5 clusters
    nclusters = 5
    myKmeans(data_normalised, nclusters)

    #Taking 7 Required attribute columns for Question 4
    data_normalised_7rows = data_normalised.values[:, [12, 9, 16, 19, 20, 21, 22]]


    trainSet = data_normalised_7rows[:375, :]
    trainSet1 = np.array(trainSet).tolist()
    testSet = data_normalised_7rows[375:, :]
    testSet1 = np.array(testSet).tolist()

    print(trainSet1)
    print(testSet1)

    K = 5  # Assumed K value
    myKNN(testSet1, trainSet1, K)
    print("Accuracy : ", accuracy(testSet1))

    fulltrainset = data_normalised.values[:375, :]
    fulltrainset1 = np.array(fulltrainset).tolist()
    fulltestset = data_normalised.values[375:, :]
    fulltestset1 = np.array(fulltestset).tolist()

    print(fulltrainset1)
    print(fulltestset1)

    K = 3 #Assumed K value for full train set
    myKNN(fulltestset1, fulltrainset1, K)
    print("Accuracy: ", accuracy((fulltestset1)))


if __name__ == '__main__':
    main()