import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import re
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})


nbaff = pd.read_csv("NBAStats.csv")
p_guards = nbaff[nbaff['Pos']=='PG']
nbaff.head(3)
#print(p_guards)
nbaff_numeric=p_guards.drop(columns=['Player', 'Pos', 'Tm'])
print(nbaff_numeric)
data_normalised = (nbaff_numeric - nbaff_numeric.mean()) / (nbaff_numeric.max() - nbaff_numeric.min())
print(data_normalised)





# Visualize the cluster
fig = plt.figure(num=None, figsize=(6, 6), edgecolor='k')

ax = fig.add_subplot(1, 1, 1, facecolor="1.0")
ax.scatter(data_normalised['FG'], data_normalised['FGA'], alpha=0.5, c='y', edgecolors='g', s=150)
plt.title("Point Guards")
plt.xlabel('Field Goals')
plt.ylabel('Field Goals Attempted')
plt.show()

#Define number of clusters
nclusters = 3

#Generating random points so that centroids can be picked from them
random_initial_pts = np.random.choice(p_guards.index, size=nclusters)

#take ncluster random points as centroids
centroids = p_guards.loc[random_initial_pts]

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


def calculate_distance(centroid, player_values):
    root_distance = 0

    for x in range(0, len(centroid)):
        difference = centroid[x] - player_values[x]
        squared_difference = difference ** 2
        root_distance += squared_difference

    euclid_distance = math.sqrt(root_distance)
    return euclid_distance


# test
print(calculate_distance([0, 0], [3, 4]))


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


# Apply to each row in point_guards
p_guards['cluster'] = p_guards.apply(lambda row: assign_to_cluster(row), axis=1)
print(p_guards.head(2))


def recalculate_centroids(df):
    new_centroids_dict = dict()

    for cluster_id in range(0, nclusters):
        df_cluster_id = df[df['cluster'] == cluster_id]

        xmean = df_cluster_id['FG'].mean()
        ymean = df_cluster_id['FGA'].mean()
        new_centroids_dict[cluster_id] = [xmean, ymean]

        # Finish the logic
    return new_centroids_dict


centroids_dict = recalculate_centroids(p_guards)

def visualize_clusters(df, num_clusters, iteration):
    colors = ['b', 'y', 'r', 'c', 'y', 'm']
    fig = plt.figure(num=None, figsize=(6, 6), edgecolor='k')
    ax = fig.add_subplot(1, 1, 1, facecolor="1.0")
    for n in range(num_clusters):
        clustered_df = df[df['cluster'] == n]
        ax.scatter(clustered_df['FG'], clustered_df['FGA'], c=colors[n-1], edgecolors='g', alpha=0.5, s=150)
        plt.xlabel('Field Goals')
        plt.ylabel('Field Goals Attempted')
        plt.title('Iteration Number %s'%(iteration))
    plt.show()
iteration=0
visualize_clusters(p_guards, 5, iteration)


def recalculate_centroids(df):
    new_centroids_dict = dict()

    for cluster_id in range(0, nclusters):
        df_cluster_id = df[df['cluster'] == cluster_id]

        xmean = df_cluster_id['FG'].mean()
        ymean = df_cluster_id['FGA'].mean()
        new_centroids_dict[cluster_id] = [xmean, ymean]

        # Finish the logic
    return new_centroids_dict


centroids_dict = recalculate_centroids(p_guards)

# Iteration 1
p_guards['cluster'] = p_guards.apply(lambda row: assign_to_cluster(row), axis=1)
visualize_clusters(p_guards, nclusters, 1)


# Iteration 2

centroids_dict = recalculate_centroids(p_guards)
p_guards['cluster'] = p_guards.apply(lambda row: assign_to_cluster(row), axis=1)
visualize_clusters(p_guards, nclusters, 2)

# Iteration 3

centroids_dict = recalculate_centroids(p_guards)
p_guards['cluster'] = p_guards.apply(lambda row: assign_to_cluster(row), axis=1)
visualize_clusters(p_guards, nclusters, 3)

