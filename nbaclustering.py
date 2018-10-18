import pandas as pd
import numpy as np


nbaff = pd.read_csv("NBAStats.csv")
nbaff.head(3)

"print(nbaff)"

p_guards = nbaff[nbaff['Pos']=='PG']
print(p_guards)

#Define number of clusters
nclusters = 3

#Generating random points so that centroids can be picked from them
random_initial_pts = np.random.choice(p_guards.index, size=nclusters)

#take ncluster random points as centroids
centroids=p_guards.loc[random_initial_pts]