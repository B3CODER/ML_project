import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

df = pd.read_json('data.json')

# Convert 'latitude' and 'longitude' to numeric type
df['latitude'] = pd.to_numeric(df['latitude'])
df['longitude'] = pd.to_numeric(df['longitude'])

plt.figure(figsize=(8, 6))
sns.scatterplot(x='latitude', y='longitude', data=df, hue='id')
plt.legend(bbox_to_anchor=[1, 0.8])
plt.show()

def get_infected_names(input_name):
    epsilon = 0.0018288  # a radial distance of 6 feet in kilometers
    model = DBSCAN(eps=epsilon, min_samples=2, metric='haversine').fit(df[['latitude', 'longitude']])
    df['cluster'] = model.labels_.tolist()

    input_name_clusters = []
    for i in range(len(df)):
        if df['id'][i] == input_name:
            if df['cluster'][i] in input_name_clusters:
                pass
            else:
                input_name_clusters.append(df['cluster'][i])

    infected_names = []
    for cluster in input_name_clusters:
        if cluster != -1:
            ids_in_cluster = df.loc[df['cluster'] == cluster, 'id']
            for i in range(len(ids_in_cluster)):
                member_id = ids_in_cluster.iloc[i]
                if (member_id not in infected_names) and (member_id != input_name):
                    infected_names.append(member_id)
                else:
                    pass
    return infected_names, model.labels_

# Example usage
infected_names, labels = get_infected_names('John')

# Plotting
fig = plt.figure(figsize=(12, 10))
sns.scatterplot(df['latitude'], df['longitude'], hue=['cluster-{}'.format(x + 1) for x in labels])
plt.legend(bbox_to_anchor=[1, 1])
plt.show()
