import pandas as pd
import numpy as np
import pickle

from rectools import Columns
from rectools.metrics import NDCG, Accuracy, MAP, Recall, MeanInvUserFreq, Serendipity
from sklearn.metrics import mean_squared_error
from rectools.dataset import Dataset

data_interim_dir = 'benchmark/data/'
user_groups = ['ua', 'ub']
data_splits = ['base', 'test']

datasets = {}

for user_group in user_groups:
    for split in data_splits:
        # Construct file paths
        interactions_path = f"{data_interim_dir}{user_group}.{split}.csv"
        user_features_path = f"{data_interim_dir}{user_group}.{split}_user_features.csv"
        item_features_path = f"{data_interim_dir}{user_group}.{split}_item_features.csv"

        # Read the data from CSV files
        interactions_df = pd.read_csv(interactions_path)
        user_features_df = pd.read_csv(user_features_path)
        item_features_df = pd.read_csv(item_features_path)

        dataset = Dataset.construct(
            interactions_df,
            user_features_df=user_features_df,
            cat_user_features=['gender', 'occupation'],  # If these were the categorical features
            item_features_df=item_features_df,
            make_dense_item_features=True  # If this is still applicable
        )

        # Store in the data dictionary
        if user_group not in datasets:
            datasets[user_group] = {}

        datasets[user_group][split] = (dataset, interactions_df)

with open("./models/best_model.pickle", 'rb') as f:
    best_model = pickle.load(f)

k = 10
ndcg = NDCG(k=k, log_base=3)
recall = Recall(k=k)
mmap = MAP(k=k)
seren = Serendipity(k=k)
miuf = MeanInvUserFreq(k=k)
results = []

for ug in datasets.keys():
    base_ds = datasets[ug]['base'][0]
    base_df = datasets[ug]['base'][1]
    test_ds = datasets[ug]['test'][0]
    test_df = datasets[ug]['test'][1]

    # Fit the model
    best_model.fit(base_ds)

    # Generate recommendations
    recs = best_model.recommend(
        users=test_df[Columns.User].unique(),
        dataset=base_ds,
        k=10,
        filter_viewed=True,
    )

    catalog = base_df[Columns.Item].unique()
    # Evaluate the model
    map_score = mmap.calc(reco=recs, interactions=test_df)
    recall_score = recall.calc(reco=recs, interactions=test_df)
    ndcg_score = ndcg.calc(reco=recs, interactions=test_df)
    seren_score = seren.calc(reco=recs, catalog=catalog, interactions=test_df, prev_interactions=base_df)
    miuf_score = miuf.calc(reco=recs, prev_interactions=base_df)

    # Calculate RMSE
    recs.rename(columns={Columns.Score: Columns.Weight}, inplace=True)
    merged_data = pd.merge(recs, test_df, on=[Columns.User, Columns.Item], suffixes=('_predicted', '_test'))
    rmse = np.sqrt(
        mean_squared_error(merged_data[Columns.Weight + '_test'], merged_data[Columns.Weight + '_predicted']))

    # Append results to the list
    results.append({
        'User Group': ug,
        'MAP': map_score,
        'Recall': recall_score,
        'NDCG': ndcg_score,
        'Serendipity': seren_score,
        'MIUF': miuf_score,
        'RMSE': rmse,
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save to CSV

average_metrics = results_df.mean().reset_index()
average_metrics.columns = ['Metric', 'value']
print(average_metrics)