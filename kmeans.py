# -*- coding:utf-8 -*-


from sklearn.cluster import KMeans
from retrieval import load_feat_db
from sklearn.externals import joblib
from config import DATASET_BASE, N_CLUSTERS
import os


if __name__ == '__main__':
    feats, labels = load_feat_db()
    model = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_jobs=-1).fit(feats)
    model_path = os.path.join(DATASET_BASE, r'models', r'kmeans.m')
    joblib.dump(model, model_path)
