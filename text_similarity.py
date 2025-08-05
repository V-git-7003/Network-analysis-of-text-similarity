from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import networkx as nx
import numpy as np
from scipy.linalg import eigvalsh

# Add TensorFlow and USE
import tensorflow as tf
import tensorflow_hub as hub



csv_path = os.path.join('archive/cnn_dailymail/train.csv')  
df = pd.read_csv(csv_path)
docs = df['highlights'].dropna().tolist()[:100] 

# Using BERT
model_bert = SentenceTransformer('all-MiniLM-L6-v2')
embeddings_bert = model_bert.encode(docs)

# Using TF-IDF
vectorizer = TfidfVectorizer()
embeddings_tfidf = vectorizer.fit_transform(docs).toarray()


sim_matrix_bert = cosine_similarity(embeddings_bert)
sim_matrix_tfidf = cosine_similarity(embeddings_tfidf)


def build_graph(sim_matrix, threshold=0.5):
    G = nx.Graph()
    n = sim_matrix.shape[0]
    for i in range(n):
        G.add_node(i)
        for j in range(i + 1, n):
            weight = sim_matrix[i, j]
            if weight >= threshold:
                G.add_edge(i, j, weight=weight)
    return G

G_bert = build_graph(sim_matrix_bert)
G_tfidf = build_graph(sim_matrix_tfidf)


print(nx.density(G_bert), nx.density(G_tfidf))
print(nx.average_clustering(G_bert), nx.average_clustering(G_tfidf))


ged = nx.graph_edit_distance(G_bert, G_tfidf)
print("Graph Edit Distance:", ged)


adj_bert = nx.to_numpy_array(G_bert)
adj_tfidf = nx.to_numpy_array(G_tfidf)

eigs_bert = eigvalsh(adj_bert)
eigs_tfidf = eigvalsh(adj_tfidf)

import matplotlib.pyplot as plt
plt.plot(eigs_bert, label='BERT')
plt.plot(eigs_tfidf, label='TF-IDF')
plt.legend()
plt.show()
