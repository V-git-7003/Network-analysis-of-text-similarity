from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import networkx as nx
import numpy as np
from scipy.linalg import eigvalsh
from sentence_transformers import SentenceTransformer
from huggingface_hub import from_pretrained_keras
from matplotlib import pyplot as plt



class text_similarity:
    def __init__(self):
        
        # bert
        self.model_bert = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings_bert = self.model_bert.encode(docs)
        
        # tf-idf
        self.vectorizer = TfidfVectorizer()
        self.embeddings_tfidf = self.vectorizer.fit_transform(docs).toarray()
        
        # universal sentence encoder
        self.use_model = from_pretrained_keras("hfarwah/universal-sentence-encoder")
        self.embeddings_use = self.use_model(docs).numpy()
    
    def cosine_similarity_matrix(self):
        
        self.sim_matrix_bert = cosine_similarity(self.embeddings_bert)
        self.sim_matrix_tfidf = cosine_similarity(self.embeddings_tfidf)
        self.sim_matrix_use = cosine_similarity(self.embeddings_use)

    def build_graph(self,sim_matrix, threshold=0.5):
        G = nx.Graph()
        n = sim_matrix.shape[0]
        for i in range(n):
            G.add_node(i)
            for j in range(i + 1, n):
                weight = sim_matrix[i, j]
                if weight >= threshold:
                    G.add_edge(i, j, weight=weight)
        return G
    
    def compute_eigenvalues(self,G_bert,G_tfidf,G_use):
        adj_bert = nx.to_numpy_array(G_bert)
        adj_tfidf = nx.to_numpy_array(G_tfidf)
        adj_use = nx.to_numpy_array(G_use)

        self.eigs_bert = eigvalsh(adj_bert)
        self.eigs_tfidf = eigvalsh(adj_tfidf)
        self.eigs_use = eigvalsh(adj_use)

    def plot_eigenvalues(self):
        plt.plot(self.eigs_bert, label='BERT')
        plt.plot(self.eigs_use, label='USE')
        plt.plot(self.eigs_tfidf, label='TF-IDF')
        plt.legend()
        plt.show()
    
    def density(self,G):
        return nx.density(G)
    
    def average_clustering(self, G):
        return nx.average_clustering(G)
    
    def top_connected_nodes(self,G, top_n=10):
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [(i, degrees[i], docs[i]) for i, _ in top_nodes]
    
 

    def top_pairs(sim_matrix, docs, top_n=6):
        n = sim_matrix.shape[0]
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((sim_matrix[i, j], i, j))
        pairs = sorted(pairs, key=lambda x: x[0], reverse=True)[:top_n]
        return [(score, docs[i], docs[j]) for score, i, j in pairs]


    
    def spring_layout_visualization(self,G,title):
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)  # Layout for better visualization
        nx.draw(G, pos, node_size=50, with_labels=False, edge_color='gray')
        plt.title(title)
        plt.show()
    
    def graph_edit_distance(self,G1,G2):
        return nx.graph_edit_distance(G1, G2)

   
    def main(self,docs):

        # Compute cosine similarity matrices
        self.cosine_similarity_matrix()

        # Build graphs
        G_bert = self.build_graph(self.sim_matrix_bert)
        G_tfidf = self.build_graph(self.sim_matrix_tfidf)
        G_use = self.build_graph(self.sim_matrix_use)

        # Density and clustering
        density_bert = self.density(G_bert)
        density_tfidf = self.density(G_tfidf)
        density_use = self.density(G_use)
        avg_clustering_bert = self.average_clustering(G_bert)    
        avg_clustering_tfidf = self.average_clustering(G_tfidf)
        avg_clustering_use = self.average_clustering(G_use)
        
        print(f"BERT Graph Density: {density_bert}, Average Clustering: {avg_clustering_bert}")
        print(f"TF-IDF Graph Density: {density_tfidf}, Average Clustering: {avg_clustering_tfidf}")
        print(f"USE Graph Density: {density_use}, Average Clustering: {avg_clustering_use}")

        # plot eigenvalues
        self.compute_eigenvalues(G_bert,G_tfidf,G_use)
        self.plot_eigenvalues()

        # In main(), after building graphs and before visualization:
        print("\nTop 6 news pairs with highest cosine similarity (BERT):")
        for score, doc1, doc2 in top_pairs(self.sim_matrix_bert, docs):
            print(f"Score: {score:.4f}\nDoc 1: {doc1}\nDoc 2: {doc2}\n")

        print("\nTop 6 news pairs with highest cosine similarity (TF-IDF):")
        for score, doc1, doc2 in top_pairs(self.sim_matrix_tfidf, docs):
            print(f"Score: {score:.4f}\nDoc 1: {doc1}\nDoc 2: {doc2}\n")

        print("\nTop 6 news pairs with highest cosine similarity (USE):")
        for score, doc1, doc2 in top_pairs(self.sim_matrix_use, docs):
            print(f"Score: {score:.4f}\nDoc 1: {doc1}\nDoc 2: {doc2}\n")


        print("\nTop 10 news (BERT) with highest connections:")
        for idx, degree, text in self.top_connected_nodes(G_bert):
            print(f"Index: {idx}, Degree: {degree}\n{text}\n")

        print("\nTop 10 news (TF-IDF) with highest connections:")
        for idx, degree, text in self.top_connected_nodes(G_tfidf):
            print(f"Index: {idx}, Degree: {degree}\n{text}\n")

        print("\nTop 10 news (USE) with highest connections:")
        for idx, degree, text in self.top_connected_nodes(G_use):
            print(f"Index: {idx}, Degree: {degree}\n{text}\n")

        # Visualize graphs
        self.spring_layout_visualization(G_bert, "BERT Graph Visualization")
        self.spring_layout_visualization(G_tfidf, "TF-IDF Graph Visualization")
        self.spring_layout_visualization(G_use, "USE Graph Visualization")
    
if __name__ == "__main__":

    # Load data
    file_path = '/content/drive/MyDrive/Project_data/CNN_news_data/train.csv'
    df = pd.read_csv(file_path)
    docs = df['article'].dropna().tolist()[:4000] # limiting to 4000 docs (ram issues)

    # Initialize text_similarity class
    ts = text_similarity()

    ts.main(docs)