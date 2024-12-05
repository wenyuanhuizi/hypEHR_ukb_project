import networkx as nx
from gensim.models import Word2Vec
import random
import os


def seed_everything(seed=0):
    random.seed(seed)  
 
    
def read_and_preprocess(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    # edges = [line.split(',') for line in content.strip().split('\n')]
    edges= [line.replace(' ', '').split(',') for line in content.strip().split('\n')]
    G = nx.Graph()
    for edge in edges:
        for i in range(len(edge)):
            for j in range(i + 1, len(edge)):
                G.add_edge(edge[i], edge[j])
    return G

def perform_random_walks(graph, num_walks=10, walk_length=40):
    walks = []
    for node in list(graph.nodes):
        for _ in range(num_walks):
            walk = [node]
            while len(walk) < walk_length:
                cur = walk[-1]
                cur_neighbors = list(graph.neighbors(cur))
                if len(cur_neighbors) > 0:
                    walk.append(random.choice(cur_neighbors))
                else:
                    break
            walks.append(list(map(str, walk)))
    return walks

def generate_embeddings(walks, vector_size=128, window=5, min_count=0, sg=1, workers=4, epochs=10):
    model = Word2Vec(sentences=walks, vector_size=vector_size, window=window, min_count=min_count, sg=sg, workers=workers, epochs=epochs)
    return model

def save_embeddings_to_txt(model, file_path):
    vocab = list(model.wv.index_to_key)
    embeddings = model.wv[vocab]
    with open(file_path, 'w') as f:
        f.write(f"{len(vocab)} {model.vector_size}\n")
        for node, embedding in zip(vocab, embeddings):
            f.write(f"{node} {' '.join(map(str, embedding))}\n")

os.chdir('../data')   
seed_everything(seed=0)
file_path = 'hyperedges-ukb.txt'  
output_file_path = 'node-embeddings-ukb'  
G = read_and_preprocess(file_path)
walks = perform_random_walks(G)
model = generate_embeddings(walks)
save_embeddings_to_txt(model, output_file_path)