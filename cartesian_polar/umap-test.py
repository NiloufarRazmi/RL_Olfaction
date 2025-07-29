# from pathlib import Path
# import torch
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from matplotlib.ticker import NullFormatter
# import matplotlib as mpl
# from tqdm.auto import tqdm
# from scipy.cluster.hierarchy import dendrogram, linkage
# from sklearn.preprocessing import StandardScaler
# import umap
# from sklearn.cluster import KMeans
# import utils
# import sys
# import os
# import numpy as np
# from environment import CONTEXTS_LABELS
# #from agent import neural_network
# import seaborn as sns
# from agent import DQN
# import pandas as pd

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# save_path = Path("save")
# data_dir_lr = save_path / "7-8-LR"
# data_dir_ew = save_path / "7-8-EW"
# data_path_lr = data_dir_lr / "data.tar" 
# data_path_ew = data_dir_ew / "data.tar"
# data_dict_lr = torch.load(data_path_lr, weights_only=False, map_location=DEVICE)
# #data_dict_lr.keys()
# #data_dict_ew = torch.load(data_path_ew, weights_only=False, map_location=DEVICE)
# #data_dict_ew.keys()

# # ASSUMING EQUAL ARCHITECTURES
# parameters = data_dict_lr['p']
# n_observations = parameters.n_observations
# n_actions = parameters.n_actions
# n_units = parameters.n_hidden_units

# model = DQN(n_observations, n_actions, n_units)
# model_path = data_dir_lr / f'trained-agent-state-0.pt'
# model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
# model.eval()

# # Extracting Model Weights
# state_dict = model.state_dict()
# layer0_weights = state_dict['mlp.0.weight']
# layer1_weights = state_dict['mlp.1.weight']
# layer2_weights = state_dict['mlp.3.weight']
# layer3_weights = state_dict['mlp.5.weight']
# layer4_weights = state_dict['mlp.7.weight']
# weights = [layer0_weights,layer1_weights,layer2_weights,layer3_weights,layer4_weights]

# first_layer_weights = weights[0]
# dominant_inputs = np.argmax(np.abs(first_layer_weights), axis=1)  # shape: (512,)

# umap_embedding = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(first_layer_weights)

# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=dominant_inputs, cmap='tab20', s=15)
# plt.colorbar(scatter, ticks=range(21), label='Dominant Input Feature')
# #plt.title(f"First Layer t-SNE Perplexity {p}")
# #plt.savefig(f'figures/firstlayercosinetSNEp{p}.png', dpi=300, bbox_inches = 'tight')
# plt.show()

import umap
import numpy as np

X = np.random.rand(500, 20)
embedding = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(X)
print("Success:", embedding.shape)

