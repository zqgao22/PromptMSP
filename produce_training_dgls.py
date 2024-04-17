import torch
from dgl import save_graphs, load_graphs
# from protein_gnn import *
import random
import numpy as np
from utils import *
from tqdm import tqdm

all_chains_rep_list = torch.load('./source_data/chains_rep_list_3_5.pt')
complex_coor_gt = torch.load('./source_data/coor_gt_list_3_5.pt')
chain_all_list = torch.load('./source_data/chain_name_list_3_5.pt')

pdb_id = []
with open("./source_data/3_5_chain_pdb.txt", "r") as f: 
    for line in f.readlines():
        pdb_id.append(line[:-1])
# pi = 10
# all_chains_rep_list = all_chains_rep_list[:pi]
# complex_coor_gt = complex_coor_gt[:pi]
# chain_all_list = chain_all_list[:pi]
# pdb_id = pdb_id[:pi]
homo_list_final = []
for i in tqdm(range(len(all_chains_rep_list))):
    homo_single = []
    x = torch.stack(all_chains_rep_list[i])
    pairwised=torch.cosine_similarity(x.unsqueeze(1),x.unsqueeze(0),dim=-1)
    homo_list = torch.where(pairwised - torch.triu(torch.ones(len(all_chains_rep_list[i]),len(all_chains_rep_list[i])),diagonal=0) > 0.9999)
    for j in range(homo_list[0].shape[0]):
        n_diff = abs(complex_coor_gt[i][homo_list[0][j]][0].shape[0] - complex_coor_gt[i][homo_list[1][j]][0].shape[0])
        if n_diff <= 3:
            homo_single.append((homo_list[0][j],homo_list[1][j]))
    homo_list_final.append(homo_single)

def generate_single_graph(chain_num):
    node_list_all = list(range(chain_num))
    # node_list = random.sample(node_list_all, random.randint(2,chain_num))
    node_list = random.sample(node_list_all, chain_num)
    index_list_all = []
    exist_list = []
    initial_2_nodes = random.sample(node_list, 2)
    index_list_all.append(initial_2_nodes)
    exist_list = exist_list + initial_2_nodes
    remain_list = list(set(node_list).difference(set(np.unique(np.array(exist_list)).tolist())))
    if len(node_list) != 2:
        for i in range(len(node_list)-2):
            new_index = [random.sample(exist_list, 1)[0],random.sample(remain_list, 1)[0]]
            exist_list = np.unique(np.array(exist_list + new_index)).tolist()
            remain_list = list(set(node_list).difference(set(np.unique(np.array(exist_list)).tolist())))
            index_list_all.append(new_index)
    return torch.tensor(index_list_all).T


all_complex_list = []
pdb_index = []

# a = creating_dgls_all(len(complex_coor_gt[i]))
dgls_3_all = creating_dgls_all(3)
dgls_4_all = creating_dgls_all(4)
dgls_5_all = creating_dgls_all(5)

for i in tqdm(range(len(all_chains_rep_list))):
    chain_num = len(complex_coor_gt[i])
    if chain_num == 3:
        sample_all_list = dgls_3_all
    elif chain_num == 4:
        sample_all_list = dgls_4_all
    else:
        sample_all_list = dgls_5_all
    all_complex_list.append(sample_all_list)
    
print('Have processed',len(all_complex_list), 'multimers')  

print('Begin to generate input dgls')
graph_list = build_dgl(all_complex_list,all_chains_rep_list)
print('Have generated',len(graph_list), 'dgls')
save_graphs('./source_data/train_oracle_dgl_train_3_5.bin',graph_list)


print('Begin to compute labels (RMSD)')
label_list = build_label(all_complex_list, complex_coor_gt, chain_all_list, homo_list_final, pdb_id)
torch.save(label_list,'./source_data/rmsd_loss_train_3_5.pt')


