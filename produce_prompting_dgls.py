import torch
from dgl import save_graphs, load_graphs
# from protein_gnn import *
import random
import numpy as np
from utils_target import *
from tqdm import tqdm
from itertools import combinations
import argparse

parser = argparse.ArgumentParser(description='target data creation', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-data_fraction', type=float, default=1.0, help='One can choose to process only a part of the data.')
args = parser.parse_args().__dict__

def assemble(ind, docking_path,complex_coor_gt, chain_all_list, pdb_id):
    unique_docking_path = torch.cat((docking_path[0][0].unsqueeze(0).unsqueeze(0),docking_path[1,:].unsqueeze(0)),dim=1).squeeze(0)
    all_coor_list = []
    root_path = './dimer_gt/' + pdb_id[ind]

    coor_gt = complex_coor_gt[ind]
    chain_list = chain_all_list[ind]
    full_path_1 = root_path + '_' + chain_list[docking_path[:,0][0]] + '_' + chain_list[docking_path[:,0][1]] + '.npy'
    full_path_2 = root_path + '_' + chain_list[docking_path[:,0][1]] + '_' + chain_list[docking_path[:,0][0]] + '.npy'
    if os.path.isfile(full_path_1):
        coor_complex = np.load(full_path_1,allow_pickle=True)
        chain_1, chain_2 = coor_complex[0].astype(np.float32),coor_complex[1].astype(np.float32)
    else:
        coor_complex = np.load(full_path_2,allow_pickle=True)
        chain_1, chain_2 = coor_complex[1].astype(np.float32),coor_complex[0].astype(np.float32)
    all_coor_list.append(chain_1)
    all_coor_list.append(chain_2)
    for i in range(docking_path.size(1)-1):
        # new_chain_path = docking_path[:,i+1][1]
        # exist_chain_path = docking_path[:,i+1][0]
        full_path_1 = root_path + '_' + chain_list[docking_path[:,i+1][0]] + '_' + chain_list[docking_path[:,i+1][1]] + '.npy'
        full_path_2 = root_path + '_' + chain_list[docking_path[:,i+1][1]] + '_' + chain_list[docking_path[:,i+1][0]] + '.npy'
        if os.path.isfile(full_path_1):
            coor_complex = np.load(full_path_1,allow_pickle=True)
            new_chain_coor, exist_chain_coor = coor_complex[1].astype(np.float32),coor_complex[0].astype(np.float32)
        else:
            coor_complex = np.load(full_path_2,allow_pickle=True)
            new_chain_coor, exist_chain_coor = coor_complex[0].astype(np.float32),coor_complex[1].astype(np.float32)
        exist_docked_chain_coor = all_coor_list[torch.where(unique_docking_path == docking_path[:,i+1][0])[0]]
        r,t = find_rigid_alignment(torch.tensor(exist_chain_coor).cuda(), torch.tensor(exist_docked_chain_coor).cuda())
        new_chain_coor_trans = (r @ torch.tensor(new_chain_coor).T.cuda()).T + t
        all_coor_list.append(np.array(new_chain_coor_trans.cpu()))
    return all_coor_list, unique_docking_path
def build_loss(i,adj, complex_coor_gt, chain_all_list, pdb_id):  
    rmsd_loss_single = []
    all_coor_list, unique_docking_path = assemble(i,adj,complex_coor_gt,chain_all_list, pdb_id)
    all_coor_list_gt = np.array(complex_coor_gt[i],dtype = object)[unique_docking_path]
    all_coor_gt = all_coor_list_gt[0][0]
    all_coor_pred = all_coor_list[0]
    for ii in range(all_coor_list_gt.shape[0]-1):
        all_coor_gt = np.concatenate((all_coor_gt, all_coor_list_gt[ii+1][0]), axis=0).astype(np.float32)
        all_coor_pred = np.concatenate((all_coor_pred, all_coor_list[ii+1]), axis=0).astype(np.float32)
    R0,T0 = find_rigid_alignment(torch.tensor(all_coor_gt), torch.tensor(all_coor_pred))
    rmsd_loss_begin = torch.sqrt((((R0.mm(torch.tensor(all_coor_gt).T)).T + T0 - torch.tensor(all_coor_pred))**2).sum(axis=1).mean())
    rmsd_loss_single.append(rmsd_loss_begin)

    for k in range(len(homo_list_final[i])):
        a = torch.clone(adj)
        a1 = torch.where(adj == homo_list_final[i][k][0])
        a2 = torch.where(adj == homo_list_final[i][k][1])
        if not a1[0].numel() == 0 and not a2[0].numel() == 0:
            a[a1],a[a2] = a[a2][0],a[a1][0]
            all_coor_list, _ = assemble(i,a,complex_coor_gt,chain_all_list, pdb_id)
            all_coor_list_gt = np.array(complex_coor_gt[i],dtype = object)[unique_docking_path]
            all_coor_gt = all_coor_list_gt[0][0]
            all_coor_pred = all_coor_list[0]
            for ii in range(all_coor_list_gt.shape[0]-1):
                all_coor_gt = np.concatenate((all_coor_gt, all_coor_list_gt[ii+1][0]), axis=0).astype(np.float32)
                all_coor_pred = np.concatenate((all_coor_pred, all_coor_list[ii+1]), axis=0).astype(np.float32)
            R0,T0 = find_rigid_alignment(torch.tensor(all_coor_gt).cuda(), torch.tensor(all_coor_pred).cuda())
            rmsd_loss = torch.sqrt((((R0.mm(torch.tensor(all_coor_gt).cuda().T)).T + T0 - torch.tensor(all_coor_pred).cuda())**2).sum(axis=1).mean())
            rmsd_loss_single.append(rmsd_loss.cpu())
    rmsd_loss_min = min(rmsd_loss_single)

    return rmsd_loss_min

def generate_single_graph(chain_num,num):
    node_list_all = list(range(chain_num))
    node_list = random.sample(node_list_all, num)
    # node_list = random.sample(node_list_all, 5)
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
    
def generate_single_graph_pro(adj,chain_num):
    node_list_all = list(range(chain_num))
    node_list = random.sample(node_list_all, chain_num)
    index_list_all = np.array(adj.T).tolist()
    exist_list = np.unique(np.array(adj)).tolist()
    remain_list = list(set(node_list).difference(set(np.unique(np.array(exist_list)).tolist())))
    new_index = [random.sample(exist_list, 1)[0],random.sample(remain_list, 1)[0]]
    index_list_all.append(new_index)
    return torch.tensor(index_list_all).T

complex_coor_gt = torch.load('./source_data/coor_gt_list_train.pt')
chain_all_list = torch.load('./source_data/chain_name_list_train.pt')
ebd_chain_list = torch.load('./source_data/chains_rep_list_train.pt')




pdb_id = []
# with open("./PDB-M/PDB-M-test.txt", "r") as f:  # 打开文件
#     for line in f.readlines():
#         pdb_id.append(line[:-1])
with open("./PDB-M/PDB-M-train.txt", "r") as f:  # 打开文件
    for line in f.readlines():
        pdb_id.append(line[:-1])
print('total number:', len(pdb_id))
all_multimer_list = []

# num_s = 10
# pdb_id = pdb_id[:num_s]
# # complex_coor_gt = torch.load('./PDB-M/coor_gt_list_test.pt')
# # chain_all_list = torch.load('./PDB-M/chain_name_list_test.pt')
# # ebd_chain_list = torch.load('./PDB-M/chains_rep_list_test.pt')
# complex_coor_gt = complex_coor_gt[:num_s]
# chain_all_list = chain_all_list[:num_s]
# ebd_chain_list = ebd_chain_list[:num_s]

homo_list_final = []
for i in tqdm(range(len(ebd_chain_list))):
    homo_single = []
    x = torch.stack(ebd_chain_list[i])
    pairwised=torch.cosine_similarity(x.unsqueeze(1),x.unsqueeze(0),dim=-1)
    homo_list = torch.where(pairwised - torch.triu(torch.ones(len(ebd_chain_list[i]),len(ebd_chain_list[i])),diagonal=0) > 0.9999)
    for j in range(homo_list[0].shape[0]):
        n_diff = abs(complex_coor_gt[i][homo_list[0][j]][0].shape[0] - complex_coor_gt[i][homo_list[1][j]][0].shape[0])
        if n_diff <= 3:
            homo_single.append((homo_list[0][j],homo_list[1][j]))
    homo_list_final.append(homo_single)


#########################discover completely correct assembly graphs###########################
label_list = []
for i in range(len(pdb_id)):
    if len(chain_all_list[i]) <= 10:
        print('processing',i,'-th multimer, phase 1')
        all_adj_list = []
        all_loss_list = []
        # for start in tqdm(list(range(len(chain_all_list[i])))):
        for start in tqdm(list(range(1))):
            docked_chains = [start]
            remain_chains = list(set(list(range(len(chain_all_list[i])))) - set(docked_chains))
            adj = torch.tensor([[],[]])
            all_loss = []
            all_adj = []
            all_pair_tmp = []
            pro_times = 1
            if pro_times == 1:
                for d_chain in docked_chains:
                    for u_chain in remain_chains:
                        adj_tmp = torch.cat((adj,torch.tensor([[d_chain],[u_chain]])),1).int()
                        loss = build_loss(i,adj_tmp, complex_coor_gt, chain_all_list, pdb_id)
                        all_loss.append(loss)
                        all_adj.append(adj_tmp)
                
                best_now_ind_list = np.where(np.array(all_loss) < 0.0001)[0]
                intera_adj = [all_adj[best_ind] for best_ind in best_now_ind_list.tolist()]
                # for ik in range(len(all_loss_tmp)):
                #     if ik == best_now_ind:
                #         all_loss_tmp[ik] = 0
                #     else:
                #         all_loss_tmp[ik] = 1
                ### get all_adj and all_best_adj ###
            all_adj_tmp, all_loss_tmp = [], []
            while pro_times != len(chain_all_list[i]) -1:
                pro_times += 1
                if pro_times > 4:
                    intera_adj = intera_adj[:20]
                print('pro_time:',pro_times)
                for adj_single in tqdm(intera_adj):
                    docked_chains = torch.cat((adj_single[0][0].unsqueeze(0).unsqueeze(0),adj_single[1,:].unsqueeze(0)),dim=1).squeeze(0).numpy().tolist()
                    remain_chains = list(set(range(len(chain_all_list[i]))) - set(docked_chains))
                    for d_chain in docked_chains:
                        for u_chain in remain_chains:
                            all_adj_tmp.append(torch.cat((adj_single,torch.tensor([[d_chain],[u_chain]])),1).int())
                            all_loss_tmp.append(build_loss(i,torch.cat((adj_single,torch.tensor([[d_chain],[u_chain]])),1).int(), \
                                                        complex_coor_gt, chain_all_list, pdb_id))
                best_now_ind_list = np.where(np.array(all_loss_tmp) < 0.0001)[0]
                intera_adj = [all_adj_tmp[best_ind] for best_ind in best_now_ind_list.tolist()]
                all_loss  += all_loss_tmp
                all_adj += all_adj_tmp
                all_adj_tmp, all_loss_tmp = [], []
            
            all_adj_list = all_adj_list + all_adj
            all_loss_list = all_loss_list + all_loss

        all_multimer_list.append(all_adj_list)
        label_list = label_list + all_loss_list
    else:
        all_multimer_list.append([])
graph_list = build_dgl(all_multimer_list,ebd_chain_list)
# label_list = build_label(all_all_adj, complex_coor_gt, chain_all_list, pdb_id)
print('Finally get ',len(label_list),'samples')
save_graphs('./target_data/train_prompt_dgls.bin',graph_list)
torch.save(label_list,'./target_data/train_prompt_rmsd.pt')
