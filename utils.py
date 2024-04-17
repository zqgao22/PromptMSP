import numpy as np
import torch
import os
import dgl
import torch.nn as nn
import scipy.spatial as spa
from tqdm import tqdm
from random import sample
import itertools
from tmtools import tm_align
import time
def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float)
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
        >>> B *= torch.tensor([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()

def assemble_rmsd_for_inference(pdb_id, docking_path,chain_list, complex_coor_gt,chains_rep):
    docking_path = docking_path.long()
    unique_docking_path = torch.cat((docking_path[0][0].unsqueeze(0).unsqueeze(0),docking_path[1,:].unsqueeze(0)),dim=1).squeeze(0)
    all_coor_list = []
    root_path = './dimer_gt/' + pdb_id

    full_path_1 = root_path + '_' + chain_list[docking_path[:,0][0]] + '_' + chain_list[docking_path[:,0][1]] + '.npy'
    full_path_2 = root_path + '_' + chain_list[docking_path[:,0][1]] + '_' + chain_list[docking_path[:,0][0]] + '.npy'

    if os.path.isfile(full_path_1):
        coor_complex = np.load(full_path_1,allow_pickle=True)
        chain_1, chain_2 = coor_complex[0],coor_complex[1]
    else:
        coor_complex = np.load(full_path_2,allow_pickle=True)
        chain_1, chain_2 = coor_complex[1],coor_complex[0]
        
    all_coor_list.append(chain_1)
    all_coor_list.append(chain_2)

    for i in range(docking_path.size(1)-1):
        # if chains_rep[docking_path[:,i+1][0]].equal(chains_rep[docking_path[:,i+1][1]]):
        if 1 == 2:
            print('there are 2 same chains being docking.')
            full_path_1 = root_path + '_' + chain_list[docking_path[:,i+1][0]] + '_' + chain_list[docking_path[:,i+1][1]] + '.npy'
            full_path_2 = root_path + '_' + chain_list[docking_path[:,i+1][1]] + '_' + chain_list[docking_path[:,i+1][0]] + '.npy'
            if os.path.isfile(full_path_1):
                coor_complex = np.load(full_path_1,allow_pickle=True)
                new_chain_coor_case_1, exist_chain_coor_case_1 = coor_complex[1],coor_complex[0]
                new_chain_coor_case_2, exist_chain_coor_case_2 = coor_complex[0],coor_complex[1]
            else:
                coor_complex = np.load(full_path_2,allow_pickle=True)
                new_chain_coor_case_1, exist_chain_coor_case_1 = coor_complex[0],coor_complex[1]
                new_chain_coor_case_2, exist_chain_coor_case_2 = coor_complex[1],coor_complex[0]
            exist_docked_chain_coor = all_coor_list[torch.where(unique_docking_path == docking_path[:,i+1][0])[0]]
            r_case_1,t_case_1 = find_rigid_alignment(torch.tensor(exist_chain_coor_case_1), torch.tensor(exist_docked_chain_coor))
            new_chain_coor_trans_case_1 = (r_case_1 @ new_chain_coor_case_1.T).T + t_case_1
            r_case_2,t_case_2 = find_rigid_alignment(torch.tensor(exist_chain_coor_case_2), torch.tensor(exist_docked_chain_coor))
            new_chain_coor_trans_case_2 = (r_case_2 @ new_chain_coor_case_2.T).T + t_case_2
            all_coor_now = all_coor_list[0]
            for ii in range(len(all_coor_list)-1):
                all_coor_now = np.concatenate((all_coor_now, all_coor_list[ii+1]), axis=0)
            if np.sum(spa.distance.cdist(all_coor_now, new_chain_coor_trans_case_1)<2) > np.sum(spa.distance.cdist(all_coor_now, new_chain_coor_trans_case_2)<2) :
                all_coor_list.append(new_chain_coor_trans_case_2)
            else:
                all_coor_list.append(new_chain_coor_trans_case_1)
        else:
            full_path_1 = root_path + '_' + chain_list[docking_path[:,i+1][0]] + '_' + chain_list[docking_path[:,i+1][1]] + '.npy'
            full_path_2 = root_path + '_' + chain_list[docking_path[:,i+1][1]] + '_' + chain_list[docking_path[:,i+1][0]] + '.npy'
            if os.path.isfile(full_path_1):
                coor_complex = np.load(full_path_1,allow_pickle=True)
                new_chain_coor, exist_chain_coor = coor_complex[1],coor_complex[0]
            else:
                coor_complex = np.load(full_path_2,allow_pickle=True)
                new_chain_coor, exist_chain_coor = coor_complex[0],coor_complex[1]
            exist_docked_chain_coor = all_coor_list[torch.where(unique_docking_path == docking_path[:,i+1][0])[0]]
            r,t = find_rigid_alignment(torch.tensor(exist_chain_coor), torch.tensor(exist_docked_chain_coor))
            new_chain_coor_trans = (r @ new_chain_coor.T).T + t
            all_coor_list.append(new_chain_coor_trans)

    all_coor_list_gt = np.array(complex_coor_gt)[unique_docking_path]
    all_coor_gt = all_coor_list_gt[0][0]
    all_coor_pred = all_coor_list[0]
    for ii in range(all_coor_list_gt.shape[0]-1):
        all_coor_gt = np.concatenate((all_coor_gt, all_coor_list_gt[ii+1][0]), axis=0)
        all_coor_pred = np.concatenate((all_coor_pred, all_coor_list[ii+1]), axis=0)
    R0,T0 = find_rigid_alignment(torch.tensor(all_coor_gt), torch.tensor(all_coor_pred))
    rmsd_inference = torch.sqrt((((R0.mm(torch.tensor(all_coor_gt).T)).T + T0 - torch.tensor(all_coor_pred))**2).sum(axis=1).mean())
    N_res = all_coor_gt.shape[0]
    eta = 1e-4
    d0 = 1.24 * pow((N_res-15),1/3) - 1.8
    R0_refine = torch.tensor(R0, requires_grad=True)
    T0_refine = torch.tensor(T0, requires_grad=True)
    for iter in range(10000):
        tm_1 = R0_refine.mm(torch.tensor(all_coor_gt).T).T + T0_refine
        tm_2 = torch.tensor(all_coor_pred)
        pdist = nn.PairwiseDistance(p=2)
        tmscore_loss = -(1 / ((pdist(tm_1, tm_2) / d0)**2 + 1)).mean()
        tmscore_loss.backward()
        R0_refine = R0_refine - eta * R0_refine.grad.detach()
        R0_refine = torch.tensor(R0_refine, requires_grad=True)
        T0_refine = T0_refine - eta * T0_refine.grad.detach()
        T0_refine = torch.tensor(T0_refine, requires_grad=True)
    return rmsd_inference, -tmscore_loss


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.tensor(labels)

def collate_target(samples):
    nodes_emb, labels = map(list, zip(*samples))
    return torch.cat([nodes_emb[i] for i in range(len(nodes_emb))] ,0), torch.tensor(labels)

def build_dgl_single_complex(complex_rep, new_edge_index, map_dic):
    g = dgl.DGLGraph((np.array(new_edge_index).tolist()[0],np.array(new_edge_index).tolist()[1]))
    g.ndata['features'] = torch.zeros((g.num_nodes(),13))
    for i in range(len(map_dic[0])):
        g.ndata['features'][i] = complex_rep[map_dic[0][i]]
    return g

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

def re_number(e_i):
    edge_index = torch.clone(e_i)
    original_list = torch.reshape(edge_index,(1,edge_index.size(0)*edge_index.size(1))).squeeze().unique().tolist()
    map_dic = [original_list, list(range(len(original_list)))]
    for i in range(len(original_list)):
        edge_index[edge_index == original_list[i]] = map_dic[1][i]
    return edge_index, map_dic

def build_dgl(edge_index_all,chain_rep_list):
    graph_list = []
    for i in tqdm(range(len(edge_index_all))):
        complex_rep = chain_rep_list[i]
        edge_index = edge_index_all[i]
        for j in range(len(edge_index)):
            new_edge_index, map_dic = re_number(edge_index[j])
            single_g = build_dgl_single_complex(complex_rep, new_edge_index, map_dic)
            graph_list.append(dgl.add_self_loop(single_g))
    
    return graph_list

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
        exist_chain_coor = exist_chain_coor
        # print(ind)
        r,t = find_rigid_alignment(torch.tensor(exist_chain_coor), torch.tensor(exist_docked_chain_coor))
        new_chain_coor_trans = (r @ new_chain_coor.T).T + t
        all_coor_list.append(np.array(new_chain_coor_trans))
    return all_coor_list, unique_docking_path


def assemble_esmfold(ind, docking_path,complex_coor_gt, chain_all_list, pdb_id):
    unique_docking_path = torch.cat((docking_path[0][0].unsqueeze(0).unsqueeze(0),docking_path[1,:].unsqueeze(0)),dim=1).squeeze(0)
    all_coor_list = []
    root_path = './dimer_esmfold/' + pdb_id[ind]

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
        exist_chain_coor = exist_chain_coor
        # print(ind)
        r,t = find_rigid_alignment(torch.tensor(exist_chain_coor), torch.tensor(exist_docked_chain_coor))
        new_chain_coor_trans = (r @ new_chain_coor.T).T + t
        all_coor_list.append(np.array(new_chain_coor_trans))
    return all_coor_list, unique_docking_path


def build_label_single_complex(i, edge_index_all, complex_coor_gt, chain_all_list, pdb_id):
    rmsd_loss_list = []
    edge_index_list = edge_index_all
    for j in tqdm(range(len(edge_index_list))):
        all_coor_list, unique_docking_path = assemble(i,edge_index_list[j],complex_coor_gt,chain_all_list, pdb_id)
        all_coor_list_gt = np.array(complex_coor_gt[i])[unique_docking_path]
        all_coor_gt = all_coor_list_gt[0][0]
        all_coor_pred = all_coor_list[0]
        for ii in range(all_coor_list_gt.shape[0]-1):
            all_coor_gt = np.concatenate((all_coor_gt, all_coor_list_gt[ii+1][0]), axis=0)
            all_coor_pred = np.concatenate((all_coor_pred, all_coor_list[ii+1]), axis=0)
        R0,T0 = find_rigid_alignment(torch.tensor(all_coor_gt), torch.tensor(all_coor_pred))
        rmsd_loss = torch.sqrt((((R0.mm(torch.tensor(all_coor_gt).T)).T + T0 - torch.tensor(all_coor_pred))**2).sum(axis=1).mean())
        rmsd_loss_list.append(rmsd_loss)
    return rmsd_loss_list
def build_dgl_inference(edge_index,complex_rep):
    new_edge_index, map_dic = re_number(edge_index)
    single_g = build_dgl_single_complex(complex_rep, new_edge_index, map_dic)
    return single_g

def split_test(chain_all_list):
    num_3 = []
    num_4 = []
    num_5 = []
    num_6 = []

    for i in tqdm(range(len(chain_all_list))):
        if len(chain_all_list[i]) == 3:
            num_3.append(i)
        elif len(chain_all_list[i]) == 4:
            num_4.append(i)
        elif len(chain_all_list[i]) == 5:
            num_5.append(i)
        elif len(chain_all_list[i]) == 6:
            num_6.append(i)
    test_3 = sample(num_3,20)
    test_4 = sample(num_4,20)
    test_5 = sample(num_5,20)
    test_6 = sample(num_6,20)
    return(test_3+test_4+test_5+test_6)



def assemble_rmsd_for_inference_with_coor(pdb_id, docking_path,chain_list, complex_coor_gt,chains_rep):
    docking_path = docking_path.long()
    unique_docking_path = torch.cat((docking_path[0][0].unsqueeze(0).unsqueeze(0),docking_path[1,:].unsqueeze(0)),dim=1).squeeze(0)
    all_coor_list = []
    root_path = './dimers_gt/' + pdb_id

    full_path_1 = root_path + '_' + chain_list[docking_path[:,0][0]] + '_' + chain_list[docking_path[:,0][1]] + '.npy'
    full_path_2 = root_path + '_' + chain_list[docking_path[:,0][1]] + '_' + chain_list[docking_path[:,0][0]] + '.npy'

    if os.path.isfile(full_path_1):
        coor_complex = np.load(full_path_1,allow_pickle=True)
        chain_1, chain_2 = coor_complex[0],coor_complex[1]
    else:
        coor_complex = np.load(full_path_2,allow_pickle=True)
        chain_1, chain_2 = coor_complex[1],coor_complex[0]
        
    all_coor_list.append(chain_1)
    all_coor_list.append(chain_2)

    for i in range(docking_path.size(1)-1):
        # if chains_rep[docking_path[:,i+1][0]].equal(chains_rep[docking_path[:,i+1][1]]):
        if 1 == 2:
            print('there are 2 same chains being docking.')
            full_path_1 = root_path + '_' + chain_list[docking_path[:,i+1][0]] + '_' + chain_list[docking_path[:,i+1][1]] + '.npy'
            full_path_2 = root_path + '_' + chain_list[docking_path[:,i+1][1]] + '_' + chain_list[docking_path[:,i+1][0]] + '.npy'
            if os.path.isfile(full_path_1):
                coor_complex = np.load(full_path_1,allow_pickle=True)
                new_chain_coor_case_1, exist_chain_coor_case_1 = coor_complex[1],coor_complex[0]
                new_chain_coor_case_2, exist_chain_coor_case_2 = coor_complex[0],coor_complex[1]
            else:
                coor_complex = np.load(full_path_2,allow_pickle=True)
                new_chain_coor_case_1, exist_chain_coor_case_1 = coor_complex[0],coor_complex[1]
                new_chain_coor_case_2, exist_chain_coor_case_2 = coor_complex[1],coor_complex[0]
            exist_docked_chain_coor = all_coor_list[torch.where(unique_docking_path == docking_path[:,i+1][0])[0]]
            r_case_1,t_case_1 = find_rigid_alignment(torch.tensor(exist_chain_coor_case_1), torch.tensor(exist_docked_chain_coor))
            new_chain_coor_trans_case_1 = (r_case_1 @ new_chain_coor_case_1.T).T + t_case_1
            r_case_2,t_case_2 = find_rigid_alignment(torch.tensor(exist_chain_coor_case_2), torch.tensor(exist_docked_chain_coor))
            new_chain_coor_trans_case_2 = (r_case_2 @ new_chain_coor_case_2.T).T + t_case_2
            all_coor_now = all_coor_list[0]
            for ii in range(len(all_coor_list)-1):
                all_coor_now = np.concatenate((all_coor_now, all_coor_list[ii+1]), axis=0)
            if np.sum(spa.distance.cdist(all_coor_now, new_chain_coor_trans_case_1)<2) > np.sum(spa.distance.cdist(all_coor_now, new_chain_coor_trans_case_2)<2) :
                all_coor_list.append(new_chain_coor_trans_case_2)
            else:
                all_coor_list.append(new_chain_coor_trans_case_1)
        else:
            full_path_1 = root_path + '_' + chain_list[docking_path[:,i+1][0]] + '_' + chain_list[docking_path[:,i+1][1]] + '.npy'
            full_path_2 = root_path + '_' + chain_list[docking_path[:,i+1][1]] + '_' + chain_list[docking_path[:,i+1][0]] + '.npy'
            if os.path.isfile(full_path_1):
                coor_complex = np.load(full_path_1,allow_pickle=True)
                new_chain_coor, exist_chain_coor = coor_complex[1],coor_complex[0]
            else:
                coor_complex = np.load(full_path_2,allow_pickle=True)
                new_chain_coor, exist_chain_coor = coor_complex[0],coor_complex[1]
            exist_docked_chain_coor = all_coor_list[torch.where(unique_docking_path == docking_path[:,i+1][0])[0]]
            r,t = find_rigid_alignment(torch.tensor(exist_chain_coor), torch.tensor(exist_docked_chain_coor))
            new_chain_coor_trans = (r @ new_chain_coor.T).T + t
            all_coor_list.append(new_chain_coor_trans)

    all_coor_list_gt = np.array(complex_coor_gt)[unique_docking_path]
    all_coor_gt = all_coor_list_gt[0][0]
    all_coor_pred = all_coor_list[0]
    for ii in range(all_coor_list_gt.shape[0]-1):
        all_coor_gt = np.concatenate((all_coor_gt, all_coor_list_gt[ii+1][0]), axis=0)
        all_coor_pred = np.concatenate((all_coor_pred, all_coor_list[ii+1]), axis=0)
    R0,T0 = find_rigid_alignment(torch.tensor(all_coor_gt), torch.tensor(all_coor_pred))
    rmsd_inference = torch.sqrt((((R0.mm(torch.tensor(all_coor_gt).T)).T + T0 - torch.tensor(all_coor_pred))**2).sum(axis=1).mean())
    N_res = all_coor_gt.shape[0]
    eta = 1e-4
    d0 = 1.24 * pow((N_res-15),1/3) - 1.8
    R0_refine = torch.tensor(R0, requires_grad=True)
    T0_refine = torch.tensor(T0, requires_grad=True)
    for iter in range(8000):
        tm_1 = R0_refine.mm(torch.tensor(all_coor_gt).T).T + T0_refine
        tm_2 = torch.tensor(all_coor_pred)
        pdist = nn.PairwiseDistance(p=2)
        tmscore_loss = -(1 / ((pdist(tm_1, tm_2) / d0)**2 + 1)).mean()
        tmscore_loss.backward()
        R0_refine = R0_refine - eta * R0_refine.grad.detach()
        R0_refine = torch.tensor(R0_refine, requires_grad=True)
        T0_refine = T0_refine - eta * T0_refine.grad.detach()
        T0_refine = torch.tensor(T0_refine, requires_grad=True)
    comb_coor = list(itertools.combinations(all_coor_list, 2))
    clash_sum = 0
    for comb in comb_coor:
        clash_sum += np.sum(spa.distance.cdist(comb[0], comb[1])<2)
    return rmsd_inference, -tmscore_loss, clash_sum

def build_label_tmscore(edge_index_all, complex_coor_gt, chain_all_list, pdb_id):
    tmscore_loss_list = []
    for i in tqdm(range(len(edge_index_all))):
        edge_index_list = edge_index_all[i]
        for j in tqdm(range(len(edge_index_list))):
            all_coor_list, unique_docking_path = assemble(i,edge_index_list[j],complex_coor_gt,chain_all_list, pdb_id)
            all_coor_list_gt = np.array(complex_coor_gt[i],dtype = object)[unique_docking_path]
            all_coor_gt = all_coor_list_gt[0][0]
            all_coor_pred = all_coor_list[0]
            for ii in range(all_coor_list_gt.shape[0]-1):
                all_coor_gt = np.concatenate((all_coor_gt, all_coor_list_gt[ii+1][0]), axis=0)
                all_coor_pred = np.concatenate((all_coor_pred, all_coor_list[ii+1]), axis=0)
            seq1 = "Q" * all_coor_gt.shape[0]
            seq2 = seq1
            # t1 = time.time()
            tmscore = tm_align(all_coor_gt, all_coor_pred, seq1, seq2).tm_norm_chain1
            # t2 = time.time()
            # print(t2-t1)
            tmscore_loss_list.append(tmscore)
    return tmscore_loss_list

# def build_label(edge_index_all, complex_coor_gt, chain_all_list, pdb_id):
#     rmsd_loss_list = []
#     for i in tqdm(range(len(edge_index_all))):
#         edge_index_list = edge_index_all[i]
#         for j in range(len(edge_index_list)):
#             all_coor_list, unique_docking_path = assemble(i,edge_index_list[j],complex_coor_gt,chain_all_list, pdb_id)
#             all_coor_list_gt = np.array(complex_coor_gt[i],dtype = object)[unique_docking_path]
#             all_coor_gt = all_coor_list_gt[0][0]
#             all_coor_pred = all_coor_list[0]
#             for ii in range(all_coor_list_gt.shape[0]-1):
#                 all_coor_gt = np.concatenate((all_coor_gt, all_coor_list_gt[ii+1][0]), axis=0).astype(np.float32)
#                 all_coor_pred = np.concatenate((all_coor_pred, all_coor_list[ii+1]), axis=0).astype(np.float32)
#             R0,T0 = find_rigid_alignment(torch.tensor(all_coor_gt).cuda(), torch.tensor(all_coor_pred).cuda())
#             rmsd_loss = torch.sqrt((((R0.mm(torch.tensor(all_coor_gt).cuda().T)).T + T0 - torch.tensor(all_coor_pred).cuda())**2).sum(axis=1).mean())
#             rmsd_loss_list.append(rmsd_loss.cpu())
#     return rmsd_loss_list

def build_label(edge_index_all, complex_coor_gt, chain_all_list, homo_list_final, pdb_id):
    rmsd_loss_list = []
    for i in tqdm(range(len(edge_index_all))):
        edge_index_list = edge_index_all[i]
        for j in range(len(edge_index_list)):
            rmsd_loss_single = []
            all_coor_list, unique_docking_path = assemble(i,edge_index_list[j],complex_coor_gt,chain_all_list, pdb_id)
            all_coor_list_gt = np.array(complex_coor_gt[i],dtype = object)[unique_docking_path]
            all_coor_gt = all_coor_list_gt[0][0]
            all_coor_pred = all_coor_list[0]
            for ii in range(all_coor_list_gt.shape[0]-1):
                all_coor_gt = np.concatenate((all_coor_gt, all_coor_list_gt[ii+1][0]), axis=0).astype(np.float32)
                all_coor_pred = np.concatenate((all_coor_pred, all_coor_list[ii+1]), axis=0).astype(np.float32)
            R0,T0 = find_rigid_alignment(torch.tensor(all_coor_gt).cuda(), torch.tensor(all_coor_pred).cuda())
            rmsd_loss = torch.sqrt((((R0.mm(torch.tensor(all_coor_gt).cuda().T)).T + T0 - torch.tensor(all_coor_pred).cuda())**2).sum(axis=1).mean())
            rmsd_loss_single.append(rmsd_loss)

            for k in range(len(homo_list_final[i])):
                a = torch.clone(edge_index_list[j])
                a1 = torch.where(edge_index_list[j] == homo_list_final[i][k][0])
                a2 = torch.where(edge_index_list[j] == homo_list_final[i][k][1])
                # print('a:',a,'a1:',a1,'a2:',a2)
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
                rmsd_loss_single.append(rmsd_loss)
            rmsd_loss_min = min(rmsd_loss_single)
            rmsd_loss_list.append(rmsd_loss_min.cpu())
    return rmsd_loss_list

def build_loss_inf(i,adj, complex_coor_gt, chain_all_list, pdb_id):  
    all_coor_list, unique_docking_path = assemble(i,adj,complex_coor_gt,chain_all_list, pdb_id)
    all_coor_list_gt = np.array(complex_coor_gt[i],dtype = object)[unique_docking_path]
    all_coor_gt = all_coor_list_gt[0][0]
    all_coor_pred = all_coor_list[0]
    for ii in range(all_coor_list_gt.shape[0]-1):
        all_coor_gt = np.concatenate((all_coor_gt, all_coor_list_gt[ii+1][0]), axis=0).astype(np.float32)
        all_coor_pred = np.concatenate((all_coor_pred, all_coor_list[ii+1]), axis=0).astype(np.float32)
    R0,T0 = find_rigid_alignment(torch.tensor(all_coor_gt), torch.tensor(all_coor_pred))
    rmsd_loss = torch.sqrt((((R0.mm(torch.tensor(all_coor_gt).T)).T + T0 - torch.tensor(all_coor_pred))**2).sum(axis=1).mean())


    N_res = all_coor_gt.shape[0]
    eta = 1e-4
    d0 = 1.24 * pow((N_res-15),1/3) - 1.8
    R0_refine = torch.tensor(R0, requires_grad=True)
    T0_refine = torch.tensor(T0, requires_grad=True)
    tmscore_loss = 0
    for iter in range(4000):
        tm_1 = R0_refine.mm(torch.tensor(all_coor_gt).T).T + T0_refine
        tm_2 = torch.tensor(all_coor_pred)
        pdist = nn.PairwiseDistance(p=2)
        tmscore_loss = -(1 / ((pdist(tm_1, tm_2) / d0)**2 + 1)).mean()
        tmscore_loss.backward()
        R0_refine = R0_refine - eta * R0_refine.grad.detach()
        R0_refine = torch.tensor(R0_refine, requires_grad=True)
        T0_refine = T0_refine - eta * T0_refine.grad.detach()
        T0_refine = torch.tensor(T0_refine, requires_grad=True)
    comb_coor = list(itertools.combinations(all_coor_list, 2))
    clash_sum = 0
    for comb in comb_coor:
        clash_sum += np.sum(spa.distance.cdist(comb[0], comb[1])<2)
    return rmsd_loss,-tmscore_loss,clash_sum
def only_clash_inf(i,adj, complex_coor_gt, chain_all_list, pdb_id):  
    all_coor_list, unique_docking_path = assemble(i,adj,complex_coor_gt,chain_all_list, pdb_id)
    comb_coor = list(itertools.combinations(all_coor_list, 2))
    clash_sum = 0
    for comb in comb_coor:
        clash_sum += np.sum(spa.distance.cdist(comb[0], comb[1])<2)
    return clash_sum

def only_clash_inf_esmfold(i,adj, complex_coor_gt, chain_all_list, pdb_id):  
    all_coor_list, unique_docking_path = assemble_esmfold(i,adj,complex_coor_gt,chain_all_list, pdb_id)
    comb_coor = list(itertools.combinations(all_coor_list, 2))
    clash_sum = 0
    for comb in comb_coor:
        clash_sum += np.sum(spa.distance.cdist(comb[0], comb[1])<2)
    return clash_sum

def build_loss_inf_esmfold(i,adj, complex_coor_gt, chain_all_list, pdb_id):  
    all_coor_list, unique_docking_path = assemble_esmfold(i,adj,complex_coor_gt,chain_all_list, pdb_id)
    all_coor_list_gt = np.array(complex_coor_gt[i],dtype = object)[unique_docking_path]
    all_coor_gt = all_coor_list_gt[0][0]
    all_coor_pred = all_coor_list[0]
    for ii in range(all_coor_list_gt.shape[0]-1):
        all_coor_gt = np.concatenate((all_coor_gt, all_coor_list_gt[ii+1][0]), axis=0).astype(np.float32)
        all_coor_pred = np.concatenate((all_coor_pred, all_coor_list[ii+1]), axis=0).astype(np.float32)
    R0,T0 = find_rigid_alignment(torch.tensor(all_coor_gt), torch.tensor(all_coor_pred))
    rmsd_loss = torch.sqrt((((R0.mm(torch.tensor(all_coor_gt).T)).T + T0 - torch.tensor(all_coor_pred))**2).sum(axis=1).mean())


    N_res = all_coor_gt.shape[0]
    eta = 1e-4
    d0 = 1.24 * pow((N_res-15),1/3) - 1.8
    R0_refine = torch.tensor(R0, requires_grad=True)
    T0_refine = torch.tensor(T0, requires_grad=True)
    tmscore_loss = 0
    for iter in range(4000):
        tm_1 = R0_refine.mm(torch.tensor(all_coor_gt).T).T + T0_refine
        tm_2 = torch.tensor(all_coor_pred)
        pdist = nn.PairwiseDistance(p=2)
        tmscore_loss = -(1 / ((pdist(tm_1, tm_2) / d0)**2 + 1)).mean()
        tmscore_loss.backward()
        R0_refine = R0_refine - eta * R0_refine.grad.detach()
        R0_refine = torch.tensor(R0_refine, requires_grad=True)
        T0_refine = T0_refine - eta * T0_refine.grad.detach()
        T0_refine = torch.tensor(T0_refine, requires_grad=True)
    comb_coor = list(itertools.combinations(all_coor_list, 2))
    clash_sum = 0
    for comb in comb_coor:
        clash_sum += np.sum(spa.distance.cdist(comb[0], comb[1])<2)
    return rmsd_loss,-tmscore_loss,clash_sum




def is_connected(edges,chain_num):
    nodes = {i for i in range(chain_num)}
    visited = set()

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for edge in edges:
            if node in edge:
                dfs(edge[0] if edge[1] == node else edge[1])

    dfs(0)
    return visited == nodes

def creating_dgls_all(chain_num):
    graphs = []
    for num_edges in range(chain_num):
        for edge_combination in itertools.combinations(itertools.combinations(range(chain_num), 2), num_edges):
            if is_connected(edge_combination,chain_num):
                graphs.append(torch.tensor(edge_combination).T)

    final_graphs = []
    for i in range(len(graphs)):
        a = graphs[i]
        for j in range(a.size(1)-1):
            while not ((a[0, j+1] == a[:, :j+1]) + (a[1, j+1] == a[:, :j+1])).any():
                print(i,'-th before:',a)
                n = j + 1
                a = torch.cat((a[:, :n], a[:, n+1:], a[:, n:n+1]), dim=1)
                print(i,'-th after:',a)
            if (a[:,j+1][1] == a[:,:j+1]).any():
                a[[0, 1], j+1] = a[[1, 0], j+1]
            else:
                assert (a[:,j+1][0] == a[:,:j+1]).any()
        print(i,'-th final:',a)
        final_graphs.append(a)
    return final_graphs

    # for graph in graphs:
        # print(graph)
# def obtaining_final_dgls(graphs):
#     final_t = []
#     for i in range(len(graphs)):
#         a = graphs[i]
#         for j in range(t.size(1)) - 1:
#             while not ((a[0, j] == a[0, :j]) | (a[1, j] == a[1, :j])).any():
#                 print('before:',a)
#                 a = torch.cat((a[:, :n-1], a[:, n:], a[:, n-1:n]), dim=1)
#                 print('after:',a)
