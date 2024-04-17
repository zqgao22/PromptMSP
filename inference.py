import torch
from dgl import save_graphs, load_graphs
from gnn_models import *
import dgl
from utils import *
from datetime import datetime 
import torch.optim as optim
from torch.optim import lr_scheduler
import random
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='inference', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-dimer_type', type=str, default='esmfold', help='The dimer type used for inference.', choices=['gt','esmfold'])
args = parser.parse_args().__dict__

complex_coor_gt = torch.load('./PDB-M/coor_gt_list_test.pt')
chain_all_list = torch.load('./PDB-M//chain_name_list_test.pt')
ebd_chain_list = torch.load('./PDB-M//chains_rep_list_test.pt')
pdb_id = []
time_list = []
with open("./PDB-M/PDB-M-test.txt", "r") as f:  # 打开文件
    for line in f.readlines():
        pdb_id.append(line[:-1])
# pdb_id = pdb_id[:190]
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

ORACLE_MODEL = torch.load('./checkpoints/source_best_after_prompting.pt')
ORACLE_MODEL = ORACLE_MODEL.to(device)
PROMPT_MODEL = torch.load('./checkpoints/prompt_best_after_prompting.pt')
PROMPT_MODEL = PROMPT_MODEL.to(device)
PROMPT_MODEL.eval()
ORACLE_MODEL.eval()
rmsd_list, tm_list = [], []
for test_ind in tqdm(range(len(pdb_id))):
    time_1 = time.time()
    docked_chains = []
    remain_chains = list(range(len(chain_all_list[test_ind])))
    start_chain = random.choice(remain_chains)
    start_chain = 0
    remain_chains.pop(start_chain)
    docked_chains.append(start_chain)
    adj_0,adj_1 = torch.tensor([]), torch.tensor([])
    while remain_chains != []:
        inf_list = []
        inf_chain_list = []
        for d_chain in docked_chains:
            for u_chain in remain_chains:
                adj_0_inf = torch.cat((adj_0,torch.tensor([d_chain])),0).int()
                adj_1_inf = torch.cat((adj_1,torch.tensor([u_chain])),0).int()
                g_tar = build_dgl_inference(torch.cat((adj_0_inf.unsqueeze(0),adj_1_inf.unsqueeze(0)),0),ebd_chain_list[test_ind])
                g_tar.remove_edges(adj_0_inf.shape[0]-1)
                # g_tar = dgl.add_self_loop(g_tar)            
                _,node_emb_inf = ORACLE_MODEL(g_tar.to(device))
                prompt_embs_inf = PROMPT_MODEL(node_emb_inf.to(device))
                l3g_inf = dgl.DGLGraph(([0,2,3],[2,3,1])).to(device)
                l3g_inf.ndata['features'] = torch.cat((node_emb_inf[0:2,:],prompt_embs_inf[0:2,:]),0)
                pred_label,_ = ORACLE_MODEL(l3g_inf)
                inf_list.append(pred_label.detach().cpu()[0][0])
                inf_chain_list.append((d_chain,u_chain))

        order = 0
        clash = 10
        # while clash > 4:
        #     selected_pair = inf_chain_list[random.choice(np.where(np.array(inf_list) == np.sort(inf_list)[order])[0].tolist())]
            
        #     # rmsd,tmscore,clash = build_loss_inf(test_ind,torch.cat((torch.cat((adj_0,torch.tensor([selected_pair[0]])),0).int().unsqueeze(0),\
        #                     # torch.cat((adj_1,torch.tensor([selected_pair[1]])),0).int().unsqueeze(0)),0), complex_coor_gt, chain_all_list, pdb_id)
            
        #     clash = only_clash_inf(test_ind,torch.cat((torch.cat((adj_0,torch.tensor([selected_pair[0]])),0).int().unsqueeze(0),\
        #                     torch.cat((adj_1,torch.tensor([selected_pair[1]])),0).int().unsqueeze(0)),0), complex_coor_gt, chain_all_list, pdb_id)
        #     order += 1
        #     if order == len(inf_list) - 1:
        #         selected_pair = inf_chain_list[random.choice(np.where(np.array(inf_list) == np.sort(inf_list)[order])[0].tolist())]
        #         # rmsd,tmscore,clash = build_loss_inf(test_ind,torch.cat((torch.cat((adj_0,torch.tensor([selected_pair[0]])),0).int().unsqueeze(0),\
        #                     # torch.cat((adj_1,torch.tensor([selected_pair[1]])),0).int().unsqueeze(0)),0), complex_coor_gt, chain_all_list, pdb_id)
        #         clash = 0
        selected_pair = inf_chain_list[np.where(np.array(inf_list) == np.sort(inf_list)[order])[0].tolist()[0]]
        # rmsd,tmscore,clash = build_loss_inf(0,torch.cat((torch.cat((adj_0,torch.tensor([selected_pair[0]])),0).int().unsqueeze(0),\
                            # torch.cat((adj_1,torch.tensor([selected_pair[1]])),0).int().unsqueeze(0)),0), complex_coor_gt, chain_all_list, pdb_id)
        
        
            
        adj_0 = torch.cat((adj_0,torch.tensor([selected_pair[0]])),0).int()
        adj_1 = torch.cat((adj_1,torch.tensor([selected_pair[1]])),0).int()
        remain_chains.pop(remain_chains.index(selected_pair[1]))
        docked_chains.append(selected_pair[1])
        if remain_chains == []:
            if args['dimer_type'] == 'esmfold':
                rmsd,tmscore,clash = build_loss_inf_esmfold(test_ind,torch.cat((adj_0.unsqueeze(0),adj_1.unsqueeze(0)),0), complex_coor_gt, chain_all_list, pdb_id) 
            elif args['dimer_type'] == 'gt':
                rmsd,tmscore,clash = build_loss_inf(test_ind,torch.cat((adj_0.unsqueeze(0),adj_1.unsqueeze(0)),0), complex_coor_gt, chain_all_list, pdb_id) 
    rmsd_list.append(rmsd)
    tm_list.append(tmscore.detach().cpu())

print('The average RMSD:',np.mean(np.array(rmsd_list)))
print('The median RMSD:',np.median(np.array(rmsd_list)))
print('The average TM-Score:',np.mean(np.array(tm_list)))
print('The median TM-Score:',np.median(np.array(tm_list)))
print('All RMSD:',np.array(rmsd_list))
print('All TM-Scores:',np.array(tm_list))
