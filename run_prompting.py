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


parser = argparse.ArgumentParser(description='prompting process', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-h_feats', type=int, default=512, help='MLP dim')
parser.add_argument('-dropout_rate', type=float, default=0., help='dropout rate')
parser.add_argument('-split_ratio', type=float, default=1., help='train-val split ratio')
parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('-bs', type=int, default=3000, help='batch size')
parser.add_argument('-epochs', type=int, default=300, help='total training epochs')
args = parser.parse_args().__dict__



def log(*pargs):
    with open('./prompting_logs/1.txt', 'a+') as w:
        w.write(" ".join(["{}".format(t) for t in pargs]))
        w.write("\n")
print('Loading processed data...')
complex_coor_gt = torch.load('./source_data/coor_gt_list_train.pt')
chain_all_list = torch.load('./source_data/chain_name_list_train.pt')
ebd_chain_list = torch.load('./source_data/chains_rep_list_train.pt')

pdb_id = []
with open("./PDB-M/PDB-M-train.txt", "r") as f:  
    for line in f.readlines():
        pdb_id.append(line[:-1])

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
log('LOADING dgls')
graph_list = load_graphs('./target_data/train_oracle_dgl_all.bin')
log('LOADING labels')
label_list = torch.load('./target_data/rmsd_loss_all.pt')
tmp = np.array(label_list)
tmp[tmp<=0.1]=0
tmp[tmp>0.1]=1
label_list = tmp.tolist()
ORACLE_MODEL = torch.load('./checkpoints/source_model_best.pt')
ORACLE_MODEL = ORACLE_MODEL.to(device)
# ORACLE_MODEL.eval()
for name, parameter in ORACLE_MODEL.named_parameters():
     if name != 'cls1.weight' and name != 'cls1.bias' and name != 'cls2.weight' and name != 'cls2.bias':
        parameter.requires_grad = False
#-----------------------------------------prepare embeddings ----------------------------------------
node_emb_list = []
if os.path.exists('./target_data/new_node_emb.pt'):
    node_emb_list = torch.load('./target_data/new_node_emb.pt')
else:
    for ii in tqdm(range(len(graph_list[0]))):
        add_index = graph_list[0][ii].edges()[0].size(0)-1
        if add_index > -1:
            v_d = graph_list[0][ii].edges()[0][add_index]
            v_u = graph_list[0][ii].edges()[1][add_index]
            graph_list[0][ii].remove_edges(add_index)
            _,node_emb = ORACLE_MODEL(graph_list[0][ii].to(device))
            node_emb_list.append(torch.cat((node_emb[v_d].unsqueeze(0),node_emb[v_u].unsqueeze(0)),0))
    torch.save(node_emb_list, './target_data/new_node_emb.pt')
#-----------------------------------------prepare embeddings ----------------------------------------
sample=[(node_emb_list[i],label_list[i]) for i in range(len(node_emb_list))]
random.shuffle(sample)
split_train_val = round(len(label_list) * args['split_ratio'])
sample_train = sample[:split_train_val]
sample_val = sample[-split_train_val:]

PROMPT_MODEL = target_model(in_feats = 26, h_feats = args['h_feats'], f_feats = 13, dropout_rate=0., activation='ReLU').to(device)
LR = args['lr']
optimizer = optim.Adam([
                {'params': PROMPT_MODEL.parameters()},
                {'params': ORACLE_MODEL.cls1.parameters()},
                {'params': ORACLE_MODEL.cls2.parameters()}
            ], lr=LR, betas=(0.9, 0.99))
scheduler = lr_scheduler.StepLR(optimizer,step_size=20,gamma = 0.8)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
epoch_losses = []
batch_size = args['bs']
lf = torch.nn.CrossEntropyLoss(weight = torch.tensor([1.,1.]).to(device))
for epoch in range(args['epochs']):
    random.shuffle(sample_train)
    PROMPT_MODEL.train() 
    print("learning rate is ", optimizer.param_groups[0]["lr"])
    epoch_loss = 0
    scheduler.step()
    all_ind_count = 0
    tmp_pred, tmp_lab = [],[]
    batch_pro_sample = split_batch(sample_train,batch_size)
    for batch_ind in tqdm(range(len(batch_pro_sample))):
        l3g_list = []
        nodes_embs, labels = collate_target(batch_pro_sample[batch_ind])
        prompt_embs = PROMPT_MODEL(nodes_embs.to(device))
        for i in range(int(nodes_embs.size(0)/2)):
            l3g = dgl.DGLGraph(([0,2,3],[2,3,1])).to(device)
            l3g.ndata['features'] = torch.cat((nodes_embs[2*i:2*i+2,:],prompt_embs[2*i:2*i+2,:]),0)
            l3g_list.append(l3g)
        pred_label,_ = ORACLE_MODEL(dgl.batch(l3g_list))
        # loss_batch = (pred_label.reshape((1,pred_label.shape[0])).squeeze(0) - \
            # torch.tensor(labels).to(device)).abs().mean()
        loss_batch = lf(torch.cat((1-pred_label,pred_label),1),torch.tensor(labels,dtype=torch.long).to(device))
        
        # tmp_pred.append(pred_label.detach().cpu()[0][0])
        # tmp_lab.append(labels)
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
        epoch_loss += loss_batch
        if batch_ind == 0:
            print(pred_label[:15,0])
            print(labels[:15])
        
    # batch_pro_sample_val = split_batch(sample_val,1000)
    # loss_val = 0

    # for batch_ind_val in tqdm(range(len(batch_pro_sample_val))):
    #     l3g_list_val = []
    #     nodes_embs, labels = collate_target(batch_pro_sample_val[batch_ind_val])
    #     ORACLE_MODEL.eval()
    #     PROMPT_MODEL.eval()
    #     prompt_embs = PROMPT_MODEL(nodes_embs.to(device))
    #     for i in range(int(nodes_embs.size(0)/2)):
    #         l3g = dgl.DGLGraph(([0,2,3],[2,3,1])).to(device)
    #         l3g.ndata['features'] = torch.cat((nodes_embs[2*i:2*i+2,:],prompt_embs[2*i:2*i+2,:]),0)
    #         l3g_list_val.append(l3g)
    #     pred_label,_ = ORACLE_MODEL(dgl.batch(l3g_list_val))
    #     loss_val_batch = lf(torch.cat((1-pred_label,pred_label),1),torch.tensor(labels,dtype=torch.long).to(device))
    #     loss_val += loss_val_batch


    # best_val = loss_val/batch_ind_val
    torch.save(ORACLE_MODEL, './checkpoints/source_best_after_prompting.pt')
    torch.save(PROMPT_MODEL, './checkpoints/prompt_best_after_prompting.pt')
    log('|training loss:',epoch_loss/batch_ind)
    # print('|training loss:',epoch_loss/batch_ind,'|val loss:',loss_val/batch_ind_val, '|best val for now:',best_val,'|')

    

    

