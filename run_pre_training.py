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

parser = argparse.ArgumentParser(description='pre-training process', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-h_feats', type=int, default=512, help='GNN dim')
parser.add_argument('-cls_h', type=int, default=256, help='classifier dim')
parser.add_argument('-num_layers', type=int, default=1, help='GNN layer number')
parser.add_argument('-dropout_rate', type=float, default=0., help='dropout rate')
parser.add_argument('-gnn_type', type=str, default='gin', choices = ['gin','gcn'])
parser.add_argument('-pooling', type=str, default='max', choices = ['mean','max','sort','weight','attention'])
parser.add_argument('-lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('-bs', type=int, default=50, help='batch size')
parser.add_argument('-epochs', type=int, default=300, help='total training epochs')
args = parser.parse_args().__dict__


def log(*pargs):
    with open('./training_logs/1.txt', 'a+') as w:
        w.write(" ".join(["{}".format(t) for t in pargs]))
        w.write("\n")
# def normalization(data):
#     _range = np.max(data) - np.min(data)
#     return (data - np.min(data)) / _range
def normalization(data):
    _range = np.max(-data) + 50 
    return (-data + 50) / _range

#-------------------data processing------------------------
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
log('LOADING dgls')
graph_list = load_graphs('./source_data/train_oracle_dgl_train_3_5.bin')
log('LOADING labels')
label_list = torch.load('./source_data/rmsd_loss_train_3_5.pt')
label_list = normalization(np.array(label_list)).tolist()
sample=[(graph_list[0][i],label_list[i]) for i in range(len(graph_list[0]))]
random.shuffle(sample)
split_train_val = round(len(label_list) * 0.9)
sample_train = sample[:split_train_val]
sample_val = sample[-split_train_val:]

#-------------------configuration------------------------
ORACLE_MODEL = GNN_oracle(h_feats=args['h_feats'] ,cls_h=args['cls_h'], num_layers=args['num_layers'], \
    dropout_rate=args['dropout_rate'], GNN=args['gnn_type'],pooling=args['pooling']).to(device)
LR = args['lr']
optimizer = optim.Adam(ORACLE_MODEL.parameters(), lr=LR, betas=(0.9, 0.99))
scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma = 0.9)
epoch_losses = []
batch_size = args['bs']
best_val = 1e4
#-------------------training------------------------
print('Begin training...')
for epoch in range(args['epochs']):
    random.shuffle(sample_train)
    ORACLE_MODEL.train() 
    print("learning rate is ", optimizer.param_groups[0]["lr"])
    epoch_loss = 0
    scheduler.step()
    all_ind_count = 0
    batch_pro_sample = split_batch(sample_train,batch_size)
    for batch_ind in tqdm(range(len(batch_pro_sample))):
        graphs,labels = collate(batch_pro_sample[batch_ind])
        # pred_rmsd_batch = ORACLE_MODEL(graphs.to(device),edge_weight = torch.ones(graphs.num_edges()).to(device))
        pred_rmsd_batch,_ = ORACLE_MODEL(graphs.to(device))
        loss_batch = (pred_rmsd_batch.reshape((1,pred_rmsd_batch.shape[0])).squeeze(0) - \
            torch.tensor(labels).to(device)).abs().mean()

        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
        epoch_loss += loss_batch
        # if epoch > 5:
            # print(pred_rmsd_batch,torch.tensor(labels))
    batch_pro_sample_val = split_batch(sample_val,16)
    loss_val = 0
    for batch_ind_val in tqdm(range(len(batch_pro_sample_val))):
        graphs_val,labels_val = collate(batch_pro_sample_val[batch_ind_val])
        ORACLE_MODEL.eval()
        pred_rmsd_batch_val,_ = ORACLE_MODEL(graphs_val.to(device))
        loss_val_batch = (pred_rmsd_batch_val.reshape((1,pred_rmsd_batch_val.shape[0])).squeeze(0) - \
                torch.tensor(labels_val).to(device)).abs().mean()
        loss_val += loss_val_batch
    if best_val > loss_val/batch_ind_val:
        best_val = loss_val/batch_ind_val
        torch.save(ORACLE_MODEL, './checkpoints/source_model_best.pt')
    log('|training loss:',epoch_loss/batch_ind,'|val loss:',loss_val/batch_ind_val, '|best val for now:',best_val,'|')

        
    