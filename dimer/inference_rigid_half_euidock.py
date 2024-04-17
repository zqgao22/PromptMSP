import os

import torch
import datetime
import itertools
os.environ['DGLBACKEND'] = 'pytorch'
import scipy.spatial as spa
from tqdm import tqdm
from datetime import datetime as dt
from utils.protein_utils import preprocess_unbound_bound, protein_to_graph_unbound_bound
from biopandas.pdb import PandasPdb
from utils.train_utils import *
from utils.args import *
from utils.ot_utils import *
from utils.zero_copy_from_numpy import *
from utils.io import create_dir
import warnings
warnings.filterwarnings("ignore")
def log(*pargs):
    with open('./dimer/dimer_log.txt', 'a+') as w:
        w.write(" ".join(["{}".format(t) for t in pargs]))
        w.write("\n")
dataset = 'dips'
method_name = 'equidock'
remove_clashes = False  # Set to true if you want to remove (most of the) steric clashes. Will increase run time.
if remove_clashes:
    method_name = method_name + '_no_clashes'
    print('Inference with postprocessing to remove clashes')
else:
    print('Inference without any postprocessing to remove clashes')


# Ligand residue locations: a_i in R^3. Receptor: b_j in R^3
# Ligand: G_l(x) = -sigma * ln( \sum_i  exp(- ||x - a_i||^2 / sigma)  ), same for G_r(x)
# Ligand surface: x such that G_l(x) = surface_ct
# Other properties: G_l(a_i) < 0, G_l(x) = infinity if x is far from all a_i
# Intersection of ligand and receptor: points x such that G_l(x) < surface_ct && G_r(x) < surface_ct
# Intersection loss: IL = \avg_i max(0, surface_ct - G_r(a_i)) + \avg_j max(0, surface_ct - G_l(b_j))
def G_fn(protein_coords, x, sigma):
    # protein_coords: (n,3) ,  x: (m,3), output: (m,)
    e = torch.exp(- torch.sum((protein_coords.view(1, -1, 3) - x.view(-1,1,3)) ** 2, dim=2) / float(sigma) )  # (m, n)
    return - sigma * torch.log(1e-3 +  e.sum(dim=1) )


def compute_body_intersection_loss(model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma = 25., surface_ct=10.):
    assert model_ligand_coors_deform.shape[1] == 3
    loss = torch.mean( torch.clamp(surface_ct - G_fn(bound_receptor_repres_nodes_loc_array, model_ligand_coors_deform, sigma), min=0) ) + \
           torch.mean( torch.clamp(surface_ct - G_fn(model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma), min=0) )
    return loss


def get_rot_mat(euler_angles):
    roll = euler_angles[0]
    yaw = euler_angles[1]
    pitch = euler_angles[2]

    tensor_0 = torch.zeros([])
    tensor_1 = torch.ones([])
    cos = torch.cos
    sin = torch.sin

    RX = torch.stack([
        torch.stack([tensor_1, tensor_0, tensor_0]),
        torch.stack([tensor_0, cos(roll), -sin(roll)]),
        torch.stack([tensor_0, sin(roll), cos(roll)])]).reshape(3, 3)

    RY = torch.stack([
        torch.stack([cos(pitch), tensor_0, sin(pitch)]),
        torch.stack([tensor_0, tensor_1, tensor_0]),
        torch.stack([-sin(pitch), tensor_0, cos(pitch)])]).reshape(3, 3)

    RZ = torch.stack([
        torch.stack([cos(yaw), -sin(yaw), tensor_0]),
        torch.stack([sin(yaw), cos(yaw), tensor_0]),
        torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

    R = torch.mm(RZ, RY)
    R = torch.mm(R, RX)
    return R



def get_residues(df):
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    residues = list(df.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
    return residues

def get_nodes_coors_numpy(filename, all_atoms=False):
            df = PandasPdb().read_pdb(filename).df['ATOM']
            if not all_atoms:
                return torch.from_numpy(df[df['atom_name'] == 'CA'][['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32))
            return torch.from_numpy(df[['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32))

def main(args):
    dataset = 'db5'
    checkpoint_filename = './dimer/db5_model_best.pth'

    print('checkpoint_filename = ', checkpoint_filename)

    checkpoint = torch.load(checkpoint_filename, map_location=args['device'])

    for k,v in checkpoint['args'].items():
        args[k] = v
    args['debug'] = False
    args['device'] = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    args['n_jobs'] = 1
    args['worker'] = 0


    model = create_model(args, log)
    model.load_state_dict(checkpoint['state_dict'])
    param_count(model, log)
    model = model.to(args['device'])
    model.eval()


    pdb_files = []
    root_txt_name = './PDB-M/PDB-M.txt'
    root_pdb_name = './pdb_all/'
    output_dir = './dimer_gt/'

    with open(root_txt_name, "r") as f:  # 打开文件
        for line in f.readlines():
            pdb_files.append(line[:-1])

    
    chain_name_list = torch.load('./source_data/true_chain_name_list.pt')
    # chain_name_list = chain_name_list[2606:]
    # pdb_files = pdb_files[2606:]
    for ind, file in enumerate(tqdm(pdb_files)):

        pdb_filename = root_pdb_name + file + '.pdb'

        out_filename = os.path.join(output_dir, file[:-2] + '.pdb')

        ppdb = PandasPdb().read_pdb(pdb_filename)

        chain_name = chain_name_list[ind]

        log('Processing protein:',file)

        df = ppdb.df['ATOM']
        # [['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)
        all_combine = list(itertools.combinations(chain_name,2))
        for sing_com in all_combine:
            chain_1_df_atom = df[df['chain_id'].isin([sing_com[0]])]
            chain_2_df_atom = df[df['chain_id'].isin([sing_com[1]])]
            chain_coor_1 = chain_1_df_atom[['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)
            chain_coor_2 = chain_2_df_atom[['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)
            

            unbound_predic_ligand, \
            unbound_predic_receptor, \
            bound_ligand_repres_nodes_loc_clean_array,\
            bound_receptor_repres_nodes_loc_clean_array = preprocess_unbound_bound(
                get_residues(chain_1_df_atom), get_residues(chain_2_df_atom),
                graph_nodes=args['graph_nodes'], pos_cutoff=args['pocket_cutoff'], inference=True)

            ligand_receptor_distance = spa.distance.cdist(bound_ligand_repres_nodes_loc_clean_array, bound_receptor_repres_nodes_loc_clean_array)
            # print(np.where(ligand_receptor_distance < 8)[0].shape[0])

            if np.where(ligand_receptor_distance < 8)[0].shape[0] < ligand_receptor_distance.shape[0] * ligand_receptor_distance.shape[1] * 0.01 * 0.01:
                # print('extract')
                protein_1_ca_coor = bound_ligand_repres_nodes_loc_clean_array
                protein_2_ca_coor = bound_receptor_repres_nodes_loc_clean_array

                ligand_graph, receptor_graph = protein_to_graph_unbound_bound(unbound_predic_ligand,
                                                                            unbound_predic_receptor,
                                                                            bound_ligand_repres_nodes_loc_clean_array,
                                                                            bound_receptor_repres_nodes_loc_clean_array,
                                                                            graph_nodes=args['graph_nodes'],
                                                                            cutoff=args['graph_cutoff'],
                                                                            max_neighbor=args['graph_max_neighbor'],
                                                                            one_hot=False,
                                                                            residue_loc_is_alphaC=args['graph_residue_loc_is_alphaC']
                                                                            )

                if args['input_edge_feats_dim'] < 0:
                    args['input_edge_feats_dim'] = ligand_graph.edata['he'].shape[1]


                ligand_graph.ndata['new_x'] = ligand_graph.ndata['x']

                assert np.linalg.norm(bound_ligand_repres_nodes_loc_clean_array - ligand_graph.ndata['x'].detach().cpu().numpy()) < 1e-1
                device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
                # Create a batch of a single DGL graph
                batch_hetero_graph = batchify_and_create_hetero_graphs_inference(ligand_graph, receptor_graph)
                
                batch_hetero_graph = batch_hetero_graph.to(args['device'])
                tmp = batch_hetero_graph.nodes['ligand'].data['new_x']
                batch_hetero_graph.nodes['ligand'].data['new_x'] = batch_hetero_graph.nodes['ligand'].data['new_x'] + torch.tensor(np.random.normal(2,2, tmp.size())).to(device).float()
                
                
                model_ligand_coors_deform_list, \
                model_keypts_ligand_list, model_keypts_receptor_list, \
                all_rotation_list, all_translation_list = model(batch_hetero_graph, epoch=0)


                rotation = all_rotation_list[0].detach().cpu().numpy()
                translation = all_translation_list[0].detach().cpu().numpy()
                
                protein_1_ca_coor = (rotation @ bound_ligand_repres_nodes_loc_clean_array.T).T+translation
                protein_2_ca_coor = bound_receptor_repres_nodes_loc_clean_array
            else:
                protein_1_ca_coor = bound_ligand_repres_nodes_loc_clean_array
                protein_2_ca_coor = bound_receptor_repres_nodes_loc_clean_array

                

            coor_pair_list = []
            coor_pair_list.append(protein_1_ca_coor)
            coor_pair_list.append(protein_2_ca_coor)
            final_pair_name = output_dir + file + '_' + sing_com[0] + '_' + sing_com[1] + '.npy'
            np.save(final_pair_name,np.array(coor_pair_list,dtype=object))
            




if __name__ == "__main__":
    main(args)