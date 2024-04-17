import os

import torch
import datetime
import itertools
os.environ['DGLBACKEND'] = 'pytorch'
import scipy.spatial as spa
from tqdm import tqdm
from datetime import datetime as dt
from utils.protein_utils import preprocess_unbound_bound
from biopandas.pdb import PandasPdb
from utils.train_utils import *
from utils.args import *
from utils.ot_utils import *
from utils.zero_copy_from_numpy import *
from utils.io import create_dir
import warnings
warnings.filterwarnings("ignore")
def log(*pargs):
    with open('/home/taofeng/ziqigao/equidock/log.txt', 'a+') as w:
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

def find_rigid_alignment(A, B):
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
    drop_list = []
    res_name_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',
           'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
           'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
            'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU', 'PTR',
           'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ',
           'TYS', 'CYX', 'GLZ']
    drop_list = list(set([item for item in df['residue_name'] if item not in set(res_name_list)]))
    for ele in drop_list:
        df = df.drop(index=df[df['residue_name']==ele].index)
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
    

    pdb_files = []
    root_txt_name = './source_data/all_chain_pdb.txt'
    root_pdb_name = './pdb_all/'
    output_dir = './dimer_gt/'
    coor_gt_list = torch.load('./source_data/coor_gt_list_all.pt') 
    with open(root_txt_name, "r") as f:  # 打开文件
        for line in f.readlines():
            pdb_files.append(line[:-1])

    chain_name_list = torch.load('./source_data/chain_name_list_all.pt')

    for ind, file in enumerate(tqdm(pdb_files)):


        pdb_filename = root_pdb_name + file + '.pdb'

        ppdb = PandasPdb().read_pdb(pdb_filename)

        chain_name = chain_name_list[ind]

        log('Processing protein:',file)

        df = ppdb.df['ATOM']
        # [['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)
        all_combine = list(itertools.combinations(chain_name,2))
        extract = False
        for sing_com in tqdm(all_combine):
            bound_ligand_repres_nodes_loc_clean_array = coor_gt_list[ind][chain_name.index(sing_com[0])][0]
            bound_receptor_repres_nodes_loc_clean_array = coor_gt_list[ind][chain_name.index(sing_com[1])][0]
            ligand_receptor_distance = spa.distance.cdist(bound_ligand_repres_nodes_loc_clean_array, bound_receptor_repres_nodes_loc_clean_array)
            if np.where(ligand_receptor_distance < 7)[0].shape[0] < ligand_receptor_distance.shape[0] * ligand_receptor_distance.shape[1] * 0.01 * 0.01:
                # print('extract')
                protein_1_ca_coor = bound_ligand_repres_nodes_loc_clean_array
                protein_2_ca_coor = bound_receptor_repres_nodes_loc_clean_array
                a1 = protein_1_ca_coor[np.where(protein_1_ca_coor[:,0] == protein_1_ca_coor[:,0].max())[0],:]
                a2 = protein_2_ca_coor[np.where(protein_2_ca_coor[:,0] == protein_2_ca_coor[:,0].max())[0],:]
                if a1.shape[0]>1:
                    a1 = a1[0,:]
                if a2.shape[0]>1:
                    a2 = a2[0,:]
                t_fast = a1 - a2
                protein_2_ca_coor = protein_2_ca_coor + t_fast
            else:
                # print('extract')
                extract = True
                protein_1_ca_coor = bound_ligand_repres_nodes_loc_clean_array
                protein_2_ca_coor = bound_receptor_repres_nodes_loc_clean_array


        

            coor_pair_list = []
            coor_pair_list.append(protein_1_ca_coor)
            coor_pair_list.append(protein_2_ca_coor)
            final_pair_name = output_dir + file + '_' + sing_com[0] + '_' + sing_com[1] + '.npy'
            np.save(final_pair_name,np.array(coor_pair_list,dtype=object))
        print(extract)




if __name__ == "__main__":
    main(args)