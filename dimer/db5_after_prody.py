import os

import torch

os.environ['DGLBACKEND'] = 'pytorch'
from prody import *
from  matplotlib.pylab  import  * 
ion ()
from datetime import datetime as dt
from utils.protein_utils import preprocess_unbound_bound, protein_to_graph_unbound_bound
from biopandas.pdb import PandasPdb
from utils.train_utils import *
from utils.args import *
from utils.ot_utils import *
from utils.zero_copy_from_numpy import *
from utils.io import create_dir
import os

import torch
from prody import *
os.environ['DGLBACKEND'] = 'pytorch'
from utils.protein_utils import rigid_transform_Kabsch_3D
from datetime import datetime as dt
from utils.protein_utils import preprocess_unbound_bound, protein_to_graph_unbound_bound
from biopandas.pdb import PandasPdb
from utils.train_utils import *
from utils.args import *
from utils.ot_utils import *
from utils.zero_copy_from_numpy import *
from utils.io import create_dir

def rmsdgzq(complex_coors_pred,complex_coors_true):
    R,b = rigid_transform_Kabsch_3D(complex_coors_pred.T, complex_coors_true.T)
    complex_coors_pred_aligned = ( (R @ complex_coors_pred.T) + b ).T
    complex_rmsd = np.sqrt(np.mean(np.sum( (complex_coors_pred_aligned - complex_coors_true) ** 2, axis=1)))
    return complex_rmsd

input_dir = '/apdcephfs/share_1364275/kaithgao/equidock_toy/equidock_public/test_sets_pdb/db5_test_random_transformed/random_transformed/'
output_dir = '/apdcephfs/share_1364275/kaithgao/equidock_toy/equidock_public/test_sets_pdb/prody_ac_2/'
pdb_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.pdb')]
for file in pdb_files:

    p = parsePDB(input_dir + file)

    p_ca = p.select('calpha')

    anm = ANM(input_dir + file)

    anm.buildHessian(p)

    anm.calcModes()

    traj_ca = traverseMode(anm[0],p, n_steps=2, rmsd=24.0)

    latter = traj_ca[-1].getCoords()

    writePDB(output_dir + file, traj_ca)

    # k = 6
    # ppdb_ligand = PandasPdb().read_pdb(input_dir + file)
    # nmp = np.array(ppdb_ligand.df['ATOM'][['x_coord', 'y_coord', 'z_coord']])
    # d0 = ppdb_ligand.df['ATOM']
    # ppdb_ligand.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = latter
    # # # d1 = np.array(d0[d0['atom_name']=='CA'][['x_coord', 'y_coord', 'z_coord']])
    # # # d2 = d1 + np.random.normal(0,k, (d1.shape))
    # # print(rmsdgzq(d1,d2))
    # ppdb_ligand.to_pdb(path='/apdcephfs/share_1364275/kaithgao/equidock_toy/equidock_public/test_sets_pdb/prody_ac_2/' + file, records=['ATOM'], gz=False)
    p_ori = parsePDB(input_dir + file)
    p_ori_ca = p_ori.select('calpha')
    p_fle = parsePDB(output_dir + file)
    p_fle_ca = p_fle.select('calpha')
    print(calcRMSD(p_fle_ca,p_ca))
# print(np.abs(np.random.normal(0,k, (nmp.shape))).mean())