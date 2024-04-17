import os
import torch
import esm
import time
from esmfold_test import *
device = torch.device("cuda") 
model_esm = esm.pretrained.esmfold_v1()
model_esm = model_esm.eval().to(device)
import torch
import datetime
import itertools
os.environ['DGLBACKEND'] = 'pytorch'
import scipy.spatial as spa
from tqdm import tqdm
from process_pipr import *
from datetime import datetime as dt
from biopandas.pdb import PandasPdb

import warnings
warnings.filterwarnings("ignore")


def main():
    pdb_files = []
    root_txt_name = './PDB-M/PDB-M-test.txt'
    root_pdb_name = './pdb_all/'
    output_dir = './dimer_esmfold/'

    with open(root_txt_name, "r") as f:  # 打开文件
        for line in f.readlines():
            pdb_files.append(line[:-1])


    coor_gt_list = torch.load('./PDB-M/coor_gt_list_test.pt') 
    chain_name_list = torch.load('./PDB-M/chain_name_list_test.pt')

    # chain_name_list = chain_name_list[29:]
    # pdb_files = pdb_files[29:]
    # coor_gt_list = coor_gt_list[29:]
    time_list = []
    for ind, file in enumerate(tqdm(pdb_files)):

        pdb_filename = root_pdb_name + file + '.pdb'

        seq_list = single_complex(pdb_filename)

        out_filename = os.path.join(output_dir, file[:-2] + '.pdb')

        ppdb = PandasPdb().read_pdb(pdb_filename)

        chain_name = chain_name_list[ind]


        df = ppdb.df['ATOM']
        # [['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)
        all_combine = list(itertools.combinations(chain_name,2))
        for sing_com in all_combine:
            final_pair_name = output_dir + file + '_' + sing_com[0] + '_' + sing_com[1] + '.npy'
            if not os.path.exists(final_pair_name): 
                chain_1_seq = seq_list[chain_name.index(sing_com[0])]
                chain_2_seq = seq_list[chain_name.index(sing_com[1])]
                time1 = time.time()
                esm_ours(model_esm, chain_1_seq + ':' + chain_2_seq)
                
                x_1, x_2 = single_complex_to_array('result.pdb')

                chain_1_df_atom = df[df['chain_id'].isin([sing_com[0]])]
                chain_2_df_atom = df[df['chain_id'].isin([sing_com[1]])]

                coor_pair_list = []
                coor_pair_list.append(x_1[0])
                coor_pair_list.append(x_2[0])
                time2 = time.time()
                time_list.append(time2 - time1)
                np.save(final_pair_name,np.array(coor_pair_list,dtype=object))
            
            


if __name__ == "__main__":
    main()