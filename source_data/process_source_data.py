import random
import logging
from dgl import save_graphs, load_graphs
import esm
import numpy as np
import os
import pickle
from biopandas.pdb import PandasPdb
import pandas as pd
from protein_util import *
from dgl import save_graphs,load_graphs
from tqdm import tqdm
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import argparse


parser = argparse.ArgumentParser(description='source data creation', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-n_min', type=int, default=3, help='the minimum chain number')
parser.add_argument('-n_max', type=int, default=5, help='the maximum chain number')
parser.add_argument('-source_or_target', type=str, default='source', help='For source, only process n=3,4,5; for target, process all multimers')
parser.add_argument('-homo_ratio', type=float, default=1.0, help='The coefficient controlling homogeneity, the larger the number, the greater the homogeneity of multimers.')
parser.add_argument('-data_fraction', type=float, default=1.0, help='One can choose to process only a part of the data.')
args = parser.parse_args().__dict__

def get_similarity_pct(query, target):
    alignments = pairwise2.align.globalxx(query, target) 
    seq_length = min(len(query), len(target))
    matches = alignments[0][2]
    percent_match = (matches / seq_length) * 100
    return percent_match

def get_residues_new_new(pdb_filename):
    residue_list_all = []
    drop_list = []
    res_name_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',
           'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
           'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
            'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU', 'PTR',
           'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ',
           'TYS', 'CYX', 'GLZ']
    df = PandasPdb().read_pdb(pdb_filename).df['ATOM']
    drop_list = list(set([item for item in df['residue_name'] if item not in set(res_name_list)]))
    for ele in drop_list:
        df = df.drop(index=df[df['residue_name']==ele].index)
    chain_num = df['chain_id'].unique().shape[0]
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    group = df.groupby('chain')
    for key, value in group:
        residue_list = list(value.groupby(['chain', 'residue', 'resname'])) 
        residue_list_all.append(residue_list)
    return residue_list_all, chain_num

def filter_residues(residues):
        residues_filtered = []
        for residue in residues:
            df = residue[1]
            Natom = df[df['atom_name'] == 'N']
            alphaCatom = df[df['atom_name'] == 'CA']
            Catom = df[df['atom_name'] == 'C']

            if Natom.shape[0] == 1 and alphaCatom.shape[0] == 1 and Catom.shape[0] == 1:
                residues_filtered.append(residue)
        return residues_filtered

def get_alphaC_loc_array(bound_predic_clean_list):
    bound_alphaC_loc_clean_list = []
    for residue in bound_predic_clean_list:
        df = residue[1]
        alphaCatom = df[df['atom_name'] == 'CA']
        alphaC_loc = alphaCatom[['x', 'y', 'z']].to_numpy().squeeze().astype(np.float32)
        assert alphaC_loc.shape == (3,), \
            f"alphac loc shape problem, shape: {alphaC_loc.shape} residue {df} resid {df['residue']}"
        bound_alphaC_loc_clean_list.append(alphaC_loc)
    if len(bound_alphaC_loc_clean_list) <= 1:
        bound_alphaC_loc_clean_list.append(np.zeros(3))
    # r,t = UniformRotation_Translation(translation_interval=5.0)
    # return (r @ np.stack(bound_alphaC_loc_clean_list, axis=0).T).T + \
    #     np.repeat(t,len(bound_alphaC_loc_clean_list),axis=0)
    return np.stack(bound_alphaC_loc_clean_list, axis=0)


def filtered_residues_2_list(f_r):
    r_list = torch.tensor([residue_type_pipr(f_r[j][0][2]) for j in range(len(f_r))]).mean(0)
    return r_list

def single_complex(pdb_file_name):
    residue_list, chain_num = get_residues_new_new(pdb_file_name)
    residue_list_all = []
    for residues in residue_list:
        if len(residues) >= 50:
            residue_list_all.append(residues)
    chain_num = len(residue_list_all)
# chain_num here considers all the chains including RNA, etc,.
    if chain_num > args['n_min']-1 and chain_num < args['n_max']+1:
        residues_filtered_list_all = []
        true_chain_name = []
        all_chains_rep = []
        for i in range(chain_num):
            filtered_residue = filter_residues(residue_list_all[i])
            if filtered_residue != [] and len(filtered_residue) > 1:
                residues_filtered_list_all.append(filtered_residue)
                true_chain_name.append(filtered_residue[0][0][0])
        chain_actual = len(residues_filtered_list_all)
# chain_actual is the number of actual protein chains whose length is > 50.
        if chain_actual > args['n_min']-1 and chain_actual < args['n_max']+1:
            residue_node_loc_list_all = []
            for residues_filtered in residues_filtered_list_all:
                residue_node_loc_list_all.append([get_alphaC_loc_array(residues_filtered)])            
            for i in range(chain_actual):
                chain_single = filtered_residues_2_list(residues_filtered_list_all[i])
                all_chains_rep.append(chain_single)
            x = torch.stack(all_chains_rep)
            pairwised=torch.cosine_similarity(x.unsqueeze(1),x.unsqueeze(0),dim=-1)
            
            if torch.where(pairwised > 0.999)[0].shape[0] >= chain_actual * chain_actual * args['homo_ratio']:
                all_chains_rep, residue_node_loc_list_all, true_chain_name = None,None,None

            return all_chains_rep, residue_node_loc_list_all, true_chain_name
        else:
            return None,None,None
    else:
        return None,None,None
        
pdb_files = []
# Kindly note that PDB-M-rough is the dataset after CD-HIT, so it is rough.
# We will get the final PDB-M dataset after running this script (if we set args['source_or_target'] = 'target').
with open("./PDB-M/PDB-M-rough.txt", "r") as f:  
    for line in f.readlines():
        pdb_files.append(line[:-1])
if args['source_or_target'] == 'source':
    log_name = './source_data/3_5_chain_pdb.txt'
else:
    log_name = './source_data/all_chain_pdb.txt'
all_chains_rep_list = []
coor_gt_list = []
true_chain_name_list = []
pdb_files = pdb_files[:round(len(pdb_files) * args['data_fraction'])]
for file in tqdm(pdb_files):
    pdb_file_name = './pdb_all/' + file + '.pdb'
    print('processing the protein:',file)
    all_chains_rep, residue_node_loc_list_all, true_chain_name = single_complex(pdb_file_name)
    if all_chains_rep != None:
        with open(log_name, "a") as file_txt:
            file_txt.write(file + "\n")
        all_chains_rep_list.append(all_chains_rep)
        coor_gt_list.append(residue_node_loc_list_all)
        true_chain_name_list.append(true_chain_name)        
if args['source_or_target'] == 'source':
    torch.save(all_chains_rep_list, './source_data/all_chains_rep_list_3_5.pt')
    torch.save(coor_gt_list, './source_data/coor_gt_list_3_5.pt')
    torch.save(true_chain_name_list, './source_data/chain_name_list_3_5.pt')
else:
    torch.save(all_chains_rep_list, './source_data/all_chains_rep_list_all.pt')
    torch.save(coor_gt_list, './source_data/coor_gt_list_all.pt')
    torch.save(true_chain_name_list, './source_data/chain_name_list_all.pt')
