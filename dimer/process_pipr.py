import random
import logging
import esm
import numpy as np
import os
import pickle
from biopandas.pdb import PandasPdb
import pandas as pd
from tqdm import tqdm

from Bio import pairwise2
from Bio.pairwise2 import format_alignment
def get_similarity_pct(query, target):
    alignments = pairwise2.align.globalxx(query, target) # 全局比对，相同的残基就给1分，不同和gap不扣分
    seq_length = min(len(query), len(target))
    matches = alignments[0][2]
    percent_match = (matches / seq_length) * 100
    return percent_match

def residue_seq(residue):

    dit = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
           'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
           'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',

           'HIP': 'H', 'HIE': 'H', 'TPO': 'T', 'HID': 'H', 'LEV': 'L', 'MEU': 'M', 'PTR': 'Y',
           'GLV': 'E', 'CYT': 'C', 'SEP': 'S', 'HIZ': 'H', 'CYM': 'C', 'GLM': 'E', 'ASQ': 'D',
           'TYS': 'Y', 'CYX': 'C', 'GLZ': 'G'}

    rare_residues = {'HIP': 'H', 'HIE': 'H', 'TPO': 'T', 'HID': 'H', 'LEV': 'L', 'MEU': 'M', 'PTR': 'Y',
           'GLV': 'E', 'CYT': 'C', 'SEP': 'S', 'HIZ': 'H', 'CYM': 'C', 'GLM': 'E', 'ASQ': 'D',
           'TYS': 'Y', 'CYX': 'C', 'GLZ': 'G'}

    if residue in rare_residues.keys():
        print('Some rare residue: ', residue)


    return dit[residue]
def get_residues(pdb_filename):
    residue_list_all = []
    df = PandasPdb().read_pdb(pdb_filename).df['ATOM']
    df = df.drop(index=df[df['residue_name']=='A'].index)
    df = df.drop(index=df[df['residue_name']=='U'].index)
    df = df.drop(index=df[df['residue_name']=='C'].index)
    df = df.drop(index=df[df['residue_name']=='T'].index)
    df = df.drop(index=df[df['residue_name']=='G'].index)
    chain_num = df['chain_id'].unique().shape[0]
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    group = df.groupby('chain')
    for key, value in group:
        residue_list = list(value.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
        residue_list_all.append(residue_list)
    return residue_list_all, chain_num

def get_residues_new(pdb_filename):
    residue_list_all = []
    drop_list = []
    res_name_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',
           'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
           'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
            'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU', 'PTR',
           'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ',
           'TYS', 'CYX', 'GLZ']
    df = PandasPdb().read_pdb(pdb_filename).df['ATOM']
    for i in range(df.shape[0]):
        if list(df[i:i+1]['residue_name'])[0] not in res_name_list:
            drop_list.append(i)
    df = df.drop(index=drop_list)
    chain_num = df['chain_id'].unique().shape[0]
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    group = df.groupby('chain')
    for key, value in group:
        residue_list = list(value.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
        residue_list_all.append(residue_list)
    return residue_list_all, chain_num

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
        residue_list = list(value.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
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
    # return (r @ np.stack(bound_alphaC_loc_clean_list, axis=0).T).T + \
    #     np.repeat(t,len(bound_alphaC_loc_clean_list),axis=0)
    return np.stack(bound_alphaC_loc_clean_list, axis=0)


def filtered_residues_2_list(f_r):
    seq_list = [residue_seq(f_r[j][0][2]) for j in range(len(f_r))]
    return seq_list

def single_complex(pdb_file_name):
    residue_list, chain_num = get_residues_new_new(pdb_file_name)
    residue_list_all = residue_list

    chain_num = len(residue_list_all)

    residues_filtered_list_all = []
    ture_chain_name = []
    all_chains_seq = []
    #------------------------main 1------------------------------
    for i in range(chain_num):
        filtered_residue = filter_residues(residue_list_all[i])
        if filtered_residue != [] and len(filtered_residue) > 1:
            residues_filtered_list_all.append(filtered_residue)
            ture_chain_name.append(filtered_residue[0][0][0])
    
    #------------------------main 1------------------------------
    chain_actual = len(residues_filtered_list_all)
    
    for i in range(chain_actual):
        chain_single = filtered_residues_2_list(residues_filtered_list_all[i])
        
        all_chains_seq.append("".join(chain_single))


    return all_chains_seq


        
def single_complex_to_array(pdb_file_name):
    residue_list, chain_num = get_residues_new_new(pdb_file_name)
    residue_list_all = residue_list

    chain_num = len(residue_list_all)

    residues_filtered_list_all = []
    ture_chain_name = []
    all_chains_seq = []
    #------------------------main 1------------------------------
    for i in range(chain_num):
        filtered_residue = filter_residues(residue_list_all[i])
        if filtered_residue != [] and len(filtered_residue) > 1:
            residues_filtered_list_all.append(filtered_residue)
            ture_chain_name.append(filtered_residue[0][0][0])
    
    #------------------------main 1------------------------------
    chain_actual = len(residues_filtered_list_all)
    residue_node_loc_list_all = []
    for residues_filtered in residues_filtered_list_all:
        residue_node_loc_list_all.append([get_alphaC_loc_array(residues_filtered)])
        


    return residue_node_loc_list_all
