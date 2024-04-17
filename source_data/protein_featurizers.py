# -*- coding: utf-8 -*-
# Node and edge featurization for molecular graphs.
# pylint: disable= no-member, arguments-differ, invalid-name

from dgllife.utils import one_hot_encoding


def residue_type_one_hot_dips(residue):

    dit = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
           'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
           'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
           'HIP': 'H', 'HIE': 'H', 'TPO': 'T', 'HID': 'H', 'LEV': 'L', 'MEU': 'M', 'PTR': 'Y',
           'GLV': 'E', 'CYT': 'C', 'SEP': 'S', 'HIZ': 'H', 'CYM': 'C', 'GLM': 'E', 'ASQ': 'D',
           'TYS': 'Y', 'CYX': 'C', 'GLZ': 'G'}
    allowable_set = ['Y', 'R', 'F', 'G', 'I', 'V', 'A', 'W', 'E', 'H', 'C', 'N', 'M', 'D', 'T', 'S', 'K', 'L', 'Q', 'P']
    res_name = residue
    if res_name not in dit.keys():
        res_name = None
    else:
        res_name = dit[res_name]
    return one_hot_encoding(res_name, allowable_set, encode_unknown=True)


def residue_type_one_hot_dips_not_one_hot(residue):

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

    indicator = {'Y': 0, 'R': 1, 'F': 2, 'G': 3, 'I': 4, 'V': 5,
                 'A': 6, 'W': 7, 'E': 8, 'H': 9, 'C': 10, 'N': 11,
                 'M': 12, 'D': 13, 'T': 14, 'S': 15, 'K': 16, 'L': 17, 'Q': 18, 'P': 19}
    res_name = residue
    if res_name not in dit.keys():
        return 20
    else:
        res_name = dit[res_name]
        return indicator[res_name]


def residue_type_pipr(residue):

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

    indicator = {'Y': [0.27962074,-0.051454283,0.114876375,0.3550331,1.0615551,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0], 
    'R': [-0.15621762,-0.19172126,-0.209409,0.026799612,1.0879921,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0], 
    'F': [0.2315121,-0.01626652,0.25592703,0.2703909,1.0793934,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0], 
    'G': [-0.07281224,0.01804472,0.22983849,-0.045492448,1.1139168,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 
    'I': [0.15077977,-0.1881559,0.33855876,0.39121667,1.0793937,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0], 
    'V': [-0.09511698,-0.11654304,0.1440215,-0.0022315443,1.1064949,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 
    'A': [-0.17691335,-0.19057421,0.045527875,-0.175985,1.1090639,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 
    'W': [0.25281385,0.12420933,0.0132171605,0.09199735,1.0842415,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0], 
    'E': [-0.06940994,-0.34011552,-0.17767446,0.251,1.0661993,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0], 
    'H': [0.019046513,-0.023256639,-0.06749539,0.16737276,1.0796973,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0], 
    'C': [-0.31572455,0.38517416,0.17325026,0.3164464,1.1512344,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0], 
    'N': [0.41597384,-0.22671205,0.31179032,0.45883527,1.0529875,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0], 
    'M': [0.06302169,-0.10206237,0.18976009,0.115588315,1.0927621,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0], 
    'D': [0.00600859,-0.1902303,-0.049640052,0.15067418,1.0812483,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0], 
    'T': [0.054446213,-0.16771607,0.22424258,-0.01337227,1.0967118,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0], 
    'S': [0.17177454,-0.16769698,0.27776834,0.10357749,1.0800852,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0], 
    'K': [0.22048187,-0.34703028,0.20346786,0.65077996,1.0620389,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0], 
    'L': [0.0075188675,-0.17002057,0.08902198,0.066686414,1.0804346,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0], 
    'Q': [0.25189143,-0.40238172,-0.046555642,0.22140719,1.0362468,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0], 
    'P': [0.017954966,-0.09864355,0.028460773,-0.12924117,1.0974121,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]}
    res_name = residue
    if res_name not in dit.keys():
        # print('UNK')
        return indicator['H']
    else:
        res_name = dit[res_name]
        return indicator[res_name]