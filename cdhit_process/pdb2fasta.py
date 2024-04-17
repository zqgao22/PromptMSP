import os
from tqdm import tqdm
name_list = []
for filename in os.listdir('../pdb_all/'):
    name_list.append(filename)
aa_codes = {
     'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E',
     'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
     'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N',
     'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S',
     'THR':'T', 'VAL':'V', 'TYR':'Y', 'TRP':'W', 'TPO':'O'}

f = open("./fasta_all.txt","w")
for i in tqdm(range(len(name_list))):
    pdb_name = name_list[i]
    file_name = '../pdb_all/' + pdb_name
    seq = '' 
    for line in open(file_name,encoding='ISO-8859-1'):
        if line[0:6] == "SEQRES":
            columns = line.split()
            for resname in columns[4:]:
                if resname in aa_codes:
                    seq = seq + aa_codes[resname]
    i = 0
    f.write ("\n>")
    pdb_name = pdb_name + "\n"
    f.write (pdb_name)

    while i < len(seq):
        f.write (seq[i:i + 64])
        i = i + 64
