import os
import shutil

result=[]
index_list = []
final = []
with open('./cluster_result.clstr','r') as f:
	for line in f:
		result.append(list(line.strip('\n').split(',')))

for i in range(len(result)):
	if result[i][0][:8] == '>Cluster':
		index_list.append(i)

for j in range(len(index_list)):
	
	final.append(result[index_list[j]+1][1][2:6])
	

with open("./dataset.txt", 'w') as f:
    for i in final:
        f.write(i+'\n')

print('After performing cd-hit, we have',len(final), 'PDB files in total')
