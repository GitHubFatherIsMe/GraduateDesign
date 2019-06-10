# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:11:32 2019

@author: Tony
"""
#%%
from Bio.PDB.PDBParser import PDBParser
import numpy as np
import os
import pickle
import pandas as pd


SS_dir='../data/SCP_10~90_SS/'
NEW_dir='NEW_SS/'
PSSM_dir='../data/SCP_10~90_pssm/'
FASTA_dir='../data/SCP_10~90_fasta/'

#%%
ss_files=[]
for root,dirs,files in os.walk(SS_dir):
    ss_files=files

egiht2three={
    'H':'α',
    'B':'β',
    'E':'β',
    'G':'α',
    'I':'α',
    'T':'l',
    'S':'l',
    'C':'l',
    }

ss2Vec={
    'α':[1,0,0],
    'β':[0,1,0],
    'l':[0,0,1]
}

#%%
#这一段代码原本事项构造一个矩阵,每一行代表一个二级结构在整个序列中的起始点,
# 前三列是二级结构的独热表示,第四列指示起始点在序列中的位置,不过现在看来可能不需要了
num_ss=[]
lengths=[]

for ss_file in ss_files[:]:
    with open(SS_dir+ss_file) as file:
        ss=file.read()
        beginners=[ss2Vec[egiht2three[ss[0]]][:]]
        beginners[0].append(round(0,4))
        for index,acid in enumerate(ss[:-1]):
            if egiht2three[acid]!=egiht2three[ss[index+1]]:
                aa=ss2Vec[egiht2three[ss[index+1]]][:]
                aa.append(round((index+1)/len(ss),4))
                beginners.append(aa)
        num_ss.append((len(beginners),ss_file,len(ss),beginners))
        lengths.append(len(beginners))

lowwerLimit=np.percentile(lengths,10)
higherLimit=np.percentile(lengths,40)

filtered_items={}
for length,file_name,init_len,metric in num_ss:
    if lowwerLimit<=length and higherLimit>=length and init_len<=250:
        filtered_items[file_name[:4]]=metric

with open('last_ss.txt','w') as file:
    file.write(str(list(filtered_items.keys())))

#%%
p=PDBParser(PERMISSIVE=1)
PDB_files=[]

coords=[]
for PDB in list(filtered_items.keys()):  
    Cα=[]
    file=open(SS_dir+PDB+'.ss.txt')
    content=file.read()
    ss_length=len(content)
    structure=p.get_structure(PDB[3:7],'SCP_PDB/pdb'+PDB+'.ent')
    for model in structure.get_list():
        for chain in model.get_list():
             for residue in chain.get_list()[:ss_length]:
                 if residue.resname!=' CA' and residue.has_id('CA'):
                     ca=residue['CA']
                     Cα.append(ca.get_coord())    
    new_file=open(NEW_dir+PDB+'.ss.txt','w')
    new_file.write(content[:len(Cα)])
    file.close()
    new_file.close()
    Cα=np.vstack(Cα)
    padding=np.ones([250-Cα.shape[0],3])
    Cα=np.vstack([Cα,padding])
    coords.append(Cα)
 
with open('pickle_coord.pkl','wb') as file:
    pickle.dump(coords,file)


#%%
# 根据NEW_dir下的ss文件生成pickle_PSSM和pickle_SS
ss_files=[]
for root,dirs,files in os.walk(NEW_dir):
    ss_files=files

num_ss=[]

for ss_file in ss_files:
    with open(NEW_dir+ss_file) as file:
        ss=file.read()
        beginners=[ss2Vec[egiht2three[ss[0]]][:]]
        beginners[0].append(round(0,4))
        for index,acid in enumerate(ss[:-1]):
            if egiht2three[acid]!=egiht2three[ss[index+1]]:
                aa=ss2Vec[egiht2three[ss[index+1]]][:]
                aa.append(round((index+1)/len(ss),4))
                beginners.append(aa)
        num_ss.append((len(ss)/250,beginners))

with open('pickle_ss.pkl','wb') as file:
    pickle.dump(num_ss,file)  

#%%
all_pssm=[]

for file in ss_files:
    pssm=pd.read_csv(PSSM_dir+file[:4]+'.pssm',skipfooter=5,skiprows=2,delim_whitespace=True,
    engine='python',index_col=False)
    pssm_metric=[]
    for row in range(pssm.shape[0]):
        pssm_metric.append(np.array(pssm.iloc[row,2:22]))
    padding=np.ones([250-pssm.shape[0],20],dtype=int)*-10
    pssm_metric=np.vstack(pssm_metric)
    pssm_metric=np.vstack([pssm_metric,padding])
    all_pssm.append(pssm_metric/10)

with open('pickle_pssm.pkl','wb') as file:
    pickle.dump(all_pssm,file)  

# generate 250*4 secondary structure matrix
#%%
ss_list=[]
for root,dir,ss in os.walk(NEW_dir):
    ss_list=ss
    print(ss)

ss_set=[]
for name in ss_list[:1]:
    ss=open(NEW_dir+name,'r')
    print(ss.read())
    ss_matrix=[]
    for index,acid in enumerate(ss.read()[:-1]):
        if acid==ss.read()[index+1]:
            # padding=np.ze
            ss_matrix.append(ss.read()[ss2Vec[egiht2three[acid]]])
        else:
            ss_set.append(np.vstack(ss_matrix).float())
            ss_matrix=[]


# with open('pickle_ssmat.pkl','wb') as file:
#     pickle.dump(ss_metric,file)  