from pyquery import PyQuery as pq
import os
import Bio.PDB.PDBList as PDBList
from Bio.PDB.PDBParser import PDBParser
import re
import numpy as np
import requests
import pickle


SS_dir = '../data/SCP_10~90_SS/'
PSSM_dir = '../data/SCP_10~90_pssm/'
FASTA_dir = '../data/fasta/'
PDB_dir = '../data/pdb/'
MAX_length = 1000


# download the cull pdb dataset
with open("../data/PDB.LIS-20170808-1.6-0.19.txt", 'r') as file:
    category = file
    pdbl = PDBList.PDBList()
    for protein in category.readlines():
        pdbl.retrieve_pdb_file(
            protein[:4], file_format="pdb", pdir="../data/pdb")


# %%
# download_fasta
host = ['https://www.rcsb.org/pdb/download/viewFastaFiles.do?structureIdList = ',
        '&compressionType = uncompressed']
PDB_files = []

PDB_files = [files for _, _, files in os.walk('../data/pdb')][0]

for file in PDB_files:
    url = host[0]+file[3:7]+host[1]
    fasta_seq = pq(url).html()
    file = open(FASTA_dir+file[3:7]+'.fasta', 'w')
    file.write(fasta_seq)


# %%
# single_chain_protein
FASTA_files = [files for _, _, files in os.walk('../data/fasta')][0]

single_chain_proteins = []
for fasta in FASTA_files:
    file = open(FASTA_dir+fasta, 'r')
    context = file.read()
    chains = len(re.findall('>.*\n', context))
    if chains == 1:
        single_chain_proteins.append(fasta)

record = open('../data/single_chain_proteins.txt', 'w')
record.write(str(single_chain_proteins))


# %%
# filter_SCP
with open('../data/single_chain_proteins.txt', 'r') as file:
    SCP_fasta = eval(file.read())
sequences = []

for file in SCP_fasta:
    seq = open(FASTA_dir+file, 'r')
    seq = ''.join(seq.readlines()[1:])
    seq = re.sub('\n', '', seq)
    sequences.append(len(seq))

# filter the sequences with length range from 10 to 90 percent
lowerLimit = np.percentile(sequences, 10)
upperLimit = np.percentile(sequences, 90)

filteredSeq = []

for file in SCP_fasta:
    seq = open(FASTA_dir+file, 'r')
    seq = ''.join(seq.readlines()[1:])
    seq = re.sub('\n', '', seq)
    if lowerLimit <= len(seq) <= upperLimit:
        filteredSeq.append(file)
print(len(filteredSeq))
record = open('../data/SCP_10~90_fasta.txt', 'w')
record.write(str(filteredSeq))

# remaining 1665 fasta

# %%
# download_secondary_structure
host = ['https://www.rcsb.org/pdb/explore/sequenceText.do?structureId = ', '&chainId = A']

with open('../data/SCP_10~90_fasta.txt', 'r') as file:
    SCP_fasta = eval(file.read())

for file in SCP_fasta:
    tds = ''
    url = host[0]+file[:4]+host[1]
    ss_seq = pq(url, encoding='gbk')("tr")
    tds = ''.join([tr.getchildren()[2].text if (index+2) %
                   3 == 0 else '' for index, tr in enumerate(ss_seq)])
    tds = tds.replace(u'\xa0', 'C').replace('\n', '')
    tds = ''.join(['' if (index+1) % 11 == 0 else tds[index]
                   for index, _ in enumerate(tds)])
    with open(SS_dir+file[:4]+'.ss', 'w') as f:
        f.write(tds)

# remaining 1623 ss,  something lost while crawling

# %%
# download_nr_database
host = ['https://ftp.ncbi.nlm.nih.gov/blast/db/nr.', '.tar.gz']

index = ['%02d' % i for i in range(100)]

for i in index:
    doc = requests.get(host[0]+i+host[1])
    with open('../data/nr/nr.'+i+'.tar.gz', 'wb') as file:
        file.write(doc.content)


# %%
# generator_pssm
blast = '../data/blastdb'
with open('../data/SCP_10~90_fasta.txt', 'r') as file:
    SCP_fasta = eval(file.read())
os.chdir(blast)

# to save time,  change 100 to 10
dblist = ' '.join(["nr.%02d" % i for i in range(100)])

os.system(
    "blastdb_aliastool -dblist '%s' -dbtype prot  -out nr -title 'nr' " % dblist)

for file in SCP_fasta:
    os.system('psiblast.exe -query ../data/fasta/%s -db nr -num_iterations 1 \
            -out_ascii_pssm ../data/SCP_10~90_pssm/%s.pssm \
            -save_pssm_after_last_round' % (file, file.split('.')[0]))


# %%
# generate the pickkle of pdb's coord
p = PDBParser(PERMISSIVE=1)

with open('../data/SCP_10~90_fasta.txt', 'r') as file:
    SCP_fasta = eval(file.read())

coords = []
for PDB in list(SCP_fasta):
    Cα = []
    file = open(SS_dir+PDB+'.ss', 'r')
    content = file.read()
    # the length of seq in ss file may be longer than it in pdb file
    ss_length = len(content)
    structure = p.get_structure(PDB[3:7], PDB_dir+PDB+'.ent')
    for model in structure.get_list():
        for chain in model.get_list():
            for residue in chain.get_list()[:ss_length]:
                if residue.resname != ' CA' and residue.has_id('CA'):
                    Cα.append(residue['CA'].get_coord())
    file.close()
    Cα = np.vstack(Cα)
    padding = np.ones([MAX_length-Cα.shape[0], 3])
    Cα = np.vstack([Cα, padding])
    coords.append(Cα)

with open('../data/pickle/pickle_coord.pkl', 'wb') as file:
    pickle.dump(coords, file)
