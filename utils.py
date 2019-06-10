# %%
import pickle
import numpy as np
import torch
import math
from collections import namedtuple
import random
import re
from os import popen, system, remove
from pyquery import PyQuery as pq


Experience = namedtuple('Experience',
                        ('states', 'actions', 'next_states', 'rewards'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def load_data():
    coords = open('../data/pickle/pickle_coord.pkl', 'rb')
    coords = torch.tensor(np.array(pickle.load(coords))).float()

    pssms = open('../data/pickle/pickle_pssm.pkl', 'rb')
    pssms = torch.tensor(np.array(pickle.load(pssms)).astype('float32'))

    structures = open('../data/pickle/pickle_ss.pkl', 'rb')
    structures = pickle.load(structures)
    records = []
    for i in range(len(structures)):
        record = []
        for structure in structures[i][1]:
            structure[-1] = round(structures[i][0]*structure[-1]*250)
            record.append(structure)
        records.append(record)
    return records, pssms, coords

# %%


def gen_pdb(tensor, index):
    pdb_file = open(index, 'w')
    for index, coord in enumerate(tensor[0]):
        row_text = 'ATOM  %5d  CB ATYR A %3d     %7.3f %7.3f %7.3f\n' % (
            index+1, index+1, coord[0], coord[1], coord[2])
        pdb_file.write(row_text)
    pdb_file.close()


# gen_pdb(tensor, '1b9o.pdb')
# %%

# 保证所有sub_chain，即二级结构的residue数都为n_length，因为需要保证 maddpg 的每个actor
# 和critic网络的输入参数维度恒定。此外，将sub_chain数量扩充至n_structures个保持恒定，因为模型
# 要求预先确定agents的数量。但是，我们并没有根据填充内容对应的actor做出的action调整
# state，它们将始终保持为0，所以这些actor无法对蛋白质折叠的任务造成影响。


class World():
    def __init__(self,):
        self.structures, self.pssms, self.coords = load_data()
        self.index = random.randint(0, 509)
        self.target = self.coords[self.index]
        # self.state: n_structures * n_length * n_samples * n_coordnates
        self.length_after_padding = 250
        self.n_structures = 45
        self.n_length = 40
        self.n_samples = 509
        self.n_coordnates = 3
        self.judge_threshold = 10
        self.last_distances = torch.ones(self.n_structures,)*100
        self.raw_sub_chain_length = []
        self.num_raw_sub_chain = 0
        self.state = self.init_state()
        self.raw_seq_coord = None

    # step表示每步更新，接受action，返回state，reward，done
    # actions: n_agents * n_actions
    def step(self, actions, generator, pssm, test, discriminator=None):
        self.adjust_state(actions, generator, pssm)
        reward, done = self.judge_state(test, discriminator, pssm)

        return self.state, reward, done

    def init_state(self):
        self.raw_sub_chain_length = []
        structure = self.structures[self.index]
        state = torch.zeros(self.length_after_padding, 3)
        iterator = 0
        for i in range(state.shape[0]):
            state[i] += iterator
            iterator += 0.05
        splitted_chain = []
        sub_start = 0
        for sub_structure in structure:
            sub_end = sub_structure[-1]+1
            sub_chain = state[sub_start:sub_end]
            self.raw_sub_chain_length.append(sub_chain.shape[0])
            padding = torch.zeros(self.n_length-sub_chain.shape[0], 3)
            splitted_chain.append(torch.cat((sub_chain, padding), dim=0))
            sub_start = sub_end
        self.num_raw_sub_chain = len(splitted_chain)
        for _ in range(self.n_structures-self.num_raw_sub_chain):
            splitted_chain.append(torch.zeros(self.n_length, 3))
        return torch.stack(splitted_chain)

    def change_target(self, index):
        self.target = self.coords[index]
        self.state = self.init_state()
        self.index = index

    def adjust_state(self, actions, generator, pssm):
        for index, action in enumerate(actions[:self.num_raw_sub_chain]):
            self.state[index][:self.raw_sub_chain_length[index]] += action
        raw_seq = []
        for index, chain_length in enumerate(self.raw_sub_chain_length):
            raw_seq.extend(self.state[index][:chain_length])
        raw_seq = torch.stack(raw_seq)
        padding = torch.zeros(
            self.length_after_padding-raw_seq.shape[0], raw_seq.shape[1])
        raw_seq = torch.cat([raw_seq, padding], dim=0).unsqueeze(0)
        raw_seq=generator(raw_seq, pssm)[0]
        self.raw_seq_coord = raw_seq
        count=0
        for index, chain_length in enumerate(self.raw_sub_chain_length):
            for i in range(chain_length):
                self.state[index][i]=self.raw_seq_coord[0][count]
                count+=1


    def judge_state(self, test, discriminator=None, pssm=None):
        state = [self.state[index][:length]
                 for index, length in enumerate(self.raw_sub_chain_length)]
        target = []
        sub_start = 0
        for length in self.raw_sub_chain_length:
            sub_end = sub_start+length
            target.append(self.target[sub_start:sub_end])
            sub_start = sub_end
        distances = torch.zeros(self.n_structures,)
        rewards = torch.zeros(self.n_structures,)
        if not test:
            # 计算当前三级结构与真实三级结构之间的差距
            for index, (sub_current, sub_target) in enumerate(zip(state, target)):
                distances[index] = torch.sqrt(
                    torch.sum(torch.pow(sub_current - sub_target, 2)))
            rewards = self.last_distances-distances
            self.last_distances = distances
            if distances.sum() < self.judge_threshold:
                return rewards, True
            else:
                return rewards, False
        if test:
            # 使用CGAN判断当前三级结构是否已具备满足给定条件下的蛋白质所应具备的特征
            # 奖励可由分辨器的输出确定
            judgement = discriminator(self.raw_seq_coord, pssm)
            if judgement > 0.45:
                return 0, True
            else:
                return 0, False

    def load_raw_data(self, file):

        job_id = popen(
            'perl jpredapi submit mode=single format=fasta file=%s silent' % file).read()[-11:-1]
        system(
            'perl jpredapi status jobid=%s getResults=yes checkEvery=60 silent' % job_id)
        system('tar -x  -f ./%s.tar.gz  -C ./%s' % (job_id, job_id))
        remove('./%s.tar.gz' % job_id)
        self.raw_sub_chain_length = []
        doc = pq(filename='./%s/%s.simple.html' % (job_id, job_id))
        structures = re.sub('<.*?>', '', doc('code').html().split('\n')[1])
        structures = re.finditer(r'(.)\1{0,}', structures)
        for structure in structures:
            self.raw_sub_chain_length.append(len(structure.group()))

        self.num_raw_sub_chain = len(self.raw_sub_chain_length)
        state = torch.zeros(self.length_after_padding, 3)
        iterator = 0
        for i in range(state.shape[0]):
            state[i] += iterator
            iterator += 0.05
        self.state = []
        location = 0
        for length in self.raw_sub_chain_length:
            sub_chain = state[location:location+length]
            padding = torch.zeros(self.n_length-sub_chain.shape[0], 3)
            self.state.append(torch.cat((sub_chain, padding), dim=0))
            location += length

        for _ in range(self.n_structures-self.num_raw_sub_chain):
            self.state.append(torch.zeros(self.n_length, 3))
        return job_id


# world = World()
# # file = '../data/fasta/1b9o.fasta.txt'
# actions = torch.randn(45, 3)
# world.adjust_state(actions)
# world.judge_state(False)
# # world.load_raw_data(file)
# %%
