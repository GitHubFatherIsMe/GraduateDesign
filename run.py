from MADDPG import MADDPG
from CGAN import CGAN
import torch
import utils
import os
import numpy as np
from argparse import ArgumentParser


parser = ArgumentParser(description='args accepted by PTSP.')
parser.add_argument('--data', type=str, default=None,
                    help="your residue sequence")
data = parser.parse_args().data

print(data)

torch.manual_seed(1234)
world = utils.World()
n_agents = 45
n_states = 40*3
n_actions = 3
capacity = 100
batch_size = 10

episodes_before_train = 10

input_dim = 23
hidden_dim = 10
output_dim = 3
num_layers = 2

padding_length = 250
pssm_row_length = 20

lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)

cgan = CGAN(lr, beta1, input_dim, hidden_dim, output_dim, num_layers)

for i in range(n_agents):
    maddpg.actors[i].load_state_dict(
        torch.load('networks/actors/actor%d.pt' % i))

cgan.netG.load_state_dict(torch.load('networks/netG.pt'))
cgan.netD.load_state_dict(torch.load('networks/netD.pt'))

# %%
job_id = world.load_raw_data(data)

pssm = torch.tensor(np.genfromtxt('%s/%s.pssm' % (job_id, job_id))).float()
padding = torch.zeros(padding_length-pssm.shape[0], pssm.shape[1])
pssm = torch.cat([pssm, padding], dim=0).unsqueeze(0)


#%%
num_epoch = 0
for _ in range(1000):
    actions = maddpg.select_action(world.state)
    _, _, judgement = world.step(actions, cgan.netG, pssm, True, cgan.netD)
    
    print(judgement)
    num_epoch += 1
    if num_epoch % 100 == 0:
        # render
        print('render!')
    if judgement:
        break

