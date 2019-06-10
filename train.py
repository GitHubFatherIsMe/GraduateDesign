from MADDPG import MADDPG
from CGAN import CGAN
import test
import numpy as np
import torch
import utils

# %%
###########################################
# train the MADDPG
###########################################
torch.autograd.set_detect_anomaly(True)

food_reward = 10.
poison_reward = -1.
encounter_reward = 0.01
n_coop = 2

reward_record = []

np.random.seed(1234)
torch.manual_seed(1234)
world = utils.World()
n_agents = 45
# n_states: 蛋白质序列长度 * 坐标维度
n_states = 40*3
# 根据空间之间坐标系，可把一个位移动作视为三维向量，故 n_action = 3
n_actions = 3
# capacity = 10000
capacity = 1000
# batch_size = 100
batch_size = 20

# n_episode = 20000
n_episode = 100
# max_steps = 1000
max_steps = 5
# valid_steps = 100
valid_steps = 1
episodes_before_train = 20
scale_reward = 0.999

n_sample = 2
reward_threshold = 10000

# %%
###########################################
# train the CGAN
###########################################
# Number of training epochs
num_epochs = 5
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

input_dim = 23
hidden_dim = 10
output_dim = 3
num_epoch = 100
num_layers = 2

torch.autograd.set_detect_anomaly(True)

structures, pssms, coords = utils.load_data()
cgan = CGAN(lr, beta1, input_dim, hidden_dim, output_dim, num_layers)

for _ in range(num_epoch):
    for pssm, coord in zip(pssms, coords):
        cgan.update(coord.unsqueeze(0), pssm.unsqueeze(0))

torch.save(cgan.netG.state_dict(), 'networks/netG.pt')
torch.save(cgan.netD.state_dict(), 'networks/netD.pt')

# %%
maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)
# %%
# 先选择一个sample做强化学习训练，然后拿一个batch的samples来测试，
# 挑准确率最低的那个sample来训练model，
# 更新参数直到遍历完整个dataset或者reward对于所有sample都让人满意为止
total_reward = 0.0
for i_episode in range(n_episode):
    # 这里是大多数RL算法相同的地方，每完成一个episode后，将环境重置为初始状态，
    # 这样就可以反复对一个任务进行优化.
    # 现进行修改，依据上述策略
    if total_reward < reward_threshold:
        world.init_state()
        obs = world.state
    else:
        mini_reward = 100
        mini_index = 0
        samples_index = torch.randint(0, 509, (n_sample,))
        for index in samples_index:
            accumulative_reward = 0
            world.change_target(index)
            obs = world.state
            for t in range(valid_steps):
                # render every 100 episodes to speed up training
                action = maddpg.select_action(obs)
                obs, reward, done = world.step(action, cgan.netG, pssms[index].unsqueeze(0), False)
                # 因为MADDPG以batch形式返回reward，所以须使用.sum()求和，此处batch_size=1
                accumulative_reward += reward.sum()
            if accumulative_reward < mini_reward:
                mini_reward = accumulative_reward
                mini_index = index
        world.change_target(mini_index)
        obs = world.state

    for t in range(max_steps):

        # render every 100 episodes to speed up training
        action = maddpg.select_action(obs)
        
        obs_, reward, done = world.step(action, cgan.netG, pssms[world.index].unsqueeze(0), False)
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None
        total_reward += reward.sum()
        maddpg.memory.push(obs, action, next_obs, reward)
        obs = next_obs

        c_loss, a_loss = maddpg.update_policy()
        if done == True:
            break
    maddpg.episode_done += 1
    print('Episode: %d, reward = %f, ' %(i_episode, total_reward))
    print(c_loss,a_loss)
    reward_record.append(total_reward)


#%%
for i, actor in enumerate(maddpg.actors):
    torch.save(actor.state_dict(), 'networks/actors/actor%d.pt' % i)

for i, critic in enumerate(maddpg.critics):
    torch.save(critic.state_dict(), 'networks/critics/critic%d.pt' % i)


