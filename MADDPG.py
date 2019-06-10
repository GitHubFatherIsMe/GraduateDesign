import torch
from model import Actor, Critic
from copy import deepcopy
from utils import ReplayMemory, Experience
from torch.optim import Adam
import torch.nn as nn
import numpy as np


scale_reward = 0.01


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)

# %%


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size,
                 capacity, episodes_before_train):
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs,
                               dim_act) for i in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.actors]

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            non_final_mask = torch.ByteTensor(list(map(lambda s: s is not None,
                                                       batch.next_states)))

            # state_batch: batch_size x n_agents x dim_obs
            state_batch = torch.stack(batch.states).detach()
            # action_batch: batch_size x n_agents x dim_act
            action_batch = torch.stack(batch.actions).detach()
            reward_batch = torch.stack(batch.rewards).detach()
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = torch.stack(
                [s for s in batch.next_states if s is not None]).detach()

            # 此处whole_state和whole_action不再仅是对应actor所观察的
            # state和action，而是所有agent的
            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](whole_state, whole_action)

            non_final_next_actions = [
                self.actors_target[i](
                    non_final_next_states[:, i, :].view(-1, 40*3))
                for i in range(self.n_agents)]
            non_final_next_actions = torch.stack(non_final_next_actions)

            target_Q = torch.zeros(self.batch_size)

            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states),
                non_final_next_actions.view(-1, self.n_agents * self.n_actions)
            ).squeeze()
            # scale_reward: to scale reward in Q functions

            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
                reward_batch[:, agent].unsqueeze(1) * scale_reward)

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :].view(self.batch_size, -1)
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return sum(c_loss), sum(a_loss)

    def select_action(self, state_batch):
        # state_batch: n_agents x state_dim
        actions = torch.zeros(self.n_agents, self.n_actions)
        for i in range(self.n_agents):
            sb = state_batch[i].detach().view(1, -1)
            act = self.actors[i](sb)

            act += torch.from_numpy(np.random.randn(self.n_actions)
                                    * self.var[i]).float()

            if self.episode_done > self.episodes_before_train and self.var[i] > 0.05:
                self.var[i] *= 0.999998
            act = torch.clamp(act, -1.0, 1.0)

            actions[i] = act
        self.steps_done += 1

        return actions

# %%
