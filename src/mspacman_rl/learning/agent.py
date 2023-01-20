import numpy as np
import os
import random
import torch

from src.mspacman_rl.learning.dnn import DQN, DoubleDQN
from src.mspacman_rl.learning.memory import Memory
from src.mspacman_rl.utils.logging import Logger


def load_agent(file_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # read serialization from torch file
    serialization = torch.load(file_path, map_location=device)

    # create instance of agent
    try:
        agent_type = serialization['type']
        state_shape = serialization['_state_shape']
        actions = serialization['_actions']
        agent = globals()[agent_type](state_shape, actions)
    except KeyError:
        raise ValueError('Incompatible agent type.')

    # deserialize and return agent
    agent.deserialize(serialization)

    # load existing agent memory
    memory_path = '{}.memory.pickle'.format(os.path.splitext(file_path)[0])
    if os.path.exists(memory_path):
        agent.load_memory(memory_path)

    return agent


class Agent(object):
    def __init__(
            self,
            state_shape,
            actions,
            batch_size=32,
            burn_in_steps=100000,
            discount_factor=0.9,
            exploration_rate=1,
            exploration_rate_decay=0.99999975,
            exploration_rate_min=0.1,
            learning_interval=3,
            learning_rate=0.00025,
            memory_limit=100000,
            memory_prioritized=False,
    ):
        # init settings
        self._state_shape = state_shape
        self._actions = actions
        self._batch_size = batch_size
        self._burn_in_steps = burn_in_steps
        self._discount_factor = discount_factor
        self._exploration_rate = exploration_rate
        self._exploration_rate_decay = exploration_rate_decay
        self._exploration_rate_min = exploration_rate_min
        self._learning_interval = learning_interval
        self._learning_rate = learning_rate

        # init internal components
        self._curr_step = 0
        self._logger = Logger()
        self._memory = Memory(memory_limit, prioritized=memory_prioritized)

        # init dnn
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._dnn = None
        self._optimizer = None
        self._loss_fn = None

    @property
    def exploration_rate(self):
        return self._exploration_rate

    @property
    def logger(self):
        return self._logger

    @property
    def memory(self):
        return self._memory

    @property
    def steps(self):
        return self._curr_step

    def description(self, width=80):
        text = '{}'.format(self.__class__.__name__)
        text = '\n'.join([
            '{}[{}]{}'.format('-' * 2, text, '-' * max(0, width - len(text) - 4)),
            'State shape:               ({})'.format(', '.join(['{}'.format(x) for x in self._state_shape])),
            'Number of actions:         {}'.format(len(self._actions)),
            'Memory limit:              {}'.format(self._memory.limit),
            'Memory prioritized:        {}'.format(self._memory.prioritized),
            'Burn-in:                   {}'.format(self._burn_in_steps),
            'Discount factor:           {:.4f}'.format(self._discount_factor),
            'Exploration rate:          {:.4f}'.format(self._exploration_rate),
            'Exploration rate (decay):  {:.4f}'.format(self._exploration_rate_decay),
            'Exploration rate (min):    {:.4f}'.format(self._exploration_rate_min),
            'Learning rate:             {:.4f}'.format(self._learning_rate),
            'Batch size:                {}'.format(self._batch_size),
            'Device:                    {}'.format(self._device),
        ])
        return text

    def deserialize(self, serialization):
        # check if serialized data is compatible with this class
        if serialization.get('type') != self.__class__.__name__:
            raise ValueError('Incompatible serialization.')
        if '_actions' in serialization:
            self._actions = serialization['_actions']
        if '_batch_size' in serialization:
            self._batch_size = serialization['_batch_size']
        if '_burn_in_steps' in serialization:
            self._burn_in_steps = serialization['_burn_in_steps']
        if '_curr_step' in serialization:
            self._curr_step = serialization['_curr_step']
        if '_dnn' in serialization and serialization['_dnn'] is not None:
            self._dnn.deserialize(serialization['_dnn'])
        if '_exploration_rate' in serialization:
            self._exploration_rate = serialization['_exploration_rate']
        if '_exploration_rate_decay' in serialization:
            self._exploration_rate_decay = serialization['_exploration_rate_decay']
        if '_exploration_rate_min' in serialization:
            self._exploration_rate_min = serialization['_exploration_rate_min']
        if '_discount_factor' in serialization:
            self._discount_factor = serialization['_discount_factor']
        if '_learning_interval' in serialization:
            self._learning_interval = serialization['_learning_interval']
        if '_learning_rate' in serialization:
            self._learning_rate = serialization['_learning_rate']
        if '_logger' in serialization:
            self._logger.deserialize(serialization['_logger'])
        if '_optimizer_state_dict' in serialization and serialization['_optimizer_state_dict'] is not None:
            self._optimizer.load_state_dict(serialization['_optimizer_state_dict'])
        if '_state_shape' in serialization:
            self._state_shape = serialization['_state_shape']
        if '_random_state' in serialization:
            random.setstate(serialization['_random_state'])
        if '_random_state_numpy' in serialization:
            np.random.set_state(serialization['_random_state_numpy'])
        if '_random_state_torch' in serialization:
            torch.set_rng_state(serialization['_random_state_torch'].cpu())

    def serialize(self):
        return {
            'type':                     self.__class__.__name__,
            '_actions':                 self._actions,
            '_batch_size':              self._batch_size,
            '_burn_in_steps':           self._burn_in_steps,
            '_curr_step':               self._curr_step,
            '_exploration_rate':        self._exploration_rate,
            '_exploration_rate_decay':  self._exploration_rate_decay,
            '_exploration_rate_min':    self._exploration_rate_min,
            '_dnn':                     self._dnn.serialize() if self._dnn is not None else None,
            '_discount_factor':         self._discount_factor,
            '_learning_interval':       self._learning_interval,
            '_learning_rate':           self._learning_rate,
            '_logger':                  self._logger.serialize(),
            '_optimizer_state_dict':    self._optimizer.state_dict() if self._optimizer is not None else None,
            '_state_shape':             self._state_shape,
            '_random_state':            random.getstate(),
            '_random_state_numpy':      np.random.get_state(),
            '_random_state_torch':      torch.get_rng_state(),
        }

    def act(self, state, exploration=True):
        raise NotImplementedError()

    def learn(self, state, action, next_state, reward, done):
        raise NotImplementedError()

    def save(self, file_path, save_log=False, save_memory=False):
        # save serialized agent to filesystem
        torch.save(self.serialize(), file_path)

        # save log
        if save_log:
            log_path = '{}.log.csv'.format(os.path.splitext(file_path)[0])
            self._logger.save(log_path, interval=100)

        # save memory to filesystem
        if save_memory:
            memory_path = '{}.memory.pickle'.format(os.path.splitext(file_path)[0])
            self._memory.save(memory_path)


class DQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # init dnn
        self._dnn = DQN(self._state_shape, len(self._actions)).to(self._device)
        self._optimizer = torch.optim.Adam(self._dnn.parameters(), lr=self._learning_rate)
        self._loss_fn = torch.nn.SmoothL1Loss().to(self._device)

    def act(self, state, exploration=True):
        # determine whether agent should explore or exploit
        if exploration and np.random.rand() < self._exploration_rate:
            # randomly choose next action
            action = np.random.randint(len(self._actions))
        else:
            with torch.no_grad():
                # reformat state
                state = state.__array__()
                state = torch.FloatTensor(state).to(self._device)
                state = state.unsqueeze(0)

                # determine q-values with dnn and choose optimal action
                q_values = self._dnn(state)
                action = torch.argmax(q_values, 1).item()

        # decrease exploration rate and increment step
        self._exploration_rate *= self._exploration_rate_decay
        self._exploration_rate = max(self._exploration_rate_min, self._exploration_rate)
        self._curr_step += 1

        return action

    def learn(self, state, action, next_state, reward, done):
        # record experience to memory
        self._memory.cache(state, action, next_state, reward, done)

        # check for burn-in delay
        if self._curr_step < self._burn_in_steps:
            return None, None

        # ensure enough memory has been stored for batch size
        if self._batch_size > self._memory.size:
            return None, None

        # check if it is time to learn
        if self._curr_step % self._learning_interval != 0:
            return None, None

        # sample from memory and convert to torch tensors
        sample = self._memory.recall(batch_size=self._batch_size)
        state = torch.FloatTensor(sample['state']).to(self._device)
        action = torch.LongTensor(sample['action']).to(self._device).squeeze()
        next_state = torch.FloatTensor(sample['next_state']).to(self._device)
        reward = torch.FloatTensor(sample['reward']).to(self._device).squeeze()
        done = torch.BoolTensor(sample['done']).to(self._device).squeeze()

        # compute td estimate: td_estimate = q(s, a)
        q_value = self._dnn(state)
        td_estimate = q_value[np.arange(self._batch_size), action]

        # compute td target
        with torch.no_grad():
            # compute optimal next action: a' = argmax_a(q(s', a))
            next_q_value = self._dnn(next_state)
            next_action = torch.argmax(next_q_value, 1)

            # compute discounted reward: td_target = r + gamma * q(s', a')
            td_target = next_q_value[np.arange(self._batch_size), next_action]
            td_target = (reward + (1 - done.float()) * self._discount_factor * td_target).float()

        # compute loss and backprop
        loss = self._loss_fn(td_estimate, td_target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # update priority for sampled memory
        self._memory.update_sample_priority(sample, loss.item())

        # return both q and loss values
        q_value = td_estimate.mean().item()
        loss_value = loss.item()
        return q_value, loss_value


class DoubleDQNAgent(Agent):
    """Ref: Hasselt et al., 2015, https://arxiv.org/abs/1509.06461
    """
    def __init__(self, *args, sync_interval=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self._sync_interval = sync_interval

        # init dnn
        self._dnn = DoubleDQN(self._state_shape, len(self._actions)).to(self._device)
        self._optimizer = torch.optim.Adam(self._dnn.parameters(), lr=self._learning_rate)
        self._loss_fn = torch.nn.SmoothL1Loss().to(self._device)

    def deserialize(self, serialization):
        super().deserialize(serialization)
        if '_sync_interval' in serialization:
            self._sync_interval = serialization['_sync_interval']

    def serialize(self):
        return {
            **(super().serialize()),
            '_sync_interval':   self._sync_interval,
        }

    def act(self, state, exploration=True):
        # determine whether agent should explore or exploit
        if exploration and np.random.rand() < self._exploration_rate:
            # randomly choose next action
            action = np.random.randint(len(self._actions))
        else:
            with torch.no_grad():
                # reformat state
                state = state.__array__()
                state = torch.FloatTensor(state).to(self._device)
                state = state.unsqueeze(0)

                # determine q-values with dnn and choose optimal action
                q_values = self._dnn(state, eval_type='online')
                action = torch.argmax(q_values, 1).item()

        # decrease exploration rate and increment step
        self._exploration_rate *= self._exploration_rate_decay
        self._exploration_rate = max(self._exploration_rate_min, self._exploration_rate)
        self._curr_step += 1

        return action

    def learn(self, state, action, next_state, reward, done):
        # record experience to memory
        self._memory.cache(state, action, next_state, reward, done)

        # check if it is time to sync networks
        if self._curr_step % self._sync_interval == 0:
            self._dnn.sync()

        # check for burn-in delay
        if self._curr_step < self._burn_in_steps:
            return None, None

        # ensure enough memory has been stored for batch size
        if self._batch_size > self._memory.size:
            return None, None

        # check if it is time to learn
        if self._curr_step % self._learning_interval != 0:
            return None, None

        # sample from memory and convert to torch tensors
        sample = self._memory.recall(batch_size=self._batch_size)
        state = torch.FloatTensor(sample['state']).to(self._device)
        action = torch.LongTensor(sample['action']).to(self._device).squeeze()
        next_state = torch.FloatTensor(sample['next_state']).to(self._device)
        reward = torch.FloatTensor(sample['reward']).to(self._device).squeeze()
        done = torch.BoolTensor(sample['done']).to(self._device).squeeze()

        # compute td estimate: td_estimate = q_online(s, a)
        q_value = self._dnn(state, eval_type='online')
        td_estimate = q_value[np.arange(self._batch_size), action]

        # compute td target
        with torch.no_grad():
            # compute optimal next action: a' = argmax_a(q_online(s', a))
            next_q_value = self._dnn(next_state, eval_type='online')
            next_action = torch.argmax(next_q_value, 1)

            # compute discounted reward: td_target = r + gamma * q_offline(s', a')
            next_q_value = self._dnn(next_state, eval_type='offline')
            td_target = next_q_value[np.arange(self._batch_size), next_action]
            td_target = (reward + (1 - done.float()) * self._discount_factor * td_target).float()

        # compute loss and backprop
        loss = self._loss_fn(td_estimate, td_target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # update priority for sampled memory
        self._memory.update_sample_priority(sample, loss.item())

        # return both q and loss values
        q_value = td_estimate.mean().item()
        loss_value = loss.item()
        return q_value, loss_value


class ClippedDoubleDQNAgent(DoubleDQN):
    """Ref: Fujimoto et al., 2018, https://arxiv.org/abs/1802.09477

    """
    def learn(self, state, action, next_state, reward, done):
        # record experience to memory
        self._memory.cache(state, action, next_state, reward, done)

        # check if it is time to sync networks
        if self._curr_step % self._sync_interval == 0:
            self._dnn.sync()

        # check for burn-in delay
        if self._curr_step < self._burn_in_steps:
            return None, None

        # ensure enough memory has been stored for batch size
        if self._batch_size > self._memory.size:
            return None, None

        # check if it is time to learn
        if self._curr_step % self._learning_interval != 0:
            return None, None

        # sample from memory and convert to torch tensors
        sample = self._memory.recall(batch_size=self._batch_size)
        state = torch.FloatTensor(sample['state']).to(self._device)
        action = torch.LongTensor(sample['action']).to(self._device).squeeze()
        next_state = torch.FloatTensor(sample['next_state']).to(self._device)
        reward = torch.FloatTensor(sample['reward']).to(self._device).squeeze()
        done = torch.BoolTensor(sample['done']).to(self._device).squeeze()

        # compute td estimate
        q_value1 = self._dnn(state, eval_type='online')
        q_value2 = self._dnn(state, eval_type='offline')
        td_estimate1 = q_value1[np.arange(self._batch_size), action]
        td_estimate2 = q_value2[np.arange(self._batch_size), action]

        # compute td target
        with torch.no_grad():
            # compute optimal next action
            next_q_value1 = self._dnn(next_state, eval_type='online')
            next_q_value2 = self._dnn(next_state, eval_type='offline')
            next_action1 = torch.argmax(next_q_value1, 1)
            next_action2 = torch.argmax(next_q_value2, 1)

            # compute discounted reward
            td_target1 = next_q_value1[np.arange(self._batch_size), next_action1]
            td_target2 = next_q_value2[np.arange(self._batch_size), next_action2]
            td_target = torch.min(td_target1, td_target2)
            td_target = (reward + (1 - done.float()) * self._discount_factor * td_target).float()

        # compute loss and backprop
        loss = (self._loss_fn(td_estimate1, td_target) + self._loss_fn(td_estimate2, td_target)) / 2
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # update priority for sampled memory
        self._memory.update_sample_priority(sample, loss.item())

        # return both q and loss values
        q_value = td_estimate1.mean().item()
        loss_value = loss.item()
        return q_value, loss_value
