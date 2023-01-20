import numpy as np
import pickle


class Memory(object):
    def __init__(self, limit, prioritized=False, priority_alpha=1.0, priority_default=1e6):
        self._store = dict()
        self._limit = limit
        self._size = 0
        self._index = 0
        self._prioritized = prioritized
        self._priority_alpha = priority_alpha
        self._priority_default = priority_default
        self._last_sample_idx = None

    @property
    def limit(self):
        return self._limit

    @property
    def prioritized(self):
        return self._prioritized

    @property
    def size(self):
        return self._size

    def cache(self, state, action, next_state, reward, done):
        # update memory location and size
        self._size = min(self._size + 1, self._limit)
        self._index = (self._index + 1) % self._size

        # reformat data to ndarrays
        state = np.array(state.__array__() * 255, dtype=np.uint8)
        action = np.array([action], dtype=np.uint8)
        next_state = np.array(next_state.__array__() * 255, dtype=np.uint8)
        reward = np.array([reward], dtype=np.float32)
        done = np.array([done], dtype=np.bool)

        # init memory store for data records
        if 'state' not in self._store:
            self._store['state'] = np.zeros((self._limit,) + state.shape, dtype=state.dtype)
        if 'action' not in self._store:
            self._store['action'] = np.zeros((self._limit,) + action.shape, dtype=action.dtype)
        if 'next_state' not in self._store:
            self._store['next_state'] = np.zeros((self._limit,) + next_state.shape, dtype=next_state.dtype)
        if 'reward' not in self._store:
            self._store['reward'] = np.zeros((self._limit,) + reward.shape, dtype=reward.dtype)
        if 'done' not in self._store:
            self._store['done'] = np.zeros((self._limit,) + done.shape, dtype=done.dtype)

        # check if memory is prioritized
        if self._prioritized:
            # init memory store for priorities
            if 'priority' not in self._store:
                self._store['priority'] = np.zeros(self._limit, dtype=np.float32)

            # set index to the lowest priority entry if memory store is full
            if self._size >= self._limit:
                self._index = np.argmin(self._store['priority'])

            # record default priority for data
            self._store['priority'][self._index] = self._priority_default

        # record data to current memory location
        self._store['state'][self._index] = state
        self._store['action'][self._index] = action
        self._store['next_state'][self._index] = next_state
        self._store['reward'][self._index] = reward
        self._store['done'][self._index] = done

    def load(self, file_path):
        # read pickle
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # deserialize data
        if 'type' in data and data['type'] != self.__class__.__name__:
            raise ValueError('Incompatible serialization.')
        if '_index' in data:
            self._index = data['_index']
        if '_limit' in data:
            self._limit = data['_limit']
        if '_size' in data:
            self._size = data['_size']
        if '_store' in data:
            self._store = data['_store']
        if '_prioritized' in data:
            self._prioritized = data['_prioritized']
        if '_priority_alpha' in data:
            self._priority_alpha = data['_priority_alpha']
        if '_priority_default' in data:
            self._priority_default = data['_priority_default']

    def save(self, file_path):
        # serialize data
        data = {
            'type':                 self.__class__.__name__,
            '_index':               self._index,
            '_limit':               self._limit,
            '_size':                self._size,
            '_store':               self._store,
            '_prioritized':         self._prioritized,
            '_priority_alpha':      self._priority_alpha,
            '_priority_default':    self._priority_default,
        }

        # write memory data to pickle
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def recall(self, batch_size=1):
        # randomly sample batch from memory
        if self._prioritized:
            # sample based on priorities
            p_dist = np.power(self._store['priority'][:self._size], self._priority_alpha)
            p_dist /= np.sum(p_dist)
            sample_idx = np.random.choice(np.arange(self._size), size=batch_size, replace=False, p=p_dist)
        else:
            # uniform sampling
            sample_idx = np.random.choice(np.arange(self._size), size=batch_size, replace=False)

        # get sample from memory store
        sample = {k: v[sample_idx] for k, v in self._store.items()}

        # convert stateervations back to floats
        sample['state'] = np.float32(sample['state']) / 255
        sample['next_state'] = np.float32(sample['next_state']) / 255

        # save sampled indices and return sample
        sample['idx'] = sample_idx
        return sample

    def update_sample_priority(self, sample, priority):
        # do nothing if memory is not prioritized
        if not self._prioritized:
            return

        # do nothing if no indices are included with sample
        if 'idx' not in sample:
            return

        # update priorities for given sample
        sample_idx = sample['idx']
        self._store['priority'][sample_idx] = priority
