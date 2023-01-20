import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import time


class Logger(object):
    def __init__(self):
        self._log = dict()
        self._start_time = None
        self._save_idx = 0

    def deserialize(self, serialization):
        if serialization.get('type') != self.__class__.__name__:
            raise ValueError('Incompatible serialization.')
        if '_log' in serialization:
            self._log = serialization['_log']

    def serialize(self):
        return {
            'type':     self.__class__.__name__,
            '_log':     self._log
        }

    def log(self, **kwargs):
        # compute elapsed time since last log entry
        if 'time' not in kwargs:
            if self._start_time is None:
                self._start_time = time.time()
            elapsed_time = time.time() - self._start_time
            elapsed_time = [*divmod(elapsed_time, 60.0)]
            elapsed_time = [*divmod(elapsed_time[0], 60.0), elapsed_time[1]]
            elapsed_time = '{:02.0f}:{:02.0f}:{:02.0f}'.format(*elapsed_time)
            kwargs = {'time': elapsed_time, **kwargs}

        # append log entry including each given value
        for k, v in kwargs.items():
            if k not in self._log:
                self._log[k] = list()
            self._log[k].append(v)

    def value(self, key, history=1):
        return np.mean(self._log[key][-history:])

    def values(self, key):
        return self._log[key]

    def save(self, file_path, interval=1):
        # verify file path is not empty
        if file_path is None or file_path == '':
            return

        # verify data has been logged
        if len(self._log) == 0:
            return

        # write log to csv file
        keys = [k for k in self._log.keys()]
        if self._save_idx == 0:
            # write header to csv file
            with open(file_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['eps'] + keys)

        # write each row of log to csv file
        with open(file_path, 'a') as f:
            writer = csv.writer(f)
            for i in range(self._save_idx, len(self._log[keys[0]])):
                if i % interval == 0:
                    a = max(0, i - interval + 1)
                    b = i + 1
                    values = [np.mean(v[a:b]) if isinstance(v[a:b], float) else v[i] for v in self._log.values()]
                    writer.writerow(['{}'.format(i)] + ['{}'.format(v) for v in values])

        # update save index
        self._save_idx = len(self._log[keys[0]])

        # create plots
        for metric in ['eps_length', 'eps_loss', 'eps_q_value', 'eps_reward', 'exploration_rate']:
            if metric in self._log:
                plot_path = '{}.{}.jpg'.format(os.path.splitext(file_path)[0], metric)
                plot_data = self._log[metric]
                #plot_data = np.convolve(plot_data, np.ones(interval), 'valid') / interval
                m, n = divmod(len(plot_data), interval)
                if m <= 0:
                    continue
                plot_data = np.average(np.array(plot_data[:m * interval]).reshape([m, interval]), axis=1)
                plt.plot(np.arange(0, m * interval, interval), plot_data)
                plt.savefig(plot_path)
                plt.clf()
