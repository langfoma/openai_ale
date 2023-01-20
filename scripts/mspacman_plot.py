import argparse
import matplotlib.pyplot as plt
import numpy as np

from mspacman_rl.learning.agent import load_agent


def main(args):
    # read data
    values = dict()
    for agent_path in args.agent_paths:
        agent = load_agent(agent_path)
        if agent is None:
            print('Invalid agent.')
            exit()
        try:
            values[agent_path] = agent.logger.values(args.metric)
        except KeyError:
            print('Invalid metric.')
            exit()

    # plot data
    fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title(args.metric)
    ax.set_xlabel('epoch')
    ax.set_ylabel(args.metric)
    k = max(1, args.interval)
    for key, data in values.items():
        m, n = divmod(len(data), k)
        if m <= 0:
            continue
        data = np.average(np.array(data[:m * k]).reshape([m, k]), axis=1)
        ax.plot(np.arange(0, m * k, k), data, label=key)
    ax.legend(bbox_to_anchor=(1, -0.2), loc='upper right', borderaxespad=0)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # suppress warnings
    import warnings
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # manage command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('agent_paths', nargs='*', type=str, default=None)
    parser.add_argument('--metric', type=str, default='eps_loss')
    parser.add_argument('--interval', type=int, default=1)

    # run main function
    main(parser.parse_args())
