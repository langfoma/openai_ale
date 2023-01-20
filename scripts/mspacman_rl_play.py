import argparse
import datetime
import gym
import os
import numpy as np
import pickle
import pygame

from mspacman_rl.utils.progress import ProgressDisplay

# set size of default game screen
GAME_SCREEN_SIZE = (160, 210)


def main(args):
    # init environment
    env = gym.make('MsPacman-v0')
    env.reset()

    if not args.replay:
        play(env, args)
    else:
        # load existing recording
        if args.recording_path is not None and args.recording_path != '' and os.path.exists(args.recording_path):
            with open(args.recording_path, 'rb') as f:
                recording = pickle.load(f)

            # replay past recording
            replay(env, recording, args)


def play(env, args):
    # init recording
    recording = dict()
    if args.actions_only:
        recording['action'] = list()
    else:
        recording['state'] = list()
        recording['action'] = list()
        recording['next_state'] = list()
        recording['reward'] = list()
        recording['done'] = list()

    # init display
    pygame.init()
    display_size = (GAME_SCREEN_SIZE[0] * args.display_scale, GAME_SCREEN_SIZE[1] * args.display_scale)
    display_screen = pygame.display.set_mode(display_size)
    game_screen = pygame.transform.scale(display_screen.copy(), GAME_SCREEN_SIZE)
    clock = pygame.time.Clock()

    # directions mapped on keyboard
    key_actions = {
        'q': 7, 'w': 2, 'e': 6,
        'a': 4, 's': 0, 'd': 3,
        'z': 9, 'x': 5, 'c': 8,
    }
    key_actions = {ord(key): value for key, value in key_actions.items()}

    try:
        # game loop
        state = env.reset()
        last_key = ord('s')
        done = False
        while not done:
            # process action
            action = key_actions[last_key]
            next_state, reward, done, info = env.step(action)

            # update recording
            if 'state' in recording:
                recording['state'].append(state)
            if 'action' in recording:
                recording['action'].append(action)
            if 'next_state' in recording:
                recording['next_state'].append(next_state)
            if 'reward' in recording:
                recording['reward'].append(reward)
            if 'done' in recording:
                recording['done'].append(done)

            # render environment
            img = env.render(mode='rgb_array')
            display_size = (GAME_SCREEN_SIZE[0] * args.display_scale, GAME_SCREEN_SIZE[1] * args.display_scale)
            pygame.surfarray.blit_array(game_screen, np.transpose(img, [1, 0, 2]))
            pygame.transform.scale(game_screen, display_size, display_screen)
            pygame.display.flip()

            # get keyboard input
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    key = event.dict['key']
                    if key in key_actions:
                        last_key = key

            # clock delay
            clock.tick(args.refresh_rate)

            # shift current state to next state
            state = next_state
    except KeyboardInterrupt:
        pass
    finally:
        # save recording to filesystem
        if args.recording_path is not None and args.recording_path != '':
            with open(args.recording_path, 'wb') as f:
                pickle.dump(recording, f)


def replay(env, recording, args):
    # get actions from recording
    if 'action' not in recording:
        return
    actions = recording['action']

    # init progress bar
    progress = ProgressDisplay(bar_length=20, show_steps=False, show_time=False)

    # init episode metrics
    eps_length = 0
    eps_reward = 0

    # init display
    pygame.init()
    display_size = (GAME_SCREEN_SIZE[0] * args.display_scale, GAME_SCREEN_SIZE[1] * args.display_scale)
    display_screen = pygame.display.set_mode(display_size)
    game_screen = pygame.transform.scale(display_screen.copy(), GAME_SCREEN_SIZE)
    clock = pygame.time.Clock()

    done = False
    while not done and eps_length < len(actions):
        # perform recorded action on environment
        action = actions[max(0, min(eps_length, len(actions) - 1))]
        next_state, reward, done, info = env.step(action)

        # render environment
        img = env.render(mode='rgb_array')
        pygame.surfarray.blit_array(game_screen, np.transpose(img, [1, 0, 2]))
        pygame.transform.scale(game_screen, display_size, display_screen)
        pygame.display.flip()

        # shift to next step
        eps_length += 1
        eps_reward += reward

        # update progress
        progress.update(eps_length, len(actions), suffix_text=' - '.join([
            'length: {:.0f}'.format(eps_length),
            'reward: {:.0f}'.format(eps_reward),
        ]))

        # clock delay
        clock.tick(args.refresh_rate)

    # terminate progress bar
    progress.terminate()


if __name__ == '__main__':
    # suppress warnings
    import warnings
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # create timestamp for now
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')

    # manage command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('recording_path', nargs='?', type=str, default=None)
    parser.add_argument('--actions-only', action='store_true')
    parser.add_argument('--display-scale', type=float, default=2.0)
    parser.add_argument('--refresh-rate', type=float, default=15)
    parser.add_argument('--replay', action='store_true')

    # run main function
    main(parser.parse_args())
