import os

import numpy as np
from tqdm import tqdm

from data_methods import Env
from models import *

if __name__ == '__main__':
    models = [
        BiCNet(n_actions=10, n_features=2, hidsizes='128,64')
    ]

    for model in models:
        actions = []
        done = False
        obs = Env.reset(model.local_model, model.predictor)
        tqdm.write(f'running {model.name}')

        for _ in tqdm(range(2)):
            action = int(model.step(obs))
            actions.append(action)
            _, _, done, info = Env.step(action)

        power_usage, latency = info
        actions = np.bincount(actions, minlength=10)

        np.savetxt(os.path.join('logs', 'debug', f'rl_power.txt'), power_usage)
