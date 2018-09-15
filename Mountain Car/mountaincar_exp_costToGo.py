# -*- coding: utf-8 -*-
#!/usr/bin/env python

from rl_glue import *  # Required for RL-Glue
RLGlue("mountaincar", "sarsa_lambda_agent")
import numpy as np

if __name__ == "__main__":
    import sys

    '''
    REFERENCE: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    '''
    def progress(count, total, prefix=''):
        bar_len = 40
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '█' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('\r%s [%s] %s%s ' % (prefix, bar, percents, '%'))
        sys.stdout.flush()  # As suggested by Rom Ruben
    '''
    REFERENCE: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    '''

    num_episodes = 1000
    progress(0, num_episodes, prefix ='Progress: ')

    RL_init()
    for i in range(num_episodes):
         RL_episode(0)
         progress(i + 1, num_episodes, prefix = 'Progress:')
    RL_agent_message("ValueFunction")
    RL_cleanup()