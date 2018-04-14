import os
from  math import pi
import numpy as np


def sample_goals(num_goals):
    #return np.random.uniform(0.0, 3.0, (num_goals,))  #随机生成目标
    return np.arange(-180,180,360/(num_goals))      #序列生成目标





os.system("python main.py --env_name 'HalfCheetah-v2' -n 5  -nd 60 -b 512 -r 700  -d 10 -ep 1000 -m 15 -sp 10000  ")
os.system("python main.py --env_name 'HalfCheetah-v2' -n 5  -nd 30 -b 512 -r 700  -d 10 -ep 1000 -m 15 -sp 10000 ")

