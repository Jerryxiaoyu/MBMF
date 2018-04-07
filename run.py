import os
from  math import pi
import numpy as np


def sample_goals(num_goals):
    #return np.random.uniform(0.0, 3.0, (num_goals,))  #随机生成目标
    return np.arange(-180,180,360/(num_goals))      #序列生成目标





os.system("python main.py --env_name 'HalfCheetah-v2' -n 10 -r 700  -d 10 -ep 1000 -m 10 -sp 1000")
os.system("python main.py --env_name 'My3-v1' 		  -n 10 -r 1000 -d 10 -ep 1000 -m 10 -sp 3000")
os.system("python main.py --env_name 'Ant-v1' 		  -n 10 -r 1000 -d 10 -ep 1000 -m 10 -sp 3000")
os.system("python main.py --env_name 'HalfCheetah-v2' -n 10 -r 700  -d 10 -ep 1000 -m 10 -sp 500")
os.system("python main.py --env_name 'HalfCheetah-v2' -n 10 -r 700  -d 10 -ep 1000 -m 10 -sp 200")
os.system("python main.py --env_name 'My3-v1' 		  -n 10 -r 1000 -d 10 -ep 1000 -m 10 -sp 1000")

#os.system("python main.py --env_name 'HalfCheetah-v2' -n 2 -r 10 -d 2 -ep 1000 -m 1 -sp 10")
# os.system("python main.py --env_name 'My3-v1' 		  -n 2 -r 10 -d 2 -ep 1000 -m 1 -sp 2")
# os.system("python main.py --env_name 'Ant-v1' 		  -n 2 -r 10 -d 2 -ep 1000 -m 1 -sp 2")