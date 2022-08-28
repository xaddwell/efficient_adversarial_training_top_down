import torch.nn as nn
import torch.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def draw_2midlayer_sim(name,sim_list,num=1,
                      normalization = True,color="red"):
    if normalization:
        sim_list = (sim_list-np.min(sim_list,axis=1)) / \
                   (np.max(sim_list,axis=1)-np.min(sim_list,axis=1))

    mean_sim = np.mean(sim_list,axis=1)[0]
    std_sim = np.std(sim_list,axis=1)[0]

    up_bound = list(mean_sim + num*std_sim)
    low_bound = list(mean_sim - num*std_sim)

    plt.plot(mean_sim,color = color,label = name)
    plt.fill_between(list(range(mean_sim.shape[-1])),
                     low_bound,up_bound,color = color,alpha = 0.2)
    plt.legend()


if __name__ == "__main__":

    sim_list = np.random.rand(1,256,100)

