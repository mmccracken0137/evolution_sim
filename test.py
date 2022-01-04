#!/usr/bin/env python
import numpy as np
from GenAgent import *

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (30,60)

dims = (80, 80)
n_agents = 800
mutation_prob = 0.01
sim = Sim(dims[0], dims[1], n_agents, mutation_prob, ngenes=8, scale=8, fps=10, diags=1,
          steps_per_gen=50)

arr = np.zeros(dims)
for i in range(dims[0]):
    for j in range(dims[1]):
        if i > 0.9 * dims[0]: # or j > 0.8*self.dims[1]:
            arr[i, j] = 1

sim.set_repro_zones(arr)

n_gens = 10
for i in range(n_gens):
    sim.run_generation()
    sim.kill_agents()
    sim.reproduce_agents()


# b = Brain(7, [4], 5)
# genome = []
# for i in range(12):
#     genome.append(Gene(24, int(2**24 * np.random.rand())))
#
# b.build_connectome(genome)
# b.mats_product()
