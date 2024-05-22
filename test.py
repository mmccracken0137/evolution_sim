#!/usr/bin/env python
import numpy as np
from GenAgent import *
import matplotlib.pyplot as plt

plt.style.use('ggplot')
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (30,60)

dims = (100, 80)
n_agents = 500
mutation_prob = 0.01
n_random_barriers = 800
sim = Sim(dims[0], dims[1], n_agents, mutation_prob, ngenes=6, 
          scale=10, fps=10, diags=1, steps_per_gen=120)

arr = np.zeros(dims)
barr = np.zeros(dims)
for i in range(dims[0]):
    for j in range(dims[1]):
        if i > 0.9 * dims[0]: # or j > 0.8*self.dims[1]:
            arr[i, j] = 1
        # if (i-2*dims[0]/3)**2 + (j-2*dims[1]/3)**2 < 10**2:
        #     barr[i, j] = 1
        # if (i-3*dims[0]/5)**2 + (j-1*dims[1]/3)**2 < 10**2:
        #     barr[i, j] = 1

sim.set_repro_zones(arr)
sim.set_barriers(sim.random_barrier(n_random_barriers, min_x_ratio=0.01, max_x_ratio=0.85)) #barr)
sim.create_agents()

n_gens = 100
for i in range(n_gens):
    sim.run_generation()
    sim.kill_agents()
    sim.set_barriers(sim.random_barrier(n_random_barriers, min_x_ratio=0.01, max_x_ratio=0.85)) #barr)
    sim.reproduce_agents()


fig = plt.figure(figsize=(7,4))
gens = range(len(sim.survival_rates))
plt.scatter(gens, sim.survival_rates)
plt.show()
