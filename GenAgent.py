#!/usr/bin/env python
import numpy as np
import os, sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import *
from rich import print
import timeit

def bitstr_2_decimal(s):
    return int(s, 2)

class Sim:

    def __init__(self, xdim, ydim, n_agents, mutate_prob, fps=10, render=True,
                 ngenes=4, genebits=24, scale=1, steps_per_gen = 250, diags=True,
                 n_hidden=[4], barriers=[]):
        self.dims = [xdim, ydim]
        self.scale = scale
        self.n_agents = n_agents
        self.agent_objs = []
        self.fps = fps
        self.FramePerSec = pygame.time.Clock()
        self.render = render
        self.n_genes = ngenes
        self.genebits = genebits
        self.steps_per_gen = steps_per_gen
        self.mutate_prob = mutate_prob
        self.diags = diags
        self.n_inputs = 0
        self.n_hidden = n_hidden
        self.generation_number = 0
        self.survival_rates = []

        self.survivals = []

        # TKTKTK add different codes for genders, barriers,
        self.ag_locations = np.zeros(self.dims, dtype=int)
        self.barriers = np.zeros(self.dims, dtype=int)
        self.repro_zones = np.zeros(self.dims, dtype=int)

        self.world = World(self.dims[0], self.dims[1], scale=self.scale)
        if render:
            pygame.display.update()

        # self.agent_objs = self.create_agents()

    def set_repro_zones(self, arr=[[]]):
        self.repro_zones = arr

    def set_barriers(self, arr):
        self.barriers = arr

    def draw_repro_zones(self):
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                if self.repro_zones[i, j] == 1:
                    pygame.draw.rect(self.world.disp, (85, 85, 65),
                                     pygame.Rect((i + 0.5) * self.scale,
                                                 (j + 0.5) * self.scale,
                                                 self.scale, self.scale))

    def draw_barriers(self):
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                if self.barriers[i, j] == 1:
                    pygame.draw.rect(self.world.disp, (0, 0, 0),
                                     pygame.Rect((i + 0.5) * self.scale,
                                                 (j + 0.5) * self.scale,
                                                 self.scale, self.scale))

    def get_neighbors(self, x, y):
        neighbors = np.zeros((3, 3))
        for i in range(-1, 2):
            for j in range(-1, 2):
                if x + i < 0 or x + i >= self.dims[0]:
                    neighbors[i + 1, j + 1] = 2
                elif y + j < 0 or y + j >= self.dims[1]:
                    neighbors[i + 1, j + 1] = 2
                else:
                    neighbors[i + 1, j + 1] = self.ag_locations[x + i, y + j]
                neighbors[1, 1] = 0
        return neighbors

    def get_barriers(self, x, y):
        barrs = np.zeros((3, 3))
        for i in range(-1, 2):
            for j in range(-1, 2):
                if x + i < 0 or x + i >= self.dims[0]:
                    barrs[i + 1, j + 1] = 2
                elif y + j < 0 or y + j >= self.dims[1]:
                    barrs[i + 1, j + 1] = 2
                else:
                    barrs[i + 1, j + 1] = self.barriers[x + i, y + j]
                barrs[1, 1] = 0
        return barrs

    def random_barrier(self, n_blocks, min_x_ratio=0.15, max_x_ratio=0.85):
        arr = np.zeros(self.dims)
        for i in range(n_blocks):
            xr, yr = -1, -1
            while (xr < min_x_ratio * self.dims[0] or xr > max_x_ratio * self.dims[0]) and arr[xr, yr] == 0:
                xr, yr = int(np.random.rand() * self.dims[0]), int(np.random.rand() * self.dims[1])
            arr[xr, yr] = 1
        return arr

    def create_agents(self):
        ags = []
        for i in range(self.n_agents):
            genome = []
            for j in range(self.n_genes):
                genome.append(Gene(self.genebits, int(np.random.rand() * 2**self.genebits)))
            repeat = 1
            x, y = 0, 0
            while repeat:
                x = int(np.random.rand() * self.dims[0])
                y = int(np.random.rand() * self.dims[1])
                if self.ag_locations[x, y] == 0 and self.barriers[x, y] == 0:
                    repeat = 0
                    self.ag_locations[x, y] += 1
            ags.append(Agent(i, x, y, genome, self.genebits))
            if i == 0:
                self.n_inputs = len(ags[0].sense(self))
                self.n_outputs = ags[0].n_outputs
            ags[-1].build_brain(self.n_inputs, self.n_hidden, self.n_outputs)
        self.agent_objs = ags

    def kill_agents(self):
        del_idxs = []
        for i, ag in enumerate(self.agent_objs):
            if self.repro_zones[ag.pos[0], ag.pos[1]] == 0:
                del_idxs.append(i)
                self.ag_locations[ag.pos[0], ag.pos[1]] -= 1
        del_idxs = sorted(del_idxs, reverse=True)
        for i in del_idxs:
            if i < len(self.agent_objs):
                self.agent_objs.pop(i)
        self.survival_rates.append(len(self.agent_objs) / self.n_agents)
        print('gen', str(self.generation_number - 1).zfill(3), 'survival rate -->',
              self.survival_rates[-1])

        # redraw survivors...
        # for i in range(10):
        self.world.disp.fill(self.world.bkgd)
        self.draw_repro_zones()
        self.draw_barriers()

        for i, ag in enumerate(self.agent_objs):
            # col = ag.color + (100 - i * 10, )
            pygame.draw.circle(self.world.disp, ag.color,
                               ((ag.pos[0] + 1) * self.scale, (ag.pos[1] + 1) * self.scale),
                               self.scale / 2)
        # render text
        timestamp = "t = end generation " + str(self.generation_number)
        self.world.draw_step_label(timestamp)
        pygame.display.update()
        self.FramePerSec.tick(self.fps)

    def reproduce_agents(self, mutate=True):
        self.ag_locations = np.zeros(self.dims, dtype=int)
        next_gen = []
        for i in range(self.n_agents):
            parent = np.random.choice(self.agent_objs)
            genome = []
            for g in parent.genome:
                if mutate:
                    g.mutate(self.mutate_prob)
                genome.append(Gene(self.genebits, g.decimal_val))
            repeat = 1
            x, y = 0, 0
            while repeat:
                x = int(np.random.rand() * self.dims[0])
                y = int(np.random.rand() * self.dims[1])
                if self.ag_locations[x, y] == 0 and self.barriers[x, y] == 0:
                    repeat = 0
                    self.ag_locations[x, y] += 1
            next_gen.append(Agent(i, x, y, genome, self.genebits))
            next_gen[-1].build_brain(self.n_inputs, self.n_hidden, self.n_outputs)
        self.agent_objs = next_gen

    def run_generation(self, draw=True, draw_edges=False):
        t = 0
        while t < self.steps_per_gen:
            if self.render:
                self.world.disp.fill(self.world.bkgd)
                # draw repro zones
                self.draw_repro_zones()
                self.draw_barriers()

                for i, ag in enumerate(self.agent_objs):
                    ag_inputs = ag.sense(self)
                    neighbors = ag.neighbors.flatten()
                    neigh_barriers = ag.neigh_barriers.flatten()
                    r = ag.brain.probable_move(ag_inputs)  #TKTKTK
                    # r = np.random.randint(0, high=len(neighbors))
                    # while neighbors[r] != 0:
                    #     r = np.random.randint(len(neighbors))
                    # ag.move(self, ag.move_incr[r])
                    if neighbors[r] == 0 and neigh_barriers[r] == 0:
                        ag.move(self, ag.move_incr[r])

                    pygame.draw.circle(self.world.disp, ag.color,
                                       ((ag.pos[0] + 1) * self.scale, (ag.pos[1] + 1) * self.scale),
                                       self.scale / 2)
                    if draw_edges:
                        pygame.draw.circle(self.world.disp, ag.edge_color,
                                       ((ag.pos[0] + 1) * self.scale, (ag.pos[1] + 1) * self.scale),
                                       self.scale / 2, width=1)
                # render text
                timestamp = "t = "
                for i in range(3 - len(str(t))): timestamp += " "
                timestamp += str(t)
                self.world.draw_step_label(timestamp)
                genstamp = 'generation = ' + str(self.generation_number)
                self.world.draw_gen_label(genstamp)
                pygame.display.update()

                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
            self.FramePerSec.tick(self.fps)
            t += 1
        self.generation_number += 1


class World:

    def __init__(self, xdim, ydim, scale=1, bkgd=(65, 65, 65)):
        pygame.init()
        self.ypad = 20
        self.xpad = 20
        self.xdim = xdim
        self.ydim = ydim
        self.scale = scale
        self.disp = pygame.display.set_mode(((xdim + 1) * scale, (ydim + 1) * scale + self.ypad))
        self.bkgd = bkgd
        self.disp.fill(self.bkgd)
        pygame.display.set_caption("world")
        self.font = pygame.font.SysFont("monospace", 16)

    def draw_step_label(self, txt):
        label = self.font.render(txt, 1, (255,255,255))
        self.disp.blit(label, (5, self.ydim * self.scale + self.ypad - 12))

    def draw_gen_label(self, txt):
        label = self.font.render(txt, 1, (255,255,255))
        self.disp.blit(label, (self.xdim * self.scale / 2,
                               self.ydim * self.scale + self.ypad - 12))

class Agent:

    def __init__(self, id, x, y, genome, n_bits):
        self.id = id
        self.genome = genome
        self.n_genes = len(genome)
        self.n_bits = n_bits
        self.pos = np.array([x, y])
        self.prevpos = self.pos

        self.move_incr = [[-1, -1], [-1, 0], [-1, 1],
                          [0, -1], [0, 0], [0, 1],
                          [1, -1], [1, 0], [1, 1]]
        self.n_outputs = len(self.move_incr)
        self.neighbors = np.zeros((3, 3))
        self.n_senses = 0

        g_str = ''
        for g in self.genome:
            g_str += g.bit_str
        total_bits = len(g_str)
        chomp = int(total_bits / 3)
        self.color = (int(g_str[0:chomp], 2) / 2**chomp * 255,
                      int(g_str[chomp:2*chomp], 2) / 2**chomp * 255,
                      int(g_str[2*chomp:3*chomp], 2) / 2**chomp * 255)
        self.edge_color = self.color

    def build_brain(self, n_inputs, n_hidden, n_outputs):
        self.brain = Brain(n_inputs, n_hidden, n_outputs)
        self.brain.build_connectome(self.genome)

    def move(self, sim, d=[0,0]):
        self.prevpos = self.pos
        self.pos = self.pos + np.array(d)
        sim.ag_locations[self.prevpos[0], self.prevpos[1]] = 0
        sim.ag_locations[self.pos[0], self.pos[1]] = 1

    def sense(self, sim):
        inputs = []

        self.neighbors = sim.get_neighbors(self.pos[0], self.pos[1])
        if not sim.diags:
            self.neighbors[0, 0] = 2
            self.neighbors[2, 0] = 2
            self.neighbors[0, 2] = 2
            self.neighbors[2, 2] = 2

        # TKTKTKTK add barriers to sense outputs
        self.neigh_barriers = sim.get_barriers(self.pos[0], self.pos[1])

        # inputs.append(self.pos[0]) # x position
        # inputs.append(self.pos[1]) # y position
        # inputs.append(sim.dims[0] - self.pos[0]) # distance from max x
        # inputs.append(sim.dims[1] - self.pos[1]) # distance from max y

        # TKTKTKTK sense barriers???

        inputs.append(self.prevpos[0]) # previous x
        inputs.append(self.prevpos[1]) # previous x

        inputs.append(np.sum(sim.ag_locations[self.pos[0], :])) # other agents in column
        inputs.append(np.sum(sim.ag_locations[:, self.pos[1]])) # other agents in row

        inputs.append(np.random.rand())

        inputs = np.concatenate((self.neighbors.flatten(), self.neigh_barriers.flatten(), inputs))

        self.n_senses = len(inputs.flatten())
        return inputs

    # def parse_genome():
    # def parse_gene():

class Brain:

    def __init__(self, n_inputs, hidden_nodes, n_outputs):
        self.n_inputs = n_inputs
        self.hidden_nodes = hidden_nodes # list of numbers of hidden nodes
        self.n_outputs = n_outputs

        self.mats = []
        if len(hidden_nodes) == 0:
            self.mats.append(np.zeros((n_inputs, n_outputs)))
        else:
            self.mats.append(np.zeros((n_inputs, hidden_nodes[0])))
            for i in range(len(hidden_nodes) - 1):
                self.mats.append(np.zeros((hidden_nodes[i], hidden_nodes[i+1])))
            self.mats.append(np.zeros((hidden_nodes[-1], n_outputs)))

    def build_connectome(self, genome):
        # a genome is a list of gene objects
        # loop over genes and turn on elements in mats
        for g in genome:
            g.print_gene
            m_idx = g.in_layer % len(self.mats)
            row = g.in_node % self.mats[m_idx].shape[0]
            col = g.out_node % self.mats[m_idx].shape[1]
            self.mats[m_idx][row][col] = g.weight

        self.matrix = self.mats[0]
        for i in range(len(self.mats) - 1):
            self.matrix = np.matmul(self.matrix, self.mats[i + 1])

    def softmax(self, vals):
        vals = np.exp(np.array(vals))
        return vals / np.sum(vals)

    def probable_move(self, inputs):
        inputs = np.reshape(np.array(inputs), (1, len(inputs)))
        outs = np.matmul(inputs, self.matrix)
        outs = self.softmax(outs).flatten()
        max_idx = np.argmax(outs) # get idx of first max value
        idxs = np.argwhere(outs == outs[max_idx]).flatten()
        return np.random.choice(idxs)

class Gene:

    def __init__(self, n_bits, decimal_val):
        self.set_values(n_bits, decimal_val)

    def set_values(self, n_bits, decimal_val):
        self.n_bits = n_bits
        self.decimal_val = decimal_val
        self.binary_str = format(self.decimal_val, '#026b')
        self.bit_str = self.binary_str[2:]
        self.binary_val = bin(decimal_val)

        self.in_layer = int(self.bit_str[0:4], 2)
        self.in_node = int(self.bit_str[4:10], 2)
        self.out_node = int(self.bit_str[10:16], 2)
        self.weight_sign_bit = int(self.bit_str[16])
        self.weight_sign = 1
        if self.weight_sign_bit == 1:
            self.weight_sign = -1
        self.weight_val = int(self.bit_str[17:], 2)
        # weight value is hard-coded as 7 bits here...
        self.weight = self.weight_sign * self.weight_val / 2**7

    def mutate(self, rate=0.0):
        r = np.random.rand()
        n_flips = int(rate / r)
        s = self.bit_str
        for i in range(n_flips):
            idx = np.random.randint(0, len(self.bit_str))
            repl = str(abs(int(s[idx]) - 1))
            s = s[:idx] + repl + s[idx+1:]
        self.set_values(self.n_bits, bitstr_2_decimal(s))

    def print_gene(self):
        print('gene:\t\t', self.binary_val, '\t', int(self.binary_val, 2))
        print('input layer:\t', bin(self.in_layer), '\t\t\t', self.in_layer)
        print('input node:\t', bin(self.in_node), '\t\t\t', self.in_node)
        print('output node:\t', bin(self.out_node), '\t\t\t', self.out_node)
        print('weight sign bit:', bin(self.weight_sign_bit))
        print('weight val:\t', bin(self.weight_val), '\t\t\t', self.weight_val)
        print('weight:\t\t', self.weight)
