'''
(1) set up a hypercubue simulation box with unit edge length in d-dimensinonal Euclidean space with peridoic boundary conditions;
(2) randomly populate N points in the box (as centers of the hyperspheres with radius R) [make sure N, R, L are chosen such that they are significant initial overlaps between the hyperspheres; make sure the coordinates of the centers are relative coorindates w.r.t. the box]
(3) obtain the contact neighbor list for each hypersphere;
(4) Start the relaxation iterations:
    (4a) isotropically rescale the box to L+dL, so the distance between the hypersphere centers are also rescaled with the box;
    (4b) loop for the list of hyperspheres and compute the total force on each hypersphere i, i.e.,
    F_total = Sum_j F_ij (note this is a vector sum)
where F_ij is the force pointing from i from overlapping neighbor j, i.e,
   F_ij = max[0, c*[R_i + R_j - d_ij] (note F_ij is a vector)
where c is a force constant, d_ij is the center-center distance bwtween hypersphere i and j
    (4c) move the hypersphere i with displacement dx = k*F_total (Note dx is a vector)
(5) repeat steps in (4) until the packing is overlapping free up to a tolerance


Test on higher dimensions
Fix the contact list problem
Partition the space in halfs (new idea)?
Check time complexity
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import math
from scipy.special import gamma


def get_truncated_normal(mean, sd, low, upp):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class Sphere:

    def __init__(self, dimension, center_pos, radius=None, neighbor_list=None):
        self.dimension = dimension
        self.center_pos = center_pos
        # might want to change this list later
        self.neighbor_list = neighbor_list if neighbor_list is not None else []
        self.radius = radius if radius is not None else 1

    def clear_neighbor_list(self):
        self.neighbor_list = []


class Packing:

    c = .4  # force constant
    change = 1.00001  # isotropically resize constant
    radius = 1

    def __init__(self, dimension, numspheres, length, spheres=None, relaxed=None):
        self.dimension = dimension
        self.numspheres = numspheres
        self.length = length
        self.spheres = spheres if spheres is not None else np.empty(
            numspheres, dtype=object)
        self.relaxed = relaxed if relaxed is not None else False

    # draws the packing in 2 dimensions, method is good
    def draw(self):
        plt.axes()
        boundary = plt.Rectangle((-self.length/2, -self.length/2),
                                 self.length, self.length, fc='None', ec='black')
        plt.gca().add_patch(boundary)
        plt.axis('scaled')
        for i in range(len(self.spheres)):
            circle = plt.Circle((self.spheres[i].center_pos[0], self.spheres[i].center_pos[1]),
                                self.radius, fc='None', ec='black')
            plt.gca().add_patch(circle)
            plt.plot(self.spheres[i].center_pos[0],
                     self.spheres[i].center_pos[1], 'bo')
        plt.show()

    def initial_fill(self):
        # self.radius = 1.5/2*(self.length/(self.numspheres**(1/self.dimension)))
        self.radius = 1
        for i in range(self.numspheres):
            x = get_truncated_normal(
                0, self.length/4, -self.length/2, self.length/2)
            center = x.rvs(size=self.dimension)
            self.spheres[i] = Sphere(
                self.dimension, np.array(center), self.radius)

    # might be able to find a more efficient way to reuse the previous neighbor_list
    def find_contact(self):
        self.relaxed = True
        for i in range(self.numspheres):
            self.spheres[i].clear_neighbor_list()
            for j in range(i+1, self.numspheres):
                diff = np.subtract(
                    self.spheres[i].center_pos, self.spheres[j].center_pos)
                if np.linalg.norm(diff) < 2*self.radius:
                    self.spheres[i].neighbor_list.append(np.array([j, 0]))
                    self.relaxed = False
                # kind of an approximate method, maybe go back and look at it with math later
                elif np.linalg.norm(diff) > self.length - 2*self.radius:
                    newcenter1 = self.spheres[i].center_pos.copy()
                    newcenter2 = self.spheres[j].center_pos.copy()
                    diff = np.subtract(newcenter1, newcenter2)
                    for k in range(self.dimension):
                        if abs(diff[k]) > self.length - 2*self.radius:
                            if self.spheres[i].center_pos[k] > self.length/2 - self.radius:
                                newcenter1[k] = newcenter1[k] - self.length
                            elif self.spheres[i].center_pos[k] < -self.length/2 + self.radius:
                                newcenter1[k] = newcenter1[k] + self.length
                            elif self.spheres[j].center_pos[k] > self.length/2 - self.radius:
                                newcenter2[k] = newcenter2[k] - self.length
                            elif self.spheres[j].center_pos[k] < -self.length/2 + self.radius:
                                newcenter2[k] = newcenter2[k] + self.length
                    diff = np.subtract(newcenter1, newcenter2)
                    if np.linalg.norm(diff) < 2*self.radius:
                        self.spheres[i].neighbor_list.append(np.array([j, 1]))
                        self.relaxed = False

    def relax_once(self):

        # isotropically rescale
        self.length = Packing.change * self.length
        for i in range(self.numspheres):
            self.spheres[i].center_pos = self.spheres[i].center_pos * \
                Packing.change

        # displace with forces
        sum = np.zeros((self.numspheres, self.dimension))
        for iindex in range(self.numspheres):
            for jobj in self.spheres[iindex].neighbor_list:
                j = self.spheres[jobj[0]]
                jindex = jobj[0]
                jkey = jobj[1]
                i = self.spheres[iindex]
                # represents direct contact between spheres
                if jkey == 0:
                    jtoi = np.subtract(
                        i.center_pos, j.center_pos)  # represents i - j
                    norm = np.linalg.norm(jtoi)
                    force = max(
                        0, (2 * self.radius - norm) * Packing.c)
                    sum[iindex, :] = sum[iindex, :] + force * jtoi / norm
                    sum[jindex, :] = sum[iindex, :] + -1 * force * jtoi / norm

                # represents wrapped contact between spheres
                elif jkey == 1:
                    newcenter1 = i.center_pos.copy()
                    newcenter2 = j.center_pos.copy()
                    diff = np.subtract(newcenter1, newcenter2)
                    for k in range(self.dimension):
                        if abs(diff[k]) > self.length - 2*self.radius:
                            if i.center_pos[k] > self.length/2 - self.radius:
                                newcenter1[k] = newcenter1[k] - self.length
                            elif i.center_pos[k] < -self.length/2 + self.radius:
                                newcenter1[k] = newcenter1[k] + self.length
                            elif j.center_pos[k] > self.length/2 - self.radius:
                                newcenter2[k] = newcenter2[k] - self.length
                            elif j.center_pos[k] < -self.length/2 + self.radius:
                                newcenter2[k] = newcenter2[k] + self.length

                    twotoone = np.subtract(newcenter1, newcenter2)
                    norm = np.linalg.norm(twotoone)
                    force = max(
                        0, (2 * self.radius - norm) * Packing.c)
                    sum[iindex, :] = sum[iindex, :] + force * twotoone / norm
                    sum[jindex, :] = sum[iindex, :] + - \
                        1 * force * twotoone / norm

        # apply the displacement and wrap locations
        for i in range(self.numspheres):
            for j in range(self.dimension):
                newcenter = self.spheres[i].center_pos[j] + sum[i, j]
                while newcenter > self.length/2:
                    newcenter -= self.length
                while newcenter < -self.length/2:
                    newcenter += self.length
                self.spheres[i].center_pos[j] = newcenter

    def relax(self):  # improve the number of steps between checking neighbors
        i = 0
        self.find_contact()
        while not self.relaxed:
            self.relax_once()
            i += 1
            if i % 3 == 0:
                self.find_contact()

    def write_results(self):
        singlevolume = (math.pi ** (self.dimension/2)) * (self.radius **
                                                          self.dimension) / gamma(self.dimension/2 + 1)
        results = str(singlevolume * self.numspheres /
                      (self.length ** self.dimension)) + '\n'
        for i in self.spheres:
            results += str(i.center_pos) + '\n'
        file = open("output.txt", "w+")
        file.write(results)
        file.close()


def main():
    # dim = 2
    # p = Packing(dim, 10**dim, 1)
    p = Packing(2, 16, 7)
    p.initial_fill()
    p.relax()
    p.draw()


if __name__ == '__main__':
    main()
