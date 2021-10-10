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
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import copy
import math
from scipy.special import gamma


def get_truncated_normal(mean, sd, low, upp):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class Sphere:

    radius = 1

    def __init__(self, dimension, center_pos):
        self.dimension = dimension
        self.center_pos = center_pos


class Packing:

    c = .5  # force constant
    k = 1  # displacement constant?
    change = 1.0001  # isotropically resize constant

    def __init__(self, dimension, numspheres, l, spheres=None, relaxed=None, contact=None):
        self.dimension = dimension
        self.numspheres = numspheres
        self.l = l
        self.spheres = spheres if spheres is not None else []
        self.relaxed = relaxed if relaxed is not None else False
        self.contact = contact if contact is not None else []

    # only usable in two dimensions
    def draw(self):
        plt.axes()
        boundary = plt.Rectangle((-self.l/2, -self.l/2), self.l, self.l, fc='None', ec='black')
        plt.gca().add_patch(boundary)
        plt.axis('scaled')
        for i in range(len(self.spheres)):
            circle = plt.Circle((self.spheres[i].center_pos[0], self.spheres[i].center_pos[1]),
                                Sphere.radius, fc='None', ec='black')
            plt.gca().add_patch(circle)
            point = plt.plot(self.spheres[i].center_pos[0], self.spheres[i].center_pos[1], 'bo')
        plt.show()

    def initial_fill(self):
        for i in range(self.numspheres):
            x = get_truncated_normal(0, self.l/4, -self.l/2, self.l/2)
            center = x.rvs(size=self.dimension)
            self.spheres.append(Sphere(self.dimension, np.array(center)))

    def find_contact(self):
        self.relaxed = True
        self.contact = np.zeros((self.numspheres, self.numspheres))
        for i in range(self.numspheres):
            for j in range(i+1, self.numspheres):
                diff = np.subtract(self.spheres[i].center_pos, self.spheres[j].center_pos)
                if np.linalg.norm(diff) < 2*Sphere.radius:
                    self.contact[i][j] = 1
                    self.relaxed = False
                # kind of an approximate method, maybe go back and look at it with math later
                elif np.linalg.norm(diff) > self.l - 2*Sphere.radius:
                    newcenter1 = copy.deepcopy(self.spheres[i].center_pos)
                    newcenter2 = copy.deepcopy(self.spheres[j].center_pos)
                    for k in range(self.dimension):
                        diff = np.subtract(newcenter1, newcenter2)
                        if abs(diff[k]) > self.l - 2*Sphere.radius:
                            if self.spheres[i].center_pos[k] > self.l/2 - Sphere.radius:
                                newcenter1[k] = newcenter1[k] - self.l
                            elif self.spheres[i].center_pos[k] < -self.l/2 + Sphere.radius:
                                newcenter1[k] = newcenter1[k] + self.l
                            elif self.spheres[j].center_pos[k] > self.l/2 - Sphere.radius:
                                newcenter2[k] = newcenter2[k] - self.l
                            elif self.spheres[j].center_pos[k] < -self.l/2 + Sphere.radius:
                                newcenter2[k] = newcenter2[k] + self.l
                    diff = np.subtract(newcenter1, newcenter2)
                    if np.linalg.norm(diff) < 2*Sphere.radius:
                        self.contact[i][j] = 2
                        self.relaxed = False

    def relax_once(self):
        # isotropically rescale
        self.l = Packing.change * self.l
        for i in range(self.numspheres):
            self.spheres[i].center_pos = np.multiply(self.spheres[i].center_pos, Packing.change)

        # displace with forces
        sum = np.array(np.zeros((self.numspheres, self.dimension)))
        for i in range(self.numspheres):
            for j in range(i+1, self.numspheres):
                if self.contact[i][j] == 1:  # represents direct contact between spheres
                    jtoi = np.subtract(self.spheres[i].center_pos, self.spheres[j].center_pos)
                    norm = np.linalg.norm(jtoi)
                    force = max(
                        0, (2 * Sphere.radius - norm) * Packing.c)
                    sum[i] += force * np.array(jtoi) / norm
                    sum[j] += -1 * force * np.array(jtoi) / norm
                elif self.contact[i][j] == 2:  # represents wrapped contact between spheres
                    newcenter1 = copy.deepcopy(self.spheres[i].center_pos)
                    newcenter2 = copy.deepcopy(self.spheres[j].center_pos)
                    for k in range(self.dimension):
                        diff = np.subtract(newcenter1, newcenter2)
                        if abs(diff[k]) > self.l - 2*Sphere.radius:
                            if self.spheres[i].center_pos[k] > self.l/2 - Sphere.radius:
                                newcenter1[k] = newcenter1[k] - self.l
                            elif self.spheres[i].center_pos[k] < -self.l/2 + Sphere.radius:
                                newcenter1[k] = newcenter1[k] + self.l
                            elif self.spheres[j].center_pos[k] > self.l/2 - Sphere.radius:
                                newcenter2[k] = newcenter2[k] - self.l
                            elif self.spheres[j].center_pos[k] < -self.l/2 + Sphere.radius:
                                newcenter2[k] = newcenter2[k] + self.l

                    twotoone = np.subtract(newcenter1, newcenter2)
                    force = max(
                        0, (2 * Sphere.radius - np.linalg.norm(twotoone)) * Packing.c)
                    sum[i] += force * np.array(twotoone) / np.linalg.norm(twotoone)
                    sum[j] += -1 * force * np.array(twotoone) / np.linalg.norm(twotoone)

        # apply the displacement and wrap locations
        for i in range(self.numspheres):
            for j in range(self.dimension):
                newcenter = self.spheres[i].center_pos[j] + sum[i][j]
                while newcenter > self.l/2:
                    newcenter -= self.l
                while newcenter < -self.l/2:
                    newcenter += self.l
                self.spheres[i].center_pos[j] = newcenter

    def relax(self):  # improve the number of steps between checking neighbors
        i = 0
        self.find_contact()
        while not self.relaxed:
            self.relax_once()
            i += 1
            if i % 3 == 0:
                self.find_contact()
        self.write_results()

    def write_results(self):
        singlevolume = (math.pi ** (self.dimension/2)) * (Sphere.radius **
                                                          self.dimension) / gamma(self.dimension/2 + 1)
        results = str(singlevolume * self.numspheres / (self.l ** self.dimension)) + '\n'
        for i in self.spheres:
            results += str(i.center_pos) + '\n'
        file = open("output.txt", "w+")
        file.write(results)
        file.close()


def main():
    p = Packing(2, 20, 7)
    p.initial_fill()
    p.relax()


if __name__ == '__main__':
    main()
