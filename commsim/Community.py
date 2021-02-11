#!/usr/bin/env python

"""
Community class object for representing multiple species ranges.
"""

import sys

sys.path.append("..")
import numpy as np
from commsim.Species import Species
from commsim.Environment import Environment


class Community:
    """
    Generates a community assemblage matrix from a commsim.Environment
    object and an list of commsim.Species objects.
    """

    def __init__(self, env, species_list):

        # get dimensions from the environment matrix
        self.env = env
        self.x = env.arr.shape[1]
        self.y = env.arr.shape[2]

        # make new array for storing all species ranges
        self.arr = np.zeros((len(species_list), self.x, self.y), dtype=np.int8)

    def __repr__(self):
        return f"<Community nspecies={self.arr.shape[0]}; nenvs={self.env.arr.shape[0]}>"

    def spp_richness(self):
        "returns the sum of species at all sites"
        pass

    def spp_range(self, idx):
        "returns the occurrence of a selected species"
        pass


if __name__ == "__main__":

    # generate several species
    SPECIES_LIST = []
    for idx in range(10):
        
        # randomly select a center coordinate and decay rate
        xcoord = np.random.uniform(0, 100)
        ycoord = np.random.uniform(0, 100)
        decay = np.random.uniform(0, 1)

        # init Species with random values
        spp = Species(
            origin=(xcoord, ycoord), 
            decay=decay, 
            bioclim={
                "mean-temp": (5, 5),
                "min-temp": (1, 5),
                "mean-precip": (0.1, 0.01),
            }
        )
        SPECIES_LIST.append(spp)

    # generate random environment
    ENV = Environment(
        shape=(100, 100), 
        bioclims={},
        elevation={},
    )

    # build community from species and environment
    comm = Community(ENV, SPECIES_LIST)
    print(comm)
