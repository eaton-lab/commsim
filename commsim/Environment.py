#!/usr/bin/env python

"""

"""

import sys
import numpy as np
import toyplot
import toyplot.browser


class Environment:
    """
    A matrix of environment layers with float values representing 
    spatial variation. Environmental variables (bioclims) are added
    as simple gradients with min,max values, and they are then 
    transformed by elevation which can apply a different coefficient
    to each variable (how strongly it correlates with elevation).

    The environmental matrix is created during init and then there
    are draw functions available to visualize the array layers.
    
    Parameters:
    -----------
    ...
    """
    def __init__(self, shape, bioclims, elevation):
              
        # env layers; starts with 1, additional can be added.
        nbioclim = max(1, len(bioclims))
        self.arr = np.zeros((nbioclim, shape[0], shape[1]), dtype=float)
        
        # elevation multipliers
        self.elevation = elevation
        self.elev = np.zeros((1, shape[0], shape[1]), dtype=float)        

        # fill base gradient layers first
        self._apply_base_gradient_layers()

        # then transform base layers by elevation
        self._apply_elevation_transform()

    def __repr__(self):
        return f"<Environment shape=({self.arr.shape})>"


    def _apply_base_gradient_layers(self):
        pass

    def _apply_elevation_transform(self):
        pass
        
    
    def add_linear_gradient(self):
        """
        Adds a linear gradient niche axis (e.g., temperature from S to N)
        """
        pass
    
    
    def add_gaussian_gradient(self, npeaks, seed):
        """
        Adds N gaussian gradients summed (e.g., mountain elevational variation)
        """
        pass
        
        
    def add_chevron_layers(self, nshapes):
        """
        Adds N chevrons to fit left to right on grid.
        """
        pass
    
    
    def get_summed_layers(self):
        """
        Return summed and normalized environmental layers.
        """
        pass
    
    
    def draw(self):
        """
        Returns a toyplot (canvas, axes) tuple of grid env.
        """

        # build drawing
        canvas, table = toyplot.matrix(self.arr[0])

        # show result in browser if not in a notebook
        if not sys.stdout.isatty():
            toyplot.browser.show(canvas)
            return (None, None)
        return canvas, table
            

if __name__ == "__main__":

    env = Environment(
        shape=(100, 100), 
        bioclims={
            "mean-temp": (0, 20, "linear-sn"),
            "min-temp": (-20, 10, "linear-we"),
            "mean-precip": (0.1, 1.0, "linear-we"),
        },
        elevation={
            "mean-temp": 0.1,
            "min-temp": 0.2,
            "mean-precip": 0.5,
        },
    )

    print(env)
    env.draw()