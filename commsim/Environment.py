#!/usr/bin/env python

"""
Generate ...
"""

from typing import Tuple, Union, Dict
import numpy as np
import pandas as pd
import scipy.stats as stats
import altair as alt
from loguru import logger


# allows large plots to be embedded in notebooks
alt.data_transformers.enable('json')


class Layer:
    """
    Base class for Environmental Layers. Allows adding layers together
    and drawing layers.
    """
    def __init__(self, shape:Tuple[int, int]=(10,10)):
        self.shape = shape
        self.arr = np.zeros(self.shape, dtype=float)
        self.data = pd.DataFrame()

    def __add__(self, layer):
        """
        Returns a new layer as the mean Z value of two layers. This
        enables `newlayer = layer1 + layer2`.
        """
        # check that layers are same shape
        assert self.data.shape == layer.data.shape, "layer shapes are different"

        # make new Layer class instance
        joined = Layer(layer.shape)
        joined.data = self.data.copy()
        joined.data.z = (self.data.z + layer.data.z) / 2.
        return joined

    def __repr__(self):
        return "<Layer>"

    def draw(self):
        """
        Returns an altair heatmap drawing of self.data
        """
        base = alt.Chart(self.data).mark_rect().encode(
            x=alt.X("x:O", axis=None),#alt.Axis(labels=False)),
            y=alt.Y('y:O', axis=None),#alt.Axis(labels=False)),
            color='z:Q',
            tooltip=[
                alt.Tooltip('z:Q', title='z')
            ],
        ).properties(title=repr(self), width=250, height=250)
        return alt.concat(base)



class Linear(Layer):
    """
    Generate a layer with an gradient applied linearly across its shape.
    """
    def __init__(
        self, 
        shape: Tuple[int, int]=(10, 10), 
        minval:float=0, 
        maxval:float=1, 
        angle:int=0,
    ):
        # init Layer base class
        super().__init__(shape=shape)

        # store user args
        self.angle = angle
        self.minval = minval
        self.maxval = maxval

        # generate linear data
        self.generate()

    def __repr__(self):
        return "<Linear Layer>"        

    def generate(self):
        """
        Generate a linear gradient across the grid at specified angle
        in degrees.
        """
        # get vectors pointed in an angled direction
        xpos = np.cos(np.deg2rad(self.angle))
        ypos = np.sin(np.deg2rad(self.angle))
        xvec = np.linspace(0, xpos, self.shape[0])
        yvec = np.linspace(0, ypos, self.shape[1])

        # get a meshgrid from vectors and set height to mean
        xspan, yspan = np.meshgrid(xvec, yvec)
        height = (xspan + yspan) / 2.

        # normalize heights to minval maxval
        height += abs(height.min())
        multiplier = (self.maxval - self.minval) / height.max()
        height *= multiplier
        height += self.minval

        # orient the grid at the specified angle
        if xpos < 0:
            xspan = xspan.ravel()[::-1]
        else:
            xspan = xspan.ravel()
        if ypos > 0:
            yspan = yspan.ravel()[::-1]
        else:
            yspan = yspan.ravel()

        # store the heights
        self.arr = height
        with np.errstate(divide='ignore', invalid='ignore'):
            self.data = pd.DataFrame({
                'x': xspan / xspan.max(),
                'y': yspan / yspan.max(),
                'z': height.ravel(),
            })



class Gaussian(Layer):
    """
    Generate a layer with random gaussian 

    Parameters
    -----------
    shape: Tuple[int, int]
        The shape of the layer as a width, height tuple.
    peaks: int
        The number of gaussian peaks to add. Overlapping gaussians will
        additively contribute to the value.
    minval: float
        Scale layer to have this minimum value.
    maxval: float
        Scale layer to have this maximum value.
    decay: float
        The std of gaussian peaks in the same units as the layer shape.
    seed: int
        Seed for drawing random locations for gaussian centers.
    """
    def __init__(
        self,
        shape:Tuple[int, int]=(10, 10), 
        peaks:int=2,
        minval:float=0,
        maxval:float=1,
        decay:float=2,
        seed:Union[int,None]=None,
        ):

        super().__init__(shape=shape)
        self.peaks = peaks
        self.minval = minval
        self.maxval = maxval
        self.decay = decay
        if seed:
            np.random.seed(seed)
        self.generate()

    def __repr__(self):
        return "<Gaussian Layer>"

    def generate(self):
        """
        help
        """
        origins = [(
            np.random.randint(0, self.shape[0]), 
            np.random.randint(0, self.shape[1]),
            ) for point in range(self.peaks)
        ]

        elev = np.zeros((self.peaks, *self.shape), dtype=np.float)
        for eidx in range(elev.shape[0]):
            for xpos in range(elev.shape[1]):
                for ypos in range(elev.shape[2]):
                    elev[eidx, xpos, ypos] = euclidean_dist(
                        origins[eidx][0], 
                        origins[eidx][1], 
                        xpos, ypos)

            # exponential decay
            # elev[eidx] = 1 - (1. * np.exp(self.decay * elev[eidx]))

            # normal (gaussian) decay
            elev[eidx] = stats.norm.pdf(elev[eidx], loc=0, scale=self.decay)
    
        # normalize elev layer to 0-1
        elev = elev.sum(axis=0)
        elev = elev - elev.min()
        elev = elev / elev.max()

        # normalize to min-max
        multiplier = (self.maxval - self.minval) / elev.max()
        elev *= multiplier
        elev += self.minval
        elev = elev.max() - elev
    
        # store data
        self.arr = elev
        xspan, yspan = np.meshgrid(
            np.linspace(0, 1, self.shape[0]), 
            np.linspace(0, 1, self.shape[1]),            
        )
        self.data = pd.DataFrame({
            'x': xspan.ravel(),
            'y': yspan.ravel()[::-1],
            'z': elev.ravel(),
        })



class Environment:
    """
    Generates and stores a 3-d matrix of environment Layers.
    Environmental variables (bioclims) are added as simple gradients 
    with min,max values, and they are then transformed by elevation 
    which can apply a different coefficient to each variable 
    (how strongly it correlates with elevation).

    The environmental matrix is created during init and then there
    are draw functions available to visualize the array layers.
    
    Parameters:
    -----------
    ...
    """
    def __init__(
        self, 
        shape:Tuple[int,int]=(10,10),
        bioclims:Union[Dict[str,Layer], None]=None,
        elevation:Union[Dict[str,float], None]=None,
        ):
              
        self.shape = (shape[0], shape[1], 1)
        self.bioclims = bioclims
        self.elevation = elevation
        # self.arr = np.zeros(arrshape, dtype=float)       
        # elevation multipliers
        # self.elevation = elevation
        # self.elev = np.zeros((1, shape[0], shape[1]), dtype=float)        

        # fill base gradient layers first
        self._apply_base_gradient_layers()

        # then transform base layers by elevation
        self._apply_elevation_transform()


    def __repr__(self):
        return f"<Environment shape=({self.shape})>"


    def _apply_base_gradient_layers(self):
        pass

    def _apply_elevation_transform(self):
        pass  
    
    def draw(self):
        """
        Returns a toyplot (canvas, axes) tuple of grid env.
        """
        # chart = alt.Chart(source=)



def euclidean_dist(xorig, yorig, xnew, ynew):
    "returns the euclidean_dist between two coordinate points"
    return np.sqrt((((xorig - xnew) ** 2) + ((yorig - ynew) ** 2)))




if __name__ == "__main__":

    pass