#!/usr/bin/env python

"""
Species Class for storing: idx, origin, decay (disperal, bioclim (dict)
"""


class Species:
    """
    A species origin is the x,y coordinate defining the center of
    its geographic range. The environment at this location will 
    determine its niche preferences, which combined with a dispersal
    decay rate, and other species occurrences, will determine its 
    occurrence in sites around origin.
    
    Parameters
    ----------
    origin (tuple or list):
        A set of x, y coordinates for the center of origin of a spp. range.
    decay (float):
        The exponential decay rate parameter for dispersal from origin.
    bioclim (dict):
        A dict of key,val pairs with keys as bioclim variable names and 
        values as tuples of (mean, std) for species tolerances.
    """
    # global species index counter
    idx = 0
    
    def __init__(self, origin, decay, bioclim):
       
        # store this instances attrs
        self.idx = Species.idx
        self.origin = origin
        self.decay = decay
        self.bioclim = bioclim
        
        # increment the global counter when an instance is made
        Species.idx += 1        
    
    def __repr__(self):
        """
        custom repr shows species global idx, origin, and decay"
        """
        attribs = (
            f"<Species idx={self.idx}, "
            f"o={self.origin}, "
            f"e={self.decay}, "
            f"bioclim={self.bioclim}>"
        )
        return attribs




if __name__ == "__main__":

    # two example Species 
    A = Species((10, 10), 0.5, {"mean-temp": (5, 1)})
    B = Species((30, 30), 0.05, {"mean-temp": (3, 3)})
    print(A)
    print(B)
