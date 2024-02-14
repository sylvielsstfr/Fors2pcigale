#! /usr/bin/env python
# pylint: disable-all

""" Convinience Function to Concatenate Results """

# Packages
import argparse

import gelato.Concatenate as C
import gelato.ConstructParams as CP

# GELATO
import gelato.Utility as U
import numpy as np
from astropy.table import Table

# Main Function
if __name__ == "__main__":

    # Parse Arguement
    args = U.parseArgs()

    # Parameters
    p = CP.construct(args.Parameters)

    # Assemble Objects
    if args.single: # Single Mode
        objects = Table([[args.Spectrum],[args.Redshift]],names=('Path','z'))
    else: # Multi Mode
        objects = U.loadObjects(args.ObjectList)

    ##Concatenate Results
    C.concatfromresults(p,objects)