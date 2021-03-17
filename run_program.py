import os
import sys
import time
import numpy as np
import math
import random as rd

import HillClimbing as HC
import general_config as GC



param_list = ['tblock1','tblock2','tblock3'] #parse CLI

S0 = {
    'n1' : 256,
    'n2' : 256,
    'n3' : 256,
    'nb_threads' : 4,
    'nb_it' : 50,
    'tblock1' : 32,
    'tblock2' : 32,
    'tblock3' : 32,
}

#Define the used parameters
GC.define_usedParameters(param_list)

HCresult = HC.HillClimbing(S0, S0['nb_it'], "flops")

print(HCresult)

