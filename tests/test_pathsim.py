'''
Created on 30.05.2014

@author: psinger
'''

from pathtools.pathsim import PathSim
import numpy as np

for window_size in [2, 3]:
    print "==========="
    print window_size
    print "==========="

    sim = PathSim(window_size=window_size, sim_func="cosine", delimiter=" ")
    
<<<<<<< HEAD:tests/test_pathsim.py
    sim.fit("../data/test_case_1")
=======
    sim.fit("data/test_case_4")
>>>>>>> 937dda7df92974f40735aaa253c172267e17a7f0:test_pathsim.py

    print sim.sim("1","1")
    print sim.sim("1","2")


