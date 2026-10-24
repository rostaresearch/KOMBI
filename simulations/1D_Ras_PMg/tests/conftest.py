import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# the driver's modules (config, util, MSM, dham, explore_sim_PMg) live one level up
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
