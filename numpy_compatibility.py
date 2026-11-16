"""
Compatibility layer for numpy version differences.
This handles the case where data was pickled with numpy 2.x but being loaded with numpy 1.x
"""

import numpy as np
import sys

# Create a fake numpy._core module for compatibility with numpy 1.x
if not hasattr(np, '_core'):
    import types
    
    # Create fake _core module
    np_core = types.ModuleType('numpy._core')
    
    # Create fake numeric module inside _core
    np_core_numeric = types.ModuleType('numpy._core.numeric')
    
    # Copy essential functions from numpy to the fake module
    np_core_numeric.asarray = np.asarray
    np_core_numeric.array = np.array
    np_core_numeric.zeros = np.zeros
    np_core_numeric.ones = np.ones
    np_core_numeric.empty = np.empty
    np_core_numeric.full = np.full
    np_core_numeric.arange = np.arange
    np_core_numeric.linspace = np.linspace
    np_core_numeric.logspace = np.logspace
    np_core_numeric.geomspace = np.geomspace
    
    # Add the missing _frombuffer function
    np_core_numeric._frombuffer = np.frombuffer
    
    # Add other common functions that might be needed
    np_core_numeric.frombuffer = np.frombuffer
    np_core_numeric.fromfile = np.fromfile
    np_core_numeric.fromstring = np.fromstring
    np_core_numeric.fromiter = np.fromiter
    
    # Add the numeric module to _core
    sys.modules['numpy._core.numeric'] = np_core_numeric
    np_core.numeric = np_core_numeric
    
    # Add the _core module to numpy
    sys.modules['numpy._core'] = np_core
    np._core = np_core
    
    print("Created numpy._core.numeric compatibility layer")