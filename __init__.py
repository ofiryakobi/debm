__version__="0.2.0"
from distutils.version import StrictVersion
import threading
import urllib.request, json 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import minimize
from .Prospect import Prospect
from .Model import Model
from .Models import *
from .resPlot import resPlot
from .makeGrid import makeGrid
from .FitMultiGame import FitMultiGame
from .saveEstimation import saveEstimation


def versions():
    try:
        print('**When using DEBM in a paper, please cite: Yakobi, O., & Roth, Y. (2022). Decision from Experience Behavior Modeling (DEBM): an open-source Python package for developing, evaluating, and visualizing behavioral models. https://doi.org/10.31234/osf.io/3emdw')
        print('DEBM: Checking for updates')
        url=urllib.request.urlopen("https://pypi.org/pypi/DEBM/json")
        data = json.loads(url.read().decode())
        newest_version=StrictVersion(data['info']['version'])
        version=StrictVersion(__version__)
        if newest_version>version:
            print('There is a newer version of DEBM available!')
            print('It is important to keep DEBM up-to-date')
            print('Go to the command line and type: pip install debm -U')
            print('****************************************************')
        else:
            print('You have the latest DEBM version, no need to update.')
    except:
        print("Could not check for updates, please run 'pip install debm -U' from time to time to make sure you are up-to-date.")

x = threading.Thread(target=versions)
x.start()