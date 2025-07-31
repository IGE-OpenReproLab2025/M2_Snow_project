# All the librairies used

import os
import glob
import csv
import calendar

# Data science
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.express as px
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
import matplotlib as mpl

# Cartographie
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Traitement d'images
from skimage.measure import find_contours

# Regridding (changement de grille pour données spatiales)
import xesmf as xe

# CDO - Climate Data Operators
from cdo import Cdo
cdo = Cdo()
