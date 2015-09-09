# -*- coding: utf-8 -*-

import numpy as np
from scitools.pyreport.options import verbose_execute

from k_modes import KModes

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import metrics

from util import Mount_CSV, Open_File


