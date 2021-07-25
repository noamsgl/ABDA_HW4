# import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.patches import Ellipse
import matplotlib.transforms
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
from torch.distributions import constraints
from tqdm import tqdm

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta, AutoNormal
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, MCMC, NUTS, predictive
from pyro.infer import Predictive

pyro.enable_validation(True)
pyro.set_rng_seed(0)

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


method = PCA(n_components=2, whiten=True)  # project to 2 dimensions
projected = method.fit_transform(np.array(inputs[tags['datum']].tolist()))

figure = pyplot.figure()
axis = figure.add_subplot(111)
# Display data
for label in labels:
  color = np.expand_dims(np.array(settings.get_color(label)), axis=0)
  pyplot.scatter(projected[labels == label, 0], projected[labels == label, 1],
                           c=color, alpha=0.5, label=label, edgecolor='none')

# Centroids
for label in labels:
# Centroids
    color = np.array(settings.get_color(label))
    # Ellipsis
    Views.confidence_ellipse(projected[labels == label, 0], projected[labels == label, 1], axis,
                            edgecolor=color, linewidth=3, zorder=0)


The confidence_ellipse came from matplotlib example.