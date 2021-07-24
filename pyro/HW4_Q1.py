import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, MCMC, NUTS, predictive, infer_discrete
from pyro.infer import Predictive

pyro.enable_validation(True)
pyro.set_rng_seed(0)
K = 3
D = 4


@config_enumerate(default='parallel')
@poutine.broadcast
def model(data):
    # Global variables.
    weights = pyro.param('weights', torch.ones(K) / K, constraint=constraints.simplex)
    with pyro.plate('components', size=K):
        scales = pyro.param('scales', (torch.rand(D, D) * torch.eye(D)).expand([K, D, D]), constraint=constraints.positive)
        locs = pyro.param('locs', torch.rand(D).expand([K, D]))
        print(f"locs.shape={locs.shape}")
        print(f"scales.shape={scales.shape}")
    with pyro.plate('data', data.size(0)):
        # Local variables.
        assignment = pyro.sample('assignment', dist.Categorical(torch.outer(torch.ones(len(data)), weights)))
        print(f"locs = {locs}, scales={scales}, assignment={assignment.shape}")
        pyro.sample('obs', dist.MultivariateNormal(locs[assignment], scales[assignment]), obs=data)
    return weights, scales, locs


@config_enumerate(default="parallel")
@poutine.broadcast
def full_guide(data):
    with pyro.plate('data', data.size(0)):
        # Local variables.
        assignment_probs = pyro.param('assignment_probs', torch.outer(torch.ones(len(data)), torch.ones(K) / K),
                                      constraint=constraints.simplex)
        pyro.sample('assignment', dist.Categorical(assignment_probs), infer={"enumerate": "sequential"})


def initialize(data):
    pyro.clear_param_store()

    optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
    elbo = TraceEnum_ELBO(max_plate_nesting=1)
    svi = SVI(model, full_guide, optim, loss=elbo)

    # Initialize weights to uniform.
    pyro.param('auto_weights', torch.ones(K) / K, constraint=constraints.simplex)

    # Assume half of the data variance is due to intra-component noise.
    var = (data.var() / 2).sqrt()
    pyro.param('auto_scales', torch.tensor([var] * 6), constraint=constraints.positive)

    # Initialize means from a subsample of data.
    pyro.param('auto_locs', data[torch.multinomial(torch.ones(len(data)) / len(data), K)])

    loss = svi.loss(model, full_guide, data)

    return loss, svi


def plot_data_and_gaussians(X, assignments, mus, sigmas):
    # Create figure
    fig = plt.figure()

    # Plot data
    pca = PCA(n_components=2)
    PCs = pca.fit_transform(X)
    plt.scatter(PCs[:, 0], PCs[:, 1], 24, c=assignments)

    # Plot cluster centers
    mus_2d = pca.transform(mus.data)
    x = [float(m[0]) for m in mus_2d]
    y = [float(m[1]) for m in mus_2d]
    plt.scatter(x, y, 99, c='red')

    # Plot ellipses for each cluster
    for sig_ix in range(len(sigmas)):
        ax = fig.gca()
        cov = np.array(sigmas[sig_ix])
        lam, v = np.linalg.eig(cov)
        lam = np.sqrt(lam)
        ell = Ellipse(xy=(x[sig_ix], y[sig_ix]),
                      width=lam[0] * 4, height=lam[1] * 4,
                      angle=np.rad2deg(np.arccos(v[0, 0])),
                      color='blue')
        ell.set_facecolor('none')
        ax.add_artist(ell)
    fig.suptitle("PCA 2d Visualization of Data and Gaussians")
    return fig


def get_X():
    url = 'https://bgu-abda.bitbucket.io/homework/04iris.csv'
    df = pd.read_csv(url)
    df = df.rename({'sepal length (cm)': 'sepal_length', 'sepal width (cm)': 'sepal_width',
                    'petal length (cm)': 'petal_length', 'petal width (cm)': 'petal_width'}, axis='columns')
    df["label"] = df['species'].astype('category').cat.codes
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = torch.tensor(df[features].values).float()
    return X

X = get_X()

trace = poutine.trace(model).get_trace(X)


global_guide = AutoDelta(poutine.block(model, expose=['weights', 'locs', 'scales']))
# global_guide = config_enumerate(global_guide, 'parallel')
_, svi = initialize(X)
for i in tqdm(range(1500)):
    svi.step(X)
#
# locs = pyro.param('locs')
# scales = pyro.param('scales')
# weights = pyro.param('weights')
# assignment_probs = pyro.param('assignment_probs')
#
# assignments = np.uint8(torch.argmax(assignment_probs, dim=1))
#
# plot_data_and_gaussians(X, df.label, locs.data, scales.data);