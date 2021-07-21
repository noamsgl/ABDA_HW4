# http://pyro.ai/examples/gmm.html

import matplotlib.pyplot as plt
import numpy as np
from pyro.infer.discrete import infer_discrete
import torch
from matplotlib.patches import Ellipse
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, MCMC, NUTS, predictive
from pyro.infer import Predictive
from sklearn.manifold import TSNE, MDS


@config_enumerate(default='parallel')
@poutine.broadcast
def model(data):
    # Global variables.
    weights = pyro.param('weights', torch.ones(K) / K, constraint=constraints.simplex)
    scales = pyro.param('scales', torch.rand(K, 2, 2) * torch.eye(2).expand(K, 2, 2), constraint=constraints.positive)
    locs = pyro.param('locs', torch.rand((K, 2)))

    with pyro.iarange('data', data.size(0)):
        # Local variables.
        assignment = pyro.sample('assignment', dist.Categorical(torch.outer(torch.ones(len(data)), weights)))
        pyro.sample('obs', dist.MultivariateNormal(locs[assignment], scales[assignment]), obs=data)


@config_enumerate(default="parallel")
@poutine.broadcast
def full_guide(data):
    with pyro.iarange('data', data.size(0)):
        # Local variables.
        assignment_probs = pyro.param('assignment_probs', torch.outer(torch.ones(len(data)), torch.ones(K) / K),
                                      constraint=constraints.simplex)
        pyro.sample('assignment', dist.Categorical(assignment_probs), infer={"enumerate": "sequential"})


def initialize(data):
    pyro.clear_param_store()

    optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
    elbo = TraceEnum_ELBO(max_iarange_nesting=1)
    svi = SVI(model, full_guide, optim, loss=elbo)

    # Initialize weights to uniform.
    pyro.param('auto_weights', torch.ones(K) / K, constraint=constraints.simplex)

    # Assume half of the data variance is due to intra-component noise.
    var = (data.var() / 2).sqrt()
    pyro.param('auto_scale', torch.tensor([var]*6), constraint=constraints.positive)

    # Initialize means from a subsample of data.
    pyro.param('auto_locs', data[torch.multinomial(torch.ones(len(data)) / len(data), K)])

    loss = svi.loss(model, full_guide, data)

    return loss, svi


def get_samples():
    num_samples = 100  # per cluster

    # 2 clusters
    # note that both covariance matrices are diagonal
    mu1 = torch.tensor([0., 5.])
    sig1 = torch.tensor([[2., 0.], [0., 3.]])

    mu2 = torch.tensor([5., 0.])
    sig2 = torch.tensor([[4., 0.], [0., 1.]])
    
    mu3 = torch.tensor([8., 8.])
    sig3 = torch.tensor([[2., 0.], [0., 1.]])

    # generate samples
    dist1 = dist.MultivariateNormal(mu1, sig1)
    samples1 = [pyro.sample('samples1', dist1) for _ in range(num_samples)]

    dist2 = dist.MultivariateNormal(mu2, sig2)
    samples2 = [pyro.sample('samples2', dist2) for _ in range(num_samples)]

    dist3 = dist.MultivariateNormal(mu3, sig3)
    samples3 = [pyro.sample('samples3', dist3) for _ in range(num_samples)]

    data = torch.cat((torch.stack(samples1), torch.stack(samples2), torch.stack(samples3)))
    return data


def plot(data, mus=None, sigmas=None, colors='black', figname='fig.png'):
    # Create figure
    fig = plt.figure()

    # Plot data
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, 24, c=colors)

    # Plot cluster centers
    if mus is not None:
        x = [float(m[0]) for m in mus]
        y = [float(m[1]) for m in mus]
        plt.scatter(x, y, 99, c='red')

    # Plot ellipses for each cluster
    if sigmas is not None:
        for sig_ix in range(K):
            ax = fig.gca()
            cov = np.array(sigmas[sig_ix])
            lam, v = np.linalg.eig(cov)
            lam = np.sqrt(lam)
            ell = Ellipse(xy=(x[sig_ix], y[sig_ix]),
                          width=lam[0]*4, height=lam[1]*4,
                          angle=np.rad2deg(np.arccos(v[0, 0])),
                          color='blue')
            ell.set_facecolor('none')
            ax.add_artist(ell)

    # Save figure
    fig.savefig(figname)


if __name__ == "__main__":
    pyro.enable_validation(True)

    # Create our model with a fixed number of components
    K = 3
    pyro.set_rng_seed(42)

    data = get_samples()

    true_colors = [0] * 100 + [1] * 100 + [2] * 100
    # plot(data, colors=true_colors, figname='pyro_init.png')

    X_embedded = TSNE(n_components=2, init='pca', random_state=0).fit_transform(data)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=true_colors)
    plt.savefig('outputs/data-tsne.png')



    global_guide = AutoDelta(poutine.block(model, expose=['weights', 'locs', 'scales']))
    global_guide = config_enumerate(global_guide, 'parallel')
    _, svi = initialize(data)

    for i in range(1001):
        svi.step(data)

        if i % 50 == 0:
            locs = pyro.param('locs')
            scales = pyro.param('scales')
            weights = pyro.param('weights')
            assignment_probs = pyro.param('assignment_probs')

            print("locs: {}".format(locs))
            print("scales: {}".format(scales))
            print('weights = {}'.format(weights))
            print('assignments: {}'.format(assignment_probs))

            # todo plot data and estimates
            assignments = np.uint8(torch.argmax(assignment_probs, dim=1))
            plot(data, locs.data, scales.data, assignments, figname='outputs/pyro_iteration{}.png'.format(i))


    # nuts_kernel = NUTS(model, adapt_step_size=True)
    # mcmc = MCMC(nuts_kernel, warmup_steps=300, num_samples=3000)
    # mcmc.run(data)
    # print(mcmc.get_samples())

    # conditioned_model = poutine.condition(model, )
    predictive = Predictive(model, num_samples=1000)
    samples = predictive(data)
    x = 5

