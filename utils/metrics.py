import numpy as np
from scipy.stats import entropy, wasserstein_distance
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist


def wasserstein_dist(real, fake):
    """Mean Wasserstein distance across all composition dimensions."""
    dists = [wasserstein_distance(real[:, i], fake[:, i])
             for i in range(real.shape[1])]
    return float(np.mean(dists)), float(np.std(dists))


def kl_divergence(real, fake, n_bins=50):
    """Mean KL divergence across all dimensions."""
    kls = []
    for i in range(real.shape[1]):
        lo = min(real[:, i].min(), fake[:, i].min())
        hi = max(real[:, i].max(), fake[:, i].max()) + 1e-8
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(real[:, i], bins=bins, density=True)
        q, _ = np.histogram(fake[:, i], bins=bins, density=True)
        p = p + 1e-10; p /= p.sum()
        q = q + 1e-10; q /= q.sum()
        kls.append(float(entropy(p, q)))
    return float(np.mean(kls))


def js_divergence(real, fake, n_bins=50):
    """Mean Jensen-Shannon divergence across all dimensions."""
    jss = []
    for i in range(real.shape[1]):
        lo = min(real[:, i].min(), fake[:, i].min())
        hi = max(real[:, i].max(), fake[:, i].max()) + 1e-8
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(real[:, i], bins=bins, density=True)
        q, _ = np.histogram(fake[:, i], bins=bins, density=True)
        p = p + 1e-10; p /= p.sum()
        q = q + 1e-10; q /= q.sum()
        m = (p + q) / 2
        jss.append(float((entropy(p, m) + entropy(q, m)) / 2))
    return float(np.mean(jss))


def mode_coverage(real, fake, n_clusters=10):
    """Fraction of real-data clusters represented in generated samples."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(real)
    real_modes = set(km.labels_)
    fake_modes = set(km.predict(fake))
    return len(fake_modes & real_modes) / len(real_modes)


def generation_diversity(samples):
    """Mean pairwise Euclidean distance among generated samples."""
    if len(samples) < 2:
        return 0.0
    return float(np.mean(pdist(samples, metric='euclidean')))


def evaluate_all(real, fake, n_clusters=10):
    """Compute all five distribution metrics at once.

    Args:
        real: (N, comp_dim) real composition array (denormalised)
        fake: (M, comp_dim) generated composition array (denormalised)

    Returns:
        dict with keys: wasserstein, wasserstein_std, kl, js,
                        mode_coverage, diversity
    """
    w, w_std = wasserstein_dist(real, fake)
    return {
        'wasserstein':     w,
        'wasserstein_std': w_std,
        'kl':              kl_divergence(real, fake),
        'js':              js_divergence(real, fake),
        'mode_coverage':   mode_coverage(real, fake, n_clusters),
        'diversity':       generation_diversity(fake),
    }
