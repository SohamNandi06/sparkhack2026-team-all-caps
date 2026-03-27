import numpy as np


def add_dp_noise(weights, sigma=0.005):   

    noisy_weights = []

    for w in weights:
        noise = np.random.normal(loc=0.0, scale=sigma, size=w.shape)
        noisy_w = w + noise
        noisy_weights.append(noisy_w)

    print(f"DP noise added (sigma={sigma})")

    return noisy_weights