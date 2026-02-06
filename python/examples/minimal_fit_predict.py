import numpy as np

from pytree import PTree

rng = np.random.default_rng(0)

n_obs = 120
n_months = 12
n_stocks = 20
n_chars = 6
n_z = 3

X = rng.normal(size=(n_obs, n_chars))
Z = rng.normal(size=(n_obs, n_z))
H = rng.normal(size=(n_months, 1))

months = np.repeat(np.arange(n_months), n_obs // n_months)
stocks = rng.integers(0, n_stocks, size=n_obs)

R = rng.normal(size=n_obs)
Y = R + rng.normal(scale=0.1, size=n_obs)

model = PTree(min_leaf_size=5, max_depth=2, num_iter=2, num_cutpoints=2)
model.fit(R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months)

leaf_index = model.predict(X, months)
print("leaf_index", leaf_index[:5])
print("json length", len(model.to_json()))
