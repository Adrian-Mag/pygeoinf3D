import numpy as np
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objects as go

from pygeoinf import EuclideanSpace, LinearOperator, MatrixLinearOperator
from pygeoinf.linear_solvers import CholeskySolver
from pygeoinf3d import BoxDomain, Lebesgue3D, Function3D
from pygeoinf3d.providers.tensor_product import BoxLaplacianSpectrumProvider
from pygeoinf3d.operators.bessel_sobolev_box import BesselSobolevBox
from pygeoinf3d.operators.sola import SOLAOperator3D
from pygeoinf3d.spaces.sobolev import Sobolev3D
from pygeoinf3d.spaces.forms import PathIntegralFunctional, VolumeKernel3D
from pygeoinf3d.core.paths import Path3D
from pygeoinf3d.providers.function_providers import GaussianProvider3D
from pygeoinf3d.plot import (
    plot_function3d_axis_slider,
    compare_function3d_axis_sliders,
    compare_panels_axis_dropdown_sliders,
)

np.random.seed(42)

box = BoxDomain(np.zeros(3), np.ones(3))
base_space = Lebesgue3D(box)

spec_provider = BoxLaplacianSpectrumProvider(box, truncation=(8, 8, 8), bc_type="neumann")
W     = BesselSobolevBox(base_space, spec_provider, shift=1.0, order=2.0)
W_inv = BesselSobolevBox(base_space, spec_provider, shift=1.0, order=-2.0)
M = Sobolev3D(base_space, W, W_inv)
n_modes = 8 ** 3

N_d = 300
N_stations = 10
np.random.seed(42)

# Stations: random (x,y) in [0.05,0.95], fixed z=1 (surface)
station_xy = np.random.uniform(0.05, 0.95, size=(N_stations, 2))
stations = np.column_stack([station_xy, np.ones(N_stations)])

# Sources: random in [0.05,0.95]^3
sources = np.random.uniform(0.05, 0.95, size=(N_d, 3))

# Assign each source to a (uniformly) random station
station_idx = np.random.randint(0, N_stations, size=N_d)

# Straight-ray Path3D: source → assigned station
paths = [
    Path3D.from_polyline(np.array([sources[i], stations[station_idx[i]]]))
    for i in range(N_d)
]

# --- 3×3×3 target grid (for geometry preview) ---
_tgx = np.linspace(0.1, 0.9, 3)
target_grid = np.array([[x, y, z] for x in _tgx for y in _tgx for z in _tgx])
N_p = len(target_grid)

# Forward forms: PathIntegralFunctional on the Sobolev space
forms_G = [PathIntegralFunctional(M, p, n_quad=200) for p in paths]
print(f"Forward forms G: {len(forms_G)} PathIntegralFunctional on Sobolev3D")

D = EuclideanSpace(N_d)
P = EuclideanSpace(N_p)

G = SOLAOperator3D(M, D, forms_G)
# True model: three Gaussians
def _true_model_callable(x):
    x = np.asarray(x, dtype=float)
    single = x.ndim == 1
    pts = x[None, :] if single else x
    c1 = np.array([0.3, 0.5, 0.5])
    c2 = np.array([0.7, 0.4, 0.5])
    c3 = np.array([0.5, 0.6, 0.3])
    val = (
        1.5 * np.exp(-np.sum((pts - c1) ** 2, axis=1) / (2 * 0.12 ** 2))
        + 1.0 * np.exp(-np.sum((pts - c2) ** 2, axis=1) / (2 * 0.10 ** 2))
        - 0.8 * np.exp(-np.sum((pts - c3) ** 2, axis=1) / (2 * 0.08 ** 2))
    )
    return float(val[0]) if single else val

m_true = Function3D(box, evaluate_callable=_true_model_callable, name="true model")

# Synthetic data with 5% noise
d_bar = G(m_true)
noise_std = 0.05 * np.max(np.abs(d_bar))
C_D_mat = noise_std ** 2 * np.eye(N_d)
C_D = MatrixLinearOperator(D, D, C_D_mat)
rng = np.random.default_rng(123)
d_tilde = d_bar + rng.normal(0, noise_std, size=N_d)

# Target forms: 3x3x3 Gaussian volume kernels on base_space (L2)
fp_T = GaussianProvider3D(box, centers=target_grid, sigmas=0.10, normalized=True)
forms_T = [VolumeKernel3D(M, fp_T.get_function(k)) for k in range(N_p)]
print(f"Target forms T : {N_p} VolumeKernel3D on {_tgx.shape[0]}x{_tgx.shape[0]}x{_tgx.shape[0]} grid")

T = SOLAOperator3D(M, P, forms_T)

solver = CholeskySolver(galerkin=True, parallel=True, n_jobs=-1)
X = T @ G.adjoint @ solver(G @ G.adjoint + C_D)

