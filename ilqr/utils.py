import sympy as sp
import numpy as np
from numba import njit
import itertools
from dataclasses import dataclass
from pathlib import Path
import random
from scipy.spatial.transform import Rotation

π = np.pi

repopath = Path(__file__).parent.parent.resolve()

@dataclass
class Point:
    """Point in 3D"""

    x: float
    y: float
    z: float = 0

    @property
    def ndim(self):
        return 2 if self.z == 0 else 3

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        return Point(self.x * other.x, self.y * other.y, self.z * other.z)

    def __repr__(self):
        return str((self.x, self.y, self.z))

    def hypot2(self):
        return self.x**2 + self.y**2 + self.z**2

def GetSyms(n_x, n_u):
  '''
      Returns matrices with symbolic variables for states and actions
      n_x: state size
      n_u: action size
  '''

  x = sp.IndexedBase('x')
  u = sp.IndexedBase('u')
  xs = sp.Matrix([x[i] for i in range(n_x)])
  us = sp.Matrix([u[i] for i in range(n_u)])
  return xs, us


def Constrain(cs, eps = 1e-4):
    '''
    Constraint via logarithmic barrier function
    Limitation: Doesn't work with infeasible initial guess.
    cs: list of constraints of form g(x, u) >= 0
    eps : parameters
    '''
    cost = 0
    for i in range(len(cs)):
        cost -= sp.log(cs[i] + eps)
    return 0.1*cost


def Bounded(vars, high, low, *params):
    '''
    Logarithmic barrier function to constrain variables.
    Limitation: Doesn't work with infeasible initial guess.
    '''
    cs = []
    for i in range(len(vars)):
        diff = (high[i] - low[i])/2
        cs.append((high[i] - vars[i])/diff)
        cs.append((vars[i] - low[i])/diff)
    return Constrain(cs, *params)


def SoftConstrain(cs, alpha = 0.01, beta = 10):
    '''
    Constraint via exponential barrier function
    cs: list of constraints of form g(x, u) >= 0
    alpha, beta : parameters
    '''
    cost = 0
    for i in range(len(cs)):
        cost += alpha*sp.exp(-beta*cs[i])
    return cost


def Smooth_abs(x, alpha = 0.25):
    '''
    smooth absolute value
    '''
    return sp.sqrt(x**2 + alpha**2) - alpha


@njit
def FiniteDiff(fun, x, u, i, eps):
  '''
     Finite difference approximation
  '''

  args = (x, u)
  fun0 = fun(x, u)

  m = x.size
  n = args[i].size

  Jac = np.zeros((m, n))
  for k in range(n):
    args[i][k] += eps
    Jac[:, k] = (fun(args[0], args[1]) - fun0)/eps
    args[i][k] -= eps

  return Jac



def sympy_to_numba(f, args, redu = True):
    '''
       Converts sympy matrix or expression to numba jitted function
    '''
    modules = [{'atan2':np.arctan2}, 'numpy']

    if isinstance(f, sp.Matrix):
        #To convert all elements to floats
        m, n = f.shape
        f += 1e-64*np.ones((m, n))

        #To eleminate extra dimension
        if (n == 1 or m == 1) and redu:
            if n == 1: f = f.T
            f = sp.Array(f)[0, :]
            f = njit(sp.lambdify(args, f, modules = modules))
            f_new = lambda *args: np.asarray(f(*args))
            return njit(f_new)

    f = sp.lambdify(args, f, modules = modules)
    return njit(f)

def split_agents(Z, z_dims):
    """Partition a cartesian product state or control for individual agents"""
    return np.split(np.atleast_2d(Z), np.cumsum(z_dims[:-1]), axis=1)

def split_agents_gen(z, z_dims):
    """Generator version of ``split_agents``"""
    dim = z_dims[0]
    for i in range(len(z_dims)):
        yield z[i * dim : (i + 1) * dim]

def compute_pairwise_distance(X, x_dims, n_d=2):
    """Compute the distance between each pair of agents"""
    assert len(set(x_dims)) == 1

    n_agents = len(x_dims)
    n_states = x_dims[0]

    if n_agents == 1:
        raise ValueError("Can't compute pairwise distance for one agent.")

    pair_inds = np.array(list(itertools.combinations(range(n_agents), 2)))
    X_agent = X.reshape(-1, n_agents, n_states).swapaxes(0, 2)
    dX = X_agent[:n_d, pair_inds[:, 0]] - X_agent[:n_d, pair_inds[:, 1]]
    return np.linalg.norm(dX, axis=0).T


def compute_pairwise_distance_nd(X, x_dims, n_dims, dec_ind=None):
    """Analog to the above whenever some agents only use distance in the x-y plane"""

    if X.ndim == 1:
        X = X.reshape(1, -1)

    n_states = x_dims[0]
    n_agents = len(x_dims)
    distances = np.zeros((X.shape[0], 0))

    pair_inds = list(itertools.combinations(range(n_agents), 2))
    if dec_ind is not None:
        pair_inds = list(
            filter(lambda pair, dec_ind=dec_ind: dec_ind in pair, pair_inds)
        )

    for (i, j) in pair_inds:
        n_dim = min(n_dims[i], n_dims[j])
        Xi = X[:, i * n_states : i * n_states + n_dim]
        Xj = X[:, j * n_states : j * n_states + n_dim]

        distances = np.c_[distances, np.linalg.norm(Xi - Xj, axis=1).reshape(-1, 1)]

    return distances

def split_agents_gen(z, z_dims):
    """Generator version of ``split_agents``"""
    dim = z_dims[0]
    for i in range(len(z_dims)):
        yield z[i * dim : (i + 1) * dim]

def uniform_block_diag(*arrs):
    """Block diagonal matrix construction for uniformly shaped arrays"""
    rdim, cdim = arrs[0].shape
    blocked = np.zeros((len(arrs) * rdim, len(arrs) * cdim))
    for i, arr in enumerate(arrs):
        blocked[rdim * i : rdim * (i + 1), cdim * i : cdim * (i + 1)] = arr

    return blocked

def face_goal(x0, xf):
    """Make the agents face the direction of their goal with a little noise"""

    VAR = 0.01
    dX = xf[:, :2] - x0[:, :2]
    headings = np.arctan2(*np.rot90(dX, 1))

    x0[:, -1] = headings + VAR * np.random.randn(x0.shape[0])
    xf[:, -1] = headings + VAR * np.random.randn(x0.shape[0])

    return x0, xf

def pos_mask(x_dims, n_d=2):
    """Return a mask that's true wherever there's a spatial position"""
    return np.array([i % x_dims[0] < n_d for i in range(sum(x_dims))])

def compute_energy(x, x_dims, n_d=2):
    """Determine the sum of distances from the origin"""
    return np.linalg.norm(x[pos_mask(x_dims, n_d)].reshape(-1, n_d), axis=1).sum()

def normalize_energy(x, x_dims, energy=10.0, n_d=2):
    """Zero-center the coordinates and then ensure the sum of
    squared distances == energy
    """

    # Don't mutate x's data for this function, keep it pure.
    x = x.copy()
    n_agents = len(x_dims)
    center = x[pos_mask(x_dims, n_d)].reshape(-1, n_d).mean(0)

    x[pos_mask(x_dims, n_d)] -= np.tile(center, n_agents).reshape(-1, 1)
    x[pos_mask(x_dims, n_d)] *= energy / compute_energy(x, x_dims, n_d)
    assert x.size == sum(x_dims)

    return x

def random_setup(
    n_agents, n_states, is_rotation=False, n_d=2, energy=None, do_face=False, **kwargs
):
    """Create a randomized set up of initial and final positions"""

    # We don't have to normlize for energy here
    x_i = randomize_locs(n_agents, n_d=n_d, **kwargs)

    # Rotate the initial points by some amount about the center.
    if is_rotation:
        θ = π + random.uniform(-π / 4, π / 4)
        R = Rotation.from_euler("z", θ).as_matrix()[:2, :2]
        x_f = x_i @ R - x_i.mean(axis=0)
    else:
        x_f = randomize_locs(n_agents, n_d=n_d, **kwargs)

    x0 = np.c_[x_i, np.zeros((n_agents, n_states - n_d))]
    xf = np.c_[x_f, np.zeros((n_agents, n_states - n_d))]

    if do_face:
        x0, xf = face_goal(x0, xf)

    x0 = x0.reshape(-1, 1)
    xf = xf.reshape(-1, 1)

    # Normalize to satisfy the desired energy of the problem.
    if energy:
        x0 = normalize_energy(x0, [n_states] * n_agents, energy, n_d)
        xf = normalize_energy(xf, [n_states] * n_agents, energy, n_d)

    return x0, xf

def randomize_locs(n_pts, random=False, rel_dist=3.0, var=3.0, n_d=2):
    """Uniformly randomize locations of points in N-D while enforcing
    a minimum separation between them.
    """

    # Distance to move away from center if we're too close.
    Δ = 0.1 * n_pts
    x = var * np.random.uniform(-1, 1, (n_pts, n_d))

    if random:
        return x

    # Determine the pair-wise indicies for an arbitrary number of agents.
    pair_inds = np.array(list(itertools.combinations(range(n_pts), 2)))
    move_inds = np.arange(n_pts)

    # Keep moving points away from center until we satisfy radius
    while move_inds.size:
        center = np.mean(x, axis=0)
        distances = compute_pairwise_distance(x.flatten(), [n_d] * n_pts).T

        move_inds = pair_inds[distances.flatten() <= rel_dist]
        x[move_inds] += Δ * (x[move_inds] - center)

    return x
