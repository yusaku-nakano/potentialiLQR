from enum import Enum, auto
import abc
import numpy as np
from .utils import split_agents_gen, uniform_block_diag

class Model(Enum):
    DoubleInt4D = 0
    DoubleInt6D = auto()
    Car3D = auto()
    Unicycle4D = auto()
    Quadcopter6D = auto()
    Human6D = auto()
    HumanLin6D = auto()
    Quadcopter12D = auto()

def f_unicycle_4d(x, u):
    """
    Compute the state derivatives for a 4D unicycle model.
    x: State vector [px, py, v, theta]
    u: Control input [a, omega]
    Returns:
        x_dot: Derivative of state vector
    """
    px, py, v, theta = x
    a, omega = u

    x_dot = np.zeros_like(x)
    x_dot[0] = v * np.cos(theta)  # dx/dt
    x_dot[1] = v * np.sin(theta)  # dy/dt
    x_dot[2] = a                  # dv/dt
    x_dot[3] = omega              # dtheta/dt
    return x_dot

def linearize_unicycle_4d(x, u, dt):
    """
    Linearize the dynamics of a 4D unicycle model around the given state and input.
    x: State vector [px, py, v, theta]
    u: Control input [a, omega]
    dt: Time step
    Returns:
        A: State transition matrix
        B: Control input matrix
    """
    px, py, v, theta = x
    a, omega = u

    # Linearized state transition matrix A
    A = np.zeros((4, 4))
    A[0, 2] = np.cos(theta)       # Partial derivative of dx/dt w.r.t. v
    A[0, 3] = -v * np.sin(theta)  # Partial derivative of dx/dt w.r.t. theta
    A[1, 2] = np.sin(theta)       # Partial derivative of dy/dt w.r.t. v
    A[1, 3] = v * np.cos(theta)   # Partial derivative of dy/dt w.r.t. theta

    # Linearized control input matrix B
    B = np.zeros((4, 2))
    B[2, 0] = 1  # Partial derivative of dv/dt w.r.t. a
    B[3, 1] = 1  # Partial derivative of dtheta/dt w.r.t. omega

    # Apply Euler discretization
    A = np.eye(4) + dt * A
    B = dt * B

    return A, B

def rk4_integration(f, x0, u, h, dh=None):
    """Classic Runge-Kutta Method with sub-integration"""

    if not dh:
        dh = h

    t = 0.0
    x = x0.copy()

    while t < h - 1e-8:
        step = min(dh, h - t)

        k0 = f(x, u)
        k1 = f(x + 0.5 * k0 * step, u)
        k2 = f(x + 0.5 * k1 * step, u)
        k3 = f(x + k2 * step, u)

        x += step * (k0 + 2.0 * k1 + 2.0 * k2 + k3) / 6.0
        t += step

    return x
    
def rk4(f, dt, x, u):
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def integrate(x, u, dt, model):
    return rk4(lambda x, u: f(x, u, model), dt, x, u)

# Linearization
def linearize(x, u, dt, model):

    nx = x.shape[0]
    nu = u.shape[0]
    A = np.empty((nx, nx))
    B = np.empty((nx, nu))

    if model == Model.Unicycle4D:
        A, B = linearize_unicycle_4d(x, u, dt)
    else:
        raise ValueError("Unsupported model.")

    return A, B

# Dynamics Function Dispatcher
def f(x, u, model):

    if model == Model.Unicycle4D:
        return f_unicycle_4d(x, u)
    else:
        raise ValueError("Unsupported model.")

class DynamicalModel(abc.ABC):
    """Simulation of a dynamical model to be applied in the iLQR solution."""

    _id = 0

    def __init__(self, nX, nU, dt, id=None):
        if not id:
            id = DynamicalModel._id
            DynamicalModel._id += 1

        self.nX = nX
        self.nU = nU
        self.dt = dt
        self.id = id
        self.NX_EYE = np.eye(self.nX, dtype=np.float32)

    def __call__(self, x, u):
        """Zero-order hold to integrate continuous dynamics f"""
        return rk4_integration(self.f, x, u, self.dt, self.dt)

    @classmethod
    def _reset_ids(cls):
        cls._id = 0

    def __repr__(self):
        return f"{type(self).__name__}(n_x: {self.n_x}, n_u: {self.n_u}, id: {self.id})"

class UnicycleDynamics4D(DynamicalModel):
    def __init__(self, dt, *args, **kwargs):
        super().__init__(4, 2, dt,*args, **kwargs)
        self.model = Model.Unicycle4D

    def __call__(self, x, u):
        return integrate(x, u, self.dt, self.model)
    
    def f(self, x, u):
        return f(x, u, self.model)

    def linearize(self, x, u):
        return linearize(x, u, self.dt, self.model)

class MultiDynamicalModel(DynamicalModel):
    """Encompasses the dynamical simulation and linearization for a collection of
    DynamicalModel's
    """

    def __init__(self, submodels):
        self.submodels = submodels
        self.n_players = len(submodels)

        self.x_dims = [submodel.nX for submodel in submodels]
        self.u_dims = [submodel.nU for submodel in submodels]
        self.ids = [submodel.id for submodel in submodels]

        super().__init__(sum(self.x_dims), sum(self.u_dims), submodels[0].dt, -1)

    def f(self, x, u):
        """Derivative of the current combined states and controls"""
        xn = np.zeros_like(x)
        nx = self.x_dims[0]
        nu = self.u_dims[0]
        for i, model in enumerate(self.submodels):
            xn[i * nx : (i + 1) * nx] = model.f(
                x[i * nx : (i + 1) * nx], u[i * nu : (i + 1) * nu]
            )
        return xn

    def __call__(self, x, u):
        """Zero-order hold to integrate continuous dynamics f"""

        # return forward_euler_integration(self.f, x, u, self.dt)
        # return rk4_integration(self.f, x, u, self.dt, self.dt)
        xn = np.zeros_like(x)
        nx = self.x_dims[0]
        nu = self.u_dims[0]
        for i, model in enumerate(self.submodels):
            xn[i * nx : (i + 1) * nx] = model.__call__(
                x[i * nx : (i + 1) * nx], u[i * nu : (i + 1) * nu]
            )
        return xn

    def linearize(self, x, u):
        sub_linearizations = [
            submodel.linearize(xi.flatten(), ui.flatten())
            for submodel, xi, ui in zip(
                self.submodels,
                split_agents_gen(x, self.x_dims),
                split_agents_gen(u, self.u_dims),
            )
        ]

        sub_As = [AB[0] for AB in sub_linearizations]
        sub_Bs = [AB[1] for AB in sub_linearizations]

        return uniform_block_diag(*sub_As), uniform_block_diag(*sub_Bs)

    def split(self, graph):
        """Split this model into submodels dictated by the interaction graph"""
        split_dynamics = []
        for problem in graph:
            split_dynamics.append(
                MultiDynamicalModel(
                    [model for model in self.submodels if model.id in graph[problem]]
                )
            )

        return split_dynamics

    def __repr__(self):
        sub_reprs = ",\n\t".join([repr(submodel) for submodel in self.submodels])
        return f"MultiDynamicalModel(\n\t{sub_reprs}\n)"