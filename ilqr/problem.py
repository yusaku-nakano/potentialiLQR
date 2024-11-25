"""Logic to combine dynamics and cost in one framework to simplify distribution"""

from .controller_piLQR import iLQR
from .cost import TrackingCost, GameCost
from .dynamics import DynamicalModel, MultiDynamicalModel


class ilqrProblem:
    """Centralized optimal control problem that combines dynamics and cost"""

    def __init__(self, dynamics, cost):
        self.dynamics = dynamics
        self.game_cost = cost
        self.n_agents = 1

        if isinstance(cost, GameCost):
            self.n_agents = len(cost.ref_costs)

    @property
    def ids(self):
        if not isinstance(self.dynamics, MultiDynamicalModel):
            raise NotImplementedError(
                "Only MultiDynamicalModel's have an 'ids' attribute"
            )
        if not self.dynamics.ids == self.game_cost.ids:
            raise ValueError(f"Dynamics and cost have inconsistent ID's: {self}")
        return self.dynamics.ids.copy()

    def split(self, graph):
        """Split up this centralized problem into a list of distributed
        sub-problems.
        """

        split_dynamics = self.dynamics.split(graph)
        split_costs = self.game_cost.split(graph)

        return [
            ilqrProblem(dynamics, cost)
            for dynamics, cost in zip(split_dynamics, split_costs)
        ]

    def extract(self, X, U, id_):
        """Extract the state and controls for a particular agent id_ from the
        concatenated problem state/controls
        """

        if id_ not in self.ids:
            raise IndexError(f"Index {id_} not in ids: {self.ids}.")

        # NOTE: Assume uniform dynamical models.
        ext_ind = self.ids.index(id_)
        x_dim = self.game_cost.x_dims[0]
        u_dim = self.game_cost.u_dims[0]
        Xi = X[:, ext_ind * x_dim : (ext_ind + 1) * x_dim]
        Ui = U[:, ext_ind * u_dim : (ext_ind + 1) * u_dim]

        return Xi, Ui

def solve_subproblem(args, **kwargs):
    """Solve the sub-problem and extract results for this agent"""

    subproblem, x0, U, id_, verbose = args
    N = U.shape[0]

    subsolver = iLQR(subproblem, N)
    Xi, Ui, _ = subsolver.solve(x0, U, verbose=verbose, **kwargs)
    return *subproblem.extract(Xi, Ui, id_), id_


def solve_subproblem_starmap(subproblem, x0, U, id_):
    """Package up the input arguments for compatiblity with mp.imap()."""
    return solve_subproblem((subproblem, x0, U, id_))


def _reset_ids():
    """Set each of the agent specific ID's to zero for understandability"""
    DynamicalModel._reset_ids()
    TrackingCost._reset_ids()