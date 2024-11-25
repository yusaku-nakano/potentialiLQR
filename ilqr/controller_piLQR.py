import numpy as np
from time import perf_counter

class iLQR:
    # Initial regularization scaling
    REGU_DELTA_INIT = 2.0
    # Minimum and maximum regularization on hessian in backward pass
    REGU_MIN = 1e-6
    REGU_MAX = 1e3

    def __init__(self, problem, N=10):
        '''
           iterative Linear Quadratic Regulator
           Args:
             dynamics: dynamics container
             cost: cost container
        '''
        self.problem = problem
        self.params = {'alphas'  : 0.5**np.arange(8), #line search candidates
                       'regu_init': 20,    #initial regularization factor
                       'max_regu' : 10000,
                       'min_regu' : 0.001}
        self.N = N
        self.cost = self.problem.game_cost
        self.f = self.problem.dynamics
        self.nX = self.problem.dynamics.nX
        self.nU = self.problem.dynamics.nU
        self.dt = self.problem.dynamics.dt
        self.regu = 1.0
        self.regu_delta = 2.0

    def rollout(self, x0, U):
        '''
        Rollout with initial state and control trajectory
        '''
        X = np.empty((U.shape[0] + 1, x0.shape[0]))
        X[0] = x0.flatten()
        J = 0
        for t in range(U.shape[0]):
            X[t + 1] = self.f(X[t], U[t])
            J += self.cost(X[t], U[t]).item()
        J += self.cost(X[-1], np.zeros((self.nU)), terminal=True).item()
        return X, J
    
    def forward_pass(self, X, U, ks, Ks, alpha):
        '''
        Forward Pass
        '''
        X_next = np.empty(X.shape)
        U_next = np.empty(U.shape)
        J_new = 0.0

        X_next[0] = X[0]
        
        for t in range(U.shape[0]):
            delta_x = X_next[t] - X[t]
            delta_u = Ks[t] @ delta_x + alpha * ks[t]

            U_next[t] = U[t] + delta_u
            X_next[t + 1] = self.f(X_next[t], U_next[t])

            J_new += self.cost(X_next[t], U_next[t]).item()
        J_new += self.cost(X_next[-1], np.zeros((self.nU)), terminal=True).item()

        return X_next, U_next, J_new

    def backward_pass(self, X, U, regu):
        '''
        Backward Pass
        '''
        ks = np.empty(U.shape)
        Ks = np.empty((U.shape[0], U.shape[1], X.shape[1]))

        delta_V = 0
        V_x, _, V_xx, _, _ = self.cost.quadraticize(X[-1], np.zeros(self.nU), terminal=True)
        regu_I = regu*np.eye(V_xx.shape[0])
        for t in range(U.shape[0] - 1, -1, -1):

            f_x, f_u = self.f.linearize(X[t], U[t])
            l_x, l_u, l_xx, l_uu, l_ux  = self.cost.quadraticize(X[t], U[t])

            # Q_terms
            Q_x  = l_x  + f_x.T@V_x
            Q_u  = l_u  + f_u.T@V_x
            Q_xx = l_xx + f_x.T@V_xx@f_x
            Q_ux = l_ux + f_u.T@V_xx@f_x
            Q_uu = l_uu + f_u.T@V_xx@f_u

            # gains
            f_u_dot_regu = f_u.T@regu_I
            Q_ux_regu = Q_ux + f_u_dot_regu@f_x
            Q_uu_regu = Q_uu + f_u_dot_regu@f_u

            k = -np.linalg.solve(Q_uu_regu,Q_u)
            K = -np.linalg.solve(Q_uu_regu,Q_ux_regu)
            ks[t] = k 
            Ks[t] = K

            # V_terms
            V_x  = Q_x + K.T@Q_u + Q_ux.T@k + K.T@Q_uu@k
            V_xx = Q_xx + 2*K.T@Q_ux + K.T@Q_uu@K
            #expected cost reduction
            delta_V += Q_u.T@k + 0.5*k.T@Q_uu@k

        return ks, Ks, delta_V

    def run_ilqr(self, x0, u_init, max_iters=50, early_stop = True,
                alphas = 0.5**np.arange(8), tol = 1e-3):
        '''
        iLQR main loop
        '''
        U = u_init

        self.regu = 1.0
        self.regu_delta = 2.0

        x0 = x0.reshape(-1, 1)

        is_converged = False
        # First forward rollout
        X, J_star = self.rollout(x0, U)
        # cost trace
        J_trace = [J_star]

        print(f"0/{max_iters}\tJ: {J_star:g}")
        # Run main loop
        for it in range(max_iters):
            accept = False
            ks, Ks, exp_cost_redu = self.backward_pass(X, U, self.regu)

            # Early termination if improvement is small
            if it > 3 and early_stop and np.abs(exp_cost_redu) < 1e-5: 
                break

            # Backtracking line search
            for alpha in alphas:
                X_next, U_next, J = self.forward_pass(X, U, ks, Ks, alpha)
                if J < J_star:
                    if abs((J_star - J) / J_star) < tol:
                        is_converged = True
                    # Accept new trajectories and lower regularization
                    J_star = J
                    X = X_next
                    U = U_next
                    # Decrease regularization parameter when the forward pass is successful
                    self.regu_delta = min(1.0, self.regu_delta) / self.REGU_DELTA_INIT
                    self.regu *= self.regu_delta
                    if self.regu <= self.REGU_MIN:
                        self.regu = 0.0

                    accept = True
                    break
            if not accept:
                # Increase regularization parameter when the forward pass fails (e.g. new trajectory does not reduce cost)
                self.regu_delta = max(1.0, self.regu_delta) * self.REGU_DELTA_INIT
                self.regu = max(self.REGU_MIN, self.regu * self.regu_delta)
                if self.regu >= self.REGU_MAX:
                    print("Exceeded max regularization term...")
                    break
            if is_converged: 
                break

            J_trace.append(J_star)

            print(
                    f"{it+1}/{max_iters}\tJ: {J_star:g}\tregu: {self.regu:g}\tregu_delta: {self.regu_delta:g}"
            )

        return X, U, J_trace, J

class RHC:
    """
    Receding Horizon Controller
    """
    def __init__(self, x0, controller, stepsize = 1):
        self.x = x0
        self.controller = controller
        self.stepsize = stepsize
        self.N = self.controller.N
    
    def solve(self, Uinit, J_converge=1.0, **kwargs):
        i = 0
        U = Uinit
        while True:
            print("-" * 50 + f"\nHorizon {i}")
            i += 1

            # Fit the current state initializing with our control sequence.
            if U.shape != (self._controller.N, self._controller.n_u):
                raise RuntimeError

            X, U, J = self._controller.solve(self.x, U, **kwargs)

            # Shift the state to our predicted value. NOTE: this can be
            # updated externally for actual sensor feedback.
            self.x = X[self.step_size]

            yield X[: self.step_size], U[: self.step_size], J

            U = U[self.step_size :]
            U = np.vstack([U, np.zeros((self.step_size, self._controller.n_u))])

            if J < J_converge:
                print("Converged!")
                break