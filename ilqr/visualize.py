#!/usr/bin/env python

from functools import reduce
from itertools import cycle
from operator import mul

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from ilqr.utils import split_agents, compute_pairwise_distance


plt.rcParams.update(
    {
        "axes.grid": False,
        "figure.constrained_layout.use": True,
        "font.serif": ["Palatino"],
        "ps.distiller.res": 8000,
    }
)


def set_bounds(xydata, ax=None, zoom=0.1):
    """
    Adjusts the axis limits of a plot based on the given data (xydata) with an optional zoom margin.
    Useful for dynamically setting plot bounds to ensure all data points are visible.
    """

    xydata = np.atleast_2d(xydata)

    if not ax:
        ax = plt.gca()

    xmarg = xydata[:, 0].ptp() * zoom
    ymarg = xydata[:, 1].ptp() * zoom
    ax.set(
        xlim=(xydata[:, 0].min() - xmarg, xydata[:, 0].max() + xmarg),
        ylim=(xydata[:, 1].min() - ymarg, xydata[:, 1].max() + ymarg),
    )


def nchoosek(n, k):
    """
    Computes combinations (n choose k) using the formula:

    C(n,k) = n! / (k! * (n-k)!)
    """

    k = min(k, n - k)
    num = reduce(mul, range(n, n - k, -1), 1)
    denom = reduce(mul, range(1, k + 1), 1)
    return num // denom

def plot_solve(X, J, x_goal, x_dims=None, color_agents=False, n_d=2, ax=None):
    """Plot the resultant trajectory on plt.gcf()"""

    if n_d not in (2, 3):
        raise ValueError()

    if not x_dims:
        x_dims = [X.shape[1]]

    if not ax:
        if n_d == 2:
            ax = plt.gca()
        else:
            ax = plt.gcf().add_subplot(projection="3d")

    N = X.shape[0]
    n = np.arange(N)
    cm = plt.cm.Set2

    X_split = split_agents(X, x_dims)
    x_goal_split = split_agents(x_goal.reshape(1, -1), x_dims)

    for i, (Xi, xg) in enumerate(zip(X_split, x_goal_split)):
        c = n
        if n_d == 2:
            if color_agents:
                c = cm.colors[i]
                ax.plot(Xi[:, 0], Xi[:, 1], c=c, lw=5)
            else:
                ax.scatter(Xi[:, 0], Xi[:, 1], c=c)
            ax.scatter(Xi[0, 0], Xi[0, 1], 80, "g", "d", label="$x_0$")
            ax.scatter(xg[0, 0], xg[0, 1], 80, "r", "x", label="$x_f$")
        else:
            if color_agents:
                # c = [cm.colors[i]] * Xi.shape[0]
                c = cm.colors[i]
            ax.plot(Xi[:, 0], Xi[:, 1], Xi[:, 2], c=c, lw=4)
            ax.scatter(
                Xi[0, 0], Xi[0, 1], Xi[0, 2], 
                s=50, c="w", marker="d", edgecolors="k", label="$x_0$")
            ax.scatter(
                xg[0, 0], xg[0, 1], xg[0, 2], 
                s=50, c="k", marker="x", label="$x_f$")
            ax.scatter(
                Xi[-1, 0], Xi[-1, 1], Xi[-1,2], 
                s=50, color=c, marker="o", edgecolors="k")
            

    plt.margins(0.1)
    plt.title(f"Final Cost: {J:f}")
    plt.draw()


def plot_pairwise_distances(X, x_dims, n_dims, radius):
    """
    Render all-pairwise distances in the trajectory
    """

    # Get current axis
    ax = plt.gca()

    # Plot all pairwise distances
    ax.plot(compute_pairwise_distance(X, x_dims, n_dims[1]))
    ax.hlines(radius, *plt.xlim(), "r", ls="--", label="$d_{prox}$")
    ax.set_title("Inter-Agent Distances")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Pairwise Distance (m)")
    ax.legend()
    plt.draw()


def _setup_gif(axes, X, xf, x_dims, radius, distances):
    
    # axes: tuple of two matplotlib axes objects
    # First axis: used to plot trajectories of agents
    # Second axis: used to plot pairwise distances between agents
    ax1, ax2 = axes
    n_agents = len(x_dims)
    # handles1: creates list of handles to store the plot elements for the trajectories of the agents.
    handles1 = []

    # Loops over agents and creates plot element for each agent's trajectory
    for _, c in zip(range(n_agents), cycle(plt.cm.tab20.colors)):
        # 
        handles1.append(
            (
                ax1.plot(0, c=c, marker="o", markersize=4)[0],
                ax1.add_artist(
                    plt.Circle(
                        (np.nan, np.nan), radius, color="k", fill=True, alpha=0.3, lw=2
                    )
                ),
            )
        )
    
    # Loops over the agents and plots the final state of each agent as "x"
    for xg in split_agents(xf, x_dims):
        ax1.scatter(xg[0, 0], xg[0, 1], c="r", marker="x", zorder=10)

    X_cat = np.vstack(split_agents(X, x_dims))
    set_bounds(X_cat, axes[0], zoom=0.15)
    ax1.set_title("Trajectories")
    plt.draw()

    handles2 = []
    # Calculating number of ways to choose 2 agents from group of n_agents agents
    # Want to plot the distances between all pair of agents
    n_pairs = nchoosek(n_agents, 2)
    for _, c in zip(range(n_pairs), cycle(plt.cm.tab20.colors)):
        handles2.append(ax2.plot(0, c=c)[0])
    ax2.hlines(radius, 0, X.shape[0], "r", ls="--", label="$d_{prox}$")
    ax2.set_ylim(0.0, distances.max())
    ax2.set_title("Inter-Distances")
    ax2.set_ylabel("Distance [m]")
    ax2.set_xlabel("Time Step")
    ax2.legend()

    return (
        handles1,
        handles2,
    )


def _animate(t, handles1, handles2, X, x_dims, distances):
    """Animate the solution into a gif"""

    for i, (xi, hi) in enumerate(zip(split_agents(X, x_dims), handles1)):
        hi[0].set_xdata(xi[:t, 0])
        hi[0].set_ydata(xi[:t, 1])
        hi[1].set_center(xi[t - 1, :2])

    for i, hi in enumerate(handles2):
        hi.set_xdata(range(t))
        hi.set_ydata(distances[:t, i])

    plt.draw()
    return (
        *handles1,
        *handles2,
    )


def make_trajectory_gif(gifname, X, xf, x_dims, radius):
    """Create a GIF of the evolving trajectory"""

    _, axes = plt.subplots(1, 2, figsize=(10, 6))

    N = X.shape[0]
    distances = compute_pairwise_distance(X, x_dims)

    handles = _setup_gif(axes, X, xf.flatten(), x_dims, radius, distances)
    anim = FuncAnimation(
        plt.gcf(),
        _animate,
        frames=N + 1,
        fargs=(*handles, X, x_dims, distances),
        repeat=True,
    )
    anim.save(gifname, fps=N // 10, dpi=100)


def eyeball_scenario(x0, xf, n_agents, n_states):
    """Render the scenario in 2D"""
    plt.clf()

    plt.gca().set_aspect("equal")
    X = np.dstack(
        [x0.reshape(n_agents, n_states), xf.reshape(n_agents, n_states)]
    ).swapaxes(1, 2)
    for i, Xi in enumerate(X):
        plt.annotate(
            "", Xi[1, :2], Xi[0, :2], arrowprops=dict(facecolor=plt.cm.tab20.colors[i])
        )
    set_bounds(X.reshape(-1, n_states), zoom=0.2)
    plt.draw()