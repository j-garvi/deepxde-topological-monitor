"""Burgers PINN with finite-H0 monitoring of |du/dx| at t=0.5.

Run after installing this package and a DeepXDE backend:
    DDE_BACKEND=pytorch python examples/burgers_monitor.py
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import deepxde as dde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deepxde_topology import TopologicalFeatureMonitor


class TimedMonitor(TopologicalFeatureMonitor):
    def __init__(self, *args, **kwargs):
        self.evaluation_seconds = 0.0
        super().__init__(*args, **kwargs)

    def _evaluate(self):
        start = perf_counter()
        super()._evaluate()
        self.evaluation_seconds += perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--period", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("example-output"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dde.config.set_random_seed(17)
    if dde.backend.backend_name == "pytorch":
        dde.backend.torch.set_num_threads(1)

    def pde(x, y):
        u_x = dde.grad.jacobian(y, x, i=0, j=0)
        u_t = dde.grad.jacobian(y, x, i=0, j=1)
        u_xx = dde.grad.hessian(y, x, i=0, j=0)
        return u_t + y * u_x - (0.01 / np.pi) * u_xx

    def gradient_magnitude(x, y):
        return dde.backend.abs(dde.grad.jacobian(y, x, i=0, j=0))

    domain = dde.geometry.GeometryXTime(
        dde.geometry.Interval(-1, 1), dde.geometry.TimeDomain(0, 1)
    )
    bc = dde.icbc.DirichletBC(domain, lambda x: 0, lambda _, boundary: boundary)
    ic = dde.icbc.IC(domain, lambda x: -np.sin(np.pi * x[:, :1]),
                    lambda _, initial: initial)
    data = dde.data.TimePDE(domain, pde, [bc, ic], num_domain=256,
                           num_boundary=32, num_initial=64, num_test=256)
    model = dde.Model(data, dde.nn.FNN([2, 32, 32, 32, 1], "tanh", "Glorot normal"))
    model.compile("adam", lr=1e-3)

    space = np.linspace(-1, 1, 129)
    x_grid = np.column_stack((space, np.full_like(space, 0.5)))
    monitor = TimedMonitor(x_grid, grid_shape=(len(space),), period=args.period,
                           operator=gradient_magnitude, superlevel=True,
                           min_persistence=0.05,
                           filename=str(args.output / "topology.dat"))
    initial_field = model.predict(x_grid, operator=gradient_magnitude).ravel()
    start = perf_counter()
    try:
        loss, state = model.train(iterations=args.iterations, callbacks=[monitor],
                                  display_every=args.period)
    finally:
        monitor.file.close()
    training_seconds = perf_counter() - start
    final_field = model.predict(x_grid, operator=gradient_magnitude).ravel()

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    axes[0].semilogy(loss.steps, np.sum(loss.loss_train, axis=1), label="Train")
    axes[0].semilogy(loss.steps, np.sum(loss.loss_test, axis=1), label="Test")
    axes[0].set(xlabel="Training iteration", ylabel="Total loss")
    axes[0].legend()
    steps = [row["step"] for row in monitor.history]
    axes[1].plot(steps, [row["total_persistence"] for row in monitor.history])
    axes[1].set(xlabel="Training iteration", ylabel="Finite H0 total persistence",
                title="Superlevel sets of |du/dx| at t=0.5")
    axes[2].plot(space, initial_field, label="Initial")
    axes[2].plot(space, final_field, label="Final")
    axes[2].set(xlabel="x", ylabel="|du/dx|", title="Monitored field")
    axes[2].legend()
    fig.savefig(args.output / "burgers_monitor.png", dpi=160)
    plt.close(fig)
    summary = {
        "backend": dde.backend.backend_name,
        "deepxde_version": dde.__version__,
        "iterations": int(state.iteration),
        "grid_points": len(space),
        "period": args.period,
        "training_seconds": training_seconds,
        "monitor_evaluation_seconds": monitor.evaluation_seconds,
        "monitor_fraction_of_training": monitor.evaluation_seconds / training_seconds,
        "initial_loss": float(np.sum(loss.loss_train[0])),
        "final_loss": float(np.sum(loss.loss_train[-1])),
        "initial_test_loss": float(np.sum(loss.loss_test[0])),
        "final_test_loss": float(np.sum(loss.loss_test[-1])),
        "history": monitor.history,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Saved plots, descriptor history, and timings to", args.output)


if __name__ == "__main__":
    main()
