"""Optional finite-H0 diagnostics for DeepXDE; no TDA runtime dependency.

Uses DeepXDE's public Callback and Model.predict interfaces.
Distributed under LGPL-2.1; see LICENSE.
"""
import sys
import numpy as np
from deepxde import config
from deepxde.callbacks import Callback

__all__ = ["TopologicalFeatureMonitor"]


def _grid_h0_persistence(field):
    """Return finite H0 persistence lifetimes for a 1D or 2D cubical grid."""
    field = np.asarray(field, dtype=float)
    if field.ndim not in (1, 2) or field.size == 0:
        raise ValueError("`field` must be a non-empty 1D or 2D array.")
    if not np.all(np.isfinite(field)):
        raise ValueError("`field` contains non-finite values.")

    values = field.ravel()
    order = np.argsort(values, kind="stable")
    parent = np.arange(values.size)
    birth = values.copy()
    active = np.zeros(values.size, dtype=bool)
    lifetimes = []

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def neighbors(index):
        if field.ndim == 1:
            if index > 0:
                yield index - 1
            if index + 1 < values.size:
                yield index + 1
            return

        nrows, ncols = field.shape
        row, col = divmod(index, ncols)
        # Closed cubical cells that touch at a corner are connected, hence the
        # eight-neighbor relation used by cubical-complex H0 persistence.
        for other_row in range(max(0, row - 1), min(nrows, row + 2)):
            for other_col in range(max(0, col - 1), min(ncols, col + 2)):
                other = other_row * ncols + other_col
                if other != index:
                    yield other

    for index in order:
        index = int(index)
        active[index] = True
        for other in neighbors(index):
            if not active[other]:
                continue
            root, other_root = find(index), find(other)
            if root == other_root:
                continue
            if birth[root] <= birth[other_root]:
                survivor, dying = root, other_root
            else:
                survivor, dying = other_root, root
            lifetime = values[index] - birth[dying]
            if lifetime > 0:
                lifetimes.append(lifetime)
            parent[dying] = survivor

    return np.asarray(lifetimes, dtype=float)


class TopologicalFeatureMonitor(Callback):
    """Monitor finite H0 persistence of a scalar field on a regular grid.

    The scalar field can be a component of the network output or the result of
    an ``operator`` passed to ``Model.predict``. At each check, the callback
    records the number of finite H0 features, their total persistence, and
    their maximum persistence. The computation uses NumPy and does not require
    a topological-data-analysis dependency.

    This is a diagnostic descriptor, not a convergence or correctness
    certificate. Its interpretation depends on the chosen field and problem.

    Args:
        x: Evaluation points ordered so the prediction can be reshaped in C
            order to ``grid_shape``.
        grid_shape: Shape of the regular 1D or 2D grid.
        period: Minimum training-iteration interval between checks, evaluated
            when the optimizer calls the epoch-end hook. Optimizers without
            that hook are monitored only at training start and end.
        component: Output component to monitor.
        operator: Optional operator with the same contract as the ``operator``
            argument of ``Model.predict``. This can produce a derived field or
            a PDE residual.
        superlevel: If True, compute sublevel persistence of the negated field.
        normalize: If True, divide the field by its maximum absolute value
            before computing persistence.
        min_persistence: Discard features with persistence less than or equal
            to this value, after optional normalization.
        filename: Output values to ``filename``. If ``None``, output to screen.
        precision: Number of digits after the decimal point in logged values.
    """

    def __init__(
        self,
        x,
        grid_shape,
        period=1000,
        component=0,
        operator=None,
        superlevel=False,
        normalize=False,
        min_persistence=0,
        filename=None,
        precision=4,
    ):
        super().__init__()
        if not isinstance(period, (int, np.integer)) or period <= 0:
            raise ValueError("`period` must be a positive integer.")
        if not isinstance(component, (int, np.integer)) or component < 0:
            raise ValueError("`component` must be a non-negative integer.")
        if not np.isfinite(min_persistence) or min_persistence < 0:
            raise ValueError("`min_persistence` must be finite and non-negative.")

        self.x = np.asarray(x, dtype=config.real(np))
        if self.x.ndim == 1:
            self.x = self.x[:, None]
        self.grid_shape = tuple(grid_shape)
        if len(self.grid_shape) not in (1, 2) or any(
            not isinstance(size, (int, np.integer)) or size <= 0
            for size in self.grid_shape
        ):
            raise ValueError("`grid_shape` must contain one or two positive sizes.")
        if self.x.shape[0] != int(np.prod(self.grid_shape)):
            raise ValueError("The number of points in `x` must match `grid_shape`.")

        self.period = period
        self.component = component
        self.operator = operator
        self.superlevel = superlevel
        self.normalize = normalize
        self.min_persistence = min_persistence
        self.precision = precision
        self.file = sys.stdout if filename is None else open(filename, "w", buffering=1)

        self.value = None
        self.history = []
        self._last_step = None

    def on_train_begin(self):
        self._evaluate()

    def on_epoch_end(self):
        if self.model.train_state.iteration - self._last_step >= self.period:
            self._evaluate()

    def on_train_end(self):
        if self.model.train_state.iteration != self._last_step:
            self._evaluate()

    def _evaluate(self):
        prediction = np.asarray(self.model.predict(self.x, operator=self.operator))
        if prediction.ndim == 1:
            if self.component != 0:
                raise ValueError("A one-dimensional prediction only has component 0.")
            field = prediction
        elif prediction.ndim == 2 and self.component < prediction.shape[1]:
            field = prediction[:, self.component]
        else:
            raise ValueError("The selected output component does not exist.")

        field = field.reshape(self.grid_shape)
        if self.superlevel:
            field = -field
        if self.normalize:
            scale = np.max(np.abs(field))
            if scale > 0:
                field = field / scale

        lifetimes = _grid_h0_persistence(field)
        lifetimes = lifetimes[lifetimes > self.min_persistence]
        self.value = {
            "num_features": int(lifetimes.size),
            "total_persistence": float(np.sum(lifetimes)),
            "max_persistence": float(np.max(lifetimes)) if lifetimes.size else 0.0,
        }
        record = {"step": self.model.train_state.iteration, **self.value}
        self.history.append(record)
        self._last_step = record["step"]
        print(
            "{} {} {:.{p}e} {:.{p}e}".format(
                record["step"],
                record["num_features"],
                record["total_persistence"],
                record["max_persistence"],
                p=self.precision,
            ),
            file=self.file,
        )
        self.file.flush()

    def get_value(self):
        """Return the most recently computed descriptor."""
        return self.value

    def get_history(self):
        """Return the descriptor history."""
        return self.history
