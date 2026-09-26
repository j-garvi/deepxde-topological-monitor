"""Tests for the dependency-free topological feature monitor."""

import io

import numpy as np
import pytest

from deepxde_topology import TopologicalFeatureMonitor, _grid_h0_persistence


class _TrainState:
    iteration = 0


class _FakeModel:
    def __init__(self, prediction):
        self.prediction = np.asarray(prediction)
        self.train_state = _TrainState()
        self.calls = []

    def predict(self, x, operator=None):
        self.calls.append((np.asarray(x).copy(), operator))
        if operator is None:
            return self.prediction
        return operator(x, self.prediction)


def test_grid_h0_persistence_known_four_basin_field():
    field = np.array([[0.0, 2.0, 0.0], [2.0, 2.0, 2.0], [0.0, 2.0, 0.0]])
    np.testing.assert_allclose(np.sort(_grid_h0_persistence(field)), [2.0, 2.0, 2.0])


def test_grid_h0_persistence_handles_rectangular_grid():
    field = np.full((3, 5), 3.0)
    field[0, 0] = 0.0
    field[0, 4] = 1.0
    np.testing.assert_allclose(_grid_h0_persistence(field), [2.0])


def test_monitor_records_initial_periodic_and_final_values():
    x = np.indices((3, 3)).reshape(2, -1).T
    prediction = np.array(
        [[0.0], [2.0], [0.0], [2.0], [2.0], [2.0], [0.0], [2.0], [0.0]]
    )
    monitor = TopologicalFeatureMonitor(x, (3, 3), period=2)
    monitor.file = io.StringIO()
    model = _FakeModel(prediction)
    monitor.set_model(model)

    monitor.on_train_begin()
    for step in range(1, 4):
        model.train_state.iteration = step
        monitor.on_epoch_end()
    monitor.on_train_end()

    assert [item["step"] for item in monitor.get_history()] == [0, 2, 3]
    assert monitor.get_value() == {
        "num_features": 3,
        "total_persistence": 6.0,
        "max_persistence": 2.0,
    }


def test_monitor_supports_operator_component_normalization_and_superlevel():
    x = np.linspace(0, 1, 5)[:, None]
    prediction = np.column_stack((np.zeros(5), [0.0, 4.0, 0.0, 4.0, 0.0]))

    def select_second_component(_, output):
        return output[:, 1:]

    monitor = TopologicalFeatureMonitor(
        x,
        (5,),
        operator=select_second_component,
        superlevel=True,
        normalize=True,
    )
    monitor.file = io.StringIO()
    model = _FakeModel(prediction)
    monitor.set_model(model)
    monitor.on_train_begin()

    assert model.calls[0][1] is select_second_component
    assert monitor.get_value() == {
        "num_features": 1,
        "total_persistence": 1.0,
        "max_persistence": 1.0,
    }


def test_monitor_validates_grid_size_and_nonfinite_predictions():
    x = np.linspace(0, 1, 4)[:, None]
    try:
        TopologicalFeatureMonitor(x, (3, 2))
    except ValueError as error:
        assert "match" in str(error)
    else:
        raise AssertionError("Mismatched grid size should raise ValueError.")

    monitor = TopologicalFeatureMonitor(x, (4,))
    monitor.file = io.StringIO()
    monitor.set_model(_FakeModel([[0.0], [np.nan], [1.0], [0.0]]))
    try:
        monitor.on_train_begin()
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("Non-finite predictions should raise ValueError.")


def make_monitor(period=2):
    monitor = TopologicalFeatureMonitor(np.arange(5)[:, None], (5,), period=period)
    monitor.file = io.StringIO()
    monitor.set_model(_FakeModel([0, 2, 0, 2, 0]))
    return monitor


def test_final_record_without_epoch_hooks():
    monitor = make_monitor()
    monitor.on_train_begin()
    monitor.model.train_state.iteration = 10
    monitor.on_train_end()
    assert [r["step"] for r in monitor.history] == [0, 10]


def test_period_counts_iterations_instead_of_hook_calls():
    monitor = make_monitor(period=10)
    monitor.on_train_begin()
    for step in (7, 14, 21, 28):
        monitor.model.train_state.iteration = step
        monitor.on_epoch_end()
    monitor.on_train_end()
    assert [r["step"] for r in monitor.history] == [0, 14, 28]


def test_zero_steps_and_repeated_training():
    monitor = make_monitor()
    monitor.on_train_begin()
    monitor.on_train_end()
    assert len(monitor.history) == 1
    monitor.on_train_begin()
    monitor.model.train_state.iteration = 2
    monitor.on_epoch_end()
    monitor.on_train_end()
    assert [r["step"] for r in monitor.history] == [0, 0, 2]


@pytest.mark.parametrize("kwargs", [
    {"period": 1.5}, {"period": 0}, {"component": 0.5},
    {"min_persistence": np.nan}, {"min_persistence": np.inf},
    {"grid_shape": (2.5, 2)},
])
def test_invalid_configuration(kwargs):
    options = {"grid_shape": (5,), **kwargs}
    with pytest.raises(ValueError):
        TopologicalFeatureMonitor(np.arange(5)[:, None], **options)


def test_threshold_and_constant_field():
    monitor = make_monitor()
    monitor.min_persistence = 2
    monitor.on_train_begin()
    assert monitor.value["num_features"] == 0
    assert _grid_h0_persistence(np.zeros((3, 4))).size == 0


def test_corner_connectivity():
    assert _grid_h0_persistence(np.array([[0, 2], [2, 0]])).size == 0


def test_matches_gudhi_random_fields():
    gudhi = pytest.importorskip("gudhi")
    rng = np.random.default_rng(17)
    for shape in [(1,), (2,), (17,), (1, 7), (7, 1), (2, 2), (3, 5), (8, 9)]:
        for discrete in (False, True):
            for _ in range(50):
                field = (rng.integers(-3, 4, size=shape).astype(float)
                         if discrete else rng.normal(size=shape))
                complex_ = gudhi.CubicalComplex(top_dimensional_cells=field)
                complex_.persistence()
                diagram = complex_.persistence_intervals_in_dimension(0)
                finite = diagram[np.isfinite(diagram[:, 1])]
                expected = np.sort(finite[:, 1] - finite[:, 0])
                np.testing.assert_allclose(np.sort(_grid_h0_persistence(field)), expected)


def test_real_pytorch_training_with_derivative():
    import deepxde as dde
    if dde.backend.backend_name != "pytorch":
        pytest.skip("Integration check uses PyTorch")
    geom = dde.geometry.Interval(-1, 1)
    data = dde.data.Function(geom, lambda x: np.sin(3 * np.pi * x), 16, 16)
    model = dde.Model(data, dde.nn.FNN([1, 8, 1], "tanh", "Glorot normal"))
    model.compile("adam", lr=0.001, verbose=0)
    x = np.linspace(-1, 1, 17)[:, None]
    monitors = [TopologicalFeatureMonitor(x, (17,), period=2),
                TopologicalFeatureMonitor(x, (17,), period=2,
                    operator=lambda x, y: dde.grad.jacobian(y, x))]
    for monitor in monitors:
        monitor.file = io.StringIO()
    model.train(iterations=5, callbacks=monitors, verbose=0)
    for monitor in monitors:
        assert [r["step"] for r in monitor.history] == [0, 2, 4, 5]
