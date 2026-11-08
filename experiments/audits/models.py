"""Proxy models used in the audits."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor


torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class ModelFit:
    prediction: np.ndarray
    validation_rmse: float
    details: dict[str, object]


def one_hot(tokens: np.ndarray, alphabet_size: int) -> np.ndarray:
    return np.eye(alphabet_size, dtype=np.float32)[tokens].reshape(len(tokens), -1)


def pairwise_features(tokens: np.ndarray, alphabet_size: int) -> sparse.csr_matrix:
    sample_count, length = tokens.shape
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    offset = 0
    for position in range(length):
        rows.append(np.arange(sample_count))
        columns.append(offset + tokens[:, position].astype(int))
        offset += alphabet_size
    for left in range(length):
        for right in range(left + 1, length):
            rows.append(np.arange(sample_count))
            columns.append(
                offset + alphabet_size * tokens[:, left].astype(int) + tokens[:, right].astype(int)
            )
            offset += alphabet_size**2
    row = np.concatenate(rows)
    column = np.concatenate(columns)
    values = np.ones(len(row), dtype=np.float32)
    return sparse.csr_matrix((values, (row, column)), shape=(sample_count, offset))


def linear_active_prediction(tokens: np.ndarray, alphabet_size: int, fitted: Ridge) -> np.ndarray:
    coefficients = np.asarray(fitted.coef_).reshape(-1)
    result = np.full(len(tokens), float(fitted.intercept_), dtype=np.float64)
    offset = 0
    for position in range(tokens.shape[1]):
        result += coefficients[offset + tokens[:, position]]
        offset += alphabet_size
    for left in range(tokens.shape[1]):
        for right in range(left + 1, tokens.shape[1]):
            result += coefficients[
                offset + alphabet_size * tokens[:, left].astype(int) + tokens[:, right].astype(int)
            ]
            offset += alphabet_size**2
    return result


def fit_pairwise_ridge(
    tokens: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    alphabet_size: int,
    seed: int,
) -> ModelFit:
    del seed
    train_features = pairwise_features(tokens[train], alphabet_size)
    validation_features = pairwise_features(tokens[validation], alphabet_size)
    choices: list[tuple[float, float, Ridge]] = []
    for penalty in (1e-4, 1e-2, 1.0, 100.0):
        fitted = Ridge(alpha=penalty, solver="lsqr", max_iter=5000).fit(train_features, target[train])
        rmse = float(np.sqrt(np.mean((fitted.predict(validation_features) - target[validation]) ** 2)))
        choices.append((rmse, penalty, fitted))
    rmse, penalty, fitted = min(choices, key=lambda item: (item[0], item[1]))
    prediction = linear_active_prediction(tokens, alphabet_size, fitted)
    return ModelFit(prediction, rmse, {
        "objective": "squared-error pairwise categorical ridge regression",
        "selected_alpha": penalty,
        "validation_choices": {str(item[1]): item[0] for item in choices},
    })


def _predict_sklearn_chunks(model: object, tokens: np.ndarray, alphabet_size: int) -> np.ndarray:
    result = np.empty(len(tokens), dtype=np.float64)
    for start in range(0, len(tokens), 100_000):
        stop = min(start + 100_000, len(tokens))
        result[start:stop] = model.predict(one_hot(tokens[start:stop], alphabet_size))
    return result


def fit_extra_trees(
    tokens: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    alphabet_size: int,
    seed: int,
) -> ModelFit:
    train_x = one_hot(tokens[train], alphabet_size)
    validation_x = one_hot(tokens[validation], alphabet_size)
    choices: list[tuple[float, int, ExtraTreesRegressor]] = []
    for leaf in (1, 2, 4):
        fitted = ExtraTreesRegressor(
            n_estimators=128,
            min_samples_leaf=leaf,
            max_features=1.0,
            random_state=seed + leaf,
            n_jobs=1,
        ).fit(train_x, target[train])
        rmse = float(np.sqrt(np.mean((fitted.predict(validation_x) - target[validation]) ** 2)))
        choices.append((rmse, leaf, fitted))
    rmse, leaf, fitted = min(choices, key=lambda item: (item[0], item[1]))
    prediction = _predict_sklearn_chunks(fitted, tokens, alphabet_size)
    return ModelFit(prediction, rmse, {
        "objective": "ExtraTrees squared-error regression on positional one-hot features",
        "n_estimators": 128,
        "n_jobs": 1,
        "selected_min_samples_leaf": leaf,
        "validation_choices": {str(item[1]): item[0] for item in choices},
    })


def fit_mlp(
    tokens: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    alphabet_size: int,
    seed: int,
) -> ModelFit:
    train_x = one_hot(tokens[train], alphabet_size)
    validation_x = one_hot(tokens[validation], alphabet_size)
    choices: list[tuple[float, float, MLPRegressor]] = []
    for penalty in (1e-5, 1e-3):
        fitted = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=penalty,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=seed,
        ).fit(train_x, target[train])
        rmse = float(np.sqrt(np.mean((fitted.predict(validation_x) - target[validation]) ** 2)))
        choices.append((rmse, penalty, fitted))
    rmse, penalty, fitted = min(choices, key=lambda item: (item[0], item[1]))
    prediction = _predict_sklearn_chunks(fitted, tokens, alphabet_size)
    return ModelFit(prediction, rmse, {
        "objective": "squared-error positional one-hot MLP regression",
        "hidden_layers": [128, 64],
        "selected_l2_penalty": penalty,
        "iterations": int(fitted.n_iter_),
        "validation_choices": {str(item[1]): item[0] for item in choices},
    })


class _ScoreNet(torch.nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def _torch_predict(model: _ScoreNet, tokens: np.ndarray, alphabet_size: int) -> np.ndarray:
    model.eval()
    result = np.empty(len(tokens), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(tokens), 100_000):
            stop = min(start + 100_000, len(tokens))
            values = torch.from_numpy(one_hot(tokens[start:stop], alphabet_size))
            result[start:stop] = model(values).cpu().numpy()
    return result


def fit_ranking_proxy(
    tokens: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    alphabet_size: int,
    seed: int,
) -> ModelFit:
    """Fit a RankNet surrogate, then positive-affine calibrate without changing rank.

    The pairwise logistic objective follows Burges et al. (2005), "Learning
    to Rank using Gradient Descent", ICML, DOI 10.1145/1102351.1102363. This
    finite surrogate is not presented as a reproduction of an offline-MBO
    search optimizer from that or any later paper.
    """
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_x = torch.from_numpy(one_hot(tokens[train], alphabet_size))
    train_y = torch.from_numpy(target[train].astype(np.float32))
    model = _ScoreNet(train_x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    fixed_left = rng.integers(len(validation), size=20_000)
    fixed_right = rng.integers(len(validation), size=20_000)
    validation_x = torch.from_numpy(one_hot(tokens[validation], alphabet_size))
    validation_y = target[validation]
    best_accuracy = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_step = 0
    for step in range(1, 801):
        left = rng.integers(len(train), size=1024)
        right = rng.integers(len(train), size=1024)
        target_order = (train_y[left] > train_y[right]).float()
        logits = model(train_x[left]) - model(train_x[right])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target_order)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 40 == 0:
            with torch.no_grad():
                values = model(validation_x).numpy()
            accuracy = float(np.mean(
                (values[fixed_left] > values[fixed_right])
                == (validation_y[fixed_left] > validation_y[fixed_right])
            ))
            if accuracy > best_accuracy + 1e-6:
                best_accuracy = accuracy
                best_state = copy.deepcopy(model.state_dict())
                best_step = step
    if best_state is None:
        raise AssertionError("RankNet training did not produce a checkpoint")
    model.load_state_dict(best_state)
    raw_prediction = _torch_predict(model, tokens, alphabet_size)
    calibrator = LinearRegression().fit(raw_prediction[validation, None], target[validation])
    slope = float(calibrator.coef_[0])
    if slope <= 0:
        slope = float(np.std(target[validation]) / max(np.std(raw_prediction[validation]), 1e-12))
        intercept = float(np.mean(target[validation]) - slope * np.mean(raw_prediction[validation]))
    else:
        intercept = float(calibrator.intercept_)
    prediction = intercept + slope * raw_prediction
    rmse = float(np.sqrt(np.mean((prediction[validation] - target[validation]) ** 2)))
    return ModelFit(prediction, rmse, {
        "objective": "RankNet pairwise logistic loss on random labeled training pairs",
        "method_reference": "Burges et al. (2005), Learning to Rank using Gradient Descent, DOI 10.1145/1102351.1102363",
        "scope": "implemented finite ranking-trained surrogate; not a reproduction of a broader offline-MBO optimizer",
        "pair_batch_size": 1024,
        "optimization_steps": 800,
        "selected_step": best_step,
        "validation_pairwise_accuracy": best_accuracy,
        "calibration": "positive affine validation calibration; candidate ordering unchanged",
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    })


def fit_com_proxy(
    tokens: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    selection: np.ndarray,
    alphabet_size: int,
    seed: int,
) -> ModelFit:
    """Finite-pool COM-style conservative regression.

    This is a documented finite discrete adaptation, not a claim of reproducing
    every optimizer detail in the original continuous COM implementation.
    """
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_x = torch.from_numpy(one_hot(tokens[train], alphabet_size))
    train_y = torch.from_numpy(target[train].astype(np.float32))
    model = _ScoreNet(train_x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(500):
        batch = rng.integers(len(train), size=256)
        loss = torch.nn.functional.mse_loss(model(train_x[batch]), train_y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    alpha = 0.1
    hard_global = selection[:1024]
    for step in range(200):
        if step % 40 == 0:
            pool_prediction = _torch_predict(model, tokens[selection], alphabet_size)
            local_order = np.lexsort((selection, -pool_prediction))
            hard_global = selection[local_order[:1024]]
        batch = rng.integers(len(train), size=256)
        hard = rng.choice(hard_global, size=256, replace=True)
        hard_x = torch.from_numpy(one_hot(tokens[hard], alphabet_size))
        train_prediction = model(train_x[batch])
        regression = torch.nn.functional.mse_loss(train_prediction, train_y[batch])
        conservatism = model(hard_x).mean() - train_prediction.mean()
        loss = regression + alpha * conservatism
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
    prediction = _torch_predict(model, tokens, alphabet_size)
    rmse = float(np.sqrt(np.mean((prediction[validation] - target[validation]) ** 2)))
    return ModelFit(prediction, rmse, {
        "objective": "training MSE + alpha*(mean predicted hard-negative score - mean predicted training score)",
        "method_reference": "Trabucco et al. (2021), Conservative Objective Models for Effective Offline Model-Based Optimization, PMLR 139:10358-10368",
        "adaptation": "finite discrete COM-style; hard negatives refreshed from highest-scored unlabeled selection candidates",
        "pretraining_steps": 500,
        "conservative_steps": 200,
        "hard_negative_count": 1024,
        "hard_negative_refresh_steps": 40,
        "alpha": alpha,
    })


def fit_model(
    name: str,
    tokens: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    selection: np.ndarray,
    alphabet_size: int,
    seed: int,
) -> ModelFit:
    functions = {
        "pairwise_ridge": fit_pairwise_ridge,
        "extra_trees": fit_extra_trees,
        "mlp": fit_mlp,
        "ranking": fit_ranking_proxy,
    }
    if name == "com":
        return fit_com_proxy(tokens, target, train, validation, selection, alphabet_size, seed)
    return functions[name](tokens, target, train, validation, alphabet_size, seed)
