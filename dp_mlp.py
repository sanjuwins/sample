"""Data-parallel XOR MLP matching the Assignment 1 implementation.

Run with, for example:
    mpiexec -n 4 python dp_mlp.py
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Parameters = tuple[FloatArray, FloatArray, FloatArray, FloatArray]
Gradients = Parameters


def make_xor_blobs(
    n: int = 400, noise: float = 0.2, seed: int = 42
) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    labels = np.array([0, 1, 1, 0], dtype=np.float64)
    indices = rng.integers(0, 4, size=n)
    features = centers[indices] + rng.normal(scale=noise, size=(n, 2))
    targets = labels[indices].reshape(-1, 1)
    return features, targets


def init_params(
    n_features: int, n_hidden: int, seed: int
) -> Parameters:
    rng = np.random.default_rng(seed)
    limit1 = np.sqrt(6.0 / (n_features + n_hidden))
    w1 = rng.uniform(-limit1, limit1, size=(n_features, n_hidden))
    b1 = np.zeros((1, n_hidden), dtype=np.float64)
    limit2 = np.sqrt(6.0 / (n_hidden + 1))
    w2 = rng.uniform(-limit2, limit2, size=(n_hidden, 1))
    b2 = np.zeros((1, 1), dtype=np.float64)
    return w1, b1, w2, b2


def forward(
    features: FloatArray, params: Parameters
) -> tuple[FloatArray, tuple[FloatArray, FloatArray]]:
    w1, b1, w2, b2 = params
    z1 = features @ w1 + b1
    a1 = np.tanh(z1)
    z2 = a1 @ w2 + b2
    predictions = 1.0 / (1.0 + np.exp(-np.clip(z2, -500.0, 500.0)))
    return predictions, (z1, a1)


def bce_loss(predictions: FloatArray, targets: FloatArray) -> float:
    clipped = np.clip(predictions, 1e-7, 1.0 - 1e-7)
    losses = -(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
    return float(np.mean(losses))


def backward(
    features: FloatArray,
    targets: FloatArray,
    predictions: FloatArray,
    cache: tuple[FloatArray, FloatArray],
    params: Parameters,
) -> Gradients:
    w2 = params[2]
    z1, a1 = cache
    batch_size = features.shape[0]
    dz2 = predictions - targets
    dw2 = a1.T @ dz2 / batch_size
    db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size
    dz1 = (dz2 @ w2.T) * (1.0 - np.tanh(z1) ** 2)
    dw1 = features.T @ dz1 / batch_size
    db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size
    return dw1, db1, dw2, db2


def broadcast_params(
    rank: int, n_features: int, n_hidden: int, seed: int, comm: MPI.Comm
) -> Parameters:
    if rank == 0:
        params = init_params(n_features, n_hidden, seed)
    else:
        params = (
            np.empty((n_features, n_hidden), dtype=np.float64),
            np.empty((1, n_hidden), dtype=np.float64),
            np.empty((n_hidden, 1), dtype=np.float64),
            np.empty((1, 1), dtype=np.float64),
        )
    for parameter in params:
        comm.Bcast(parameter, root=0)
    return params


def train_distributed(
    features: FloatArray,
    targets: FloatArray,
    params: Parameters,
    epochs: int,
    global_batch_size: int,
    learning_rate: float,
    comm: MPI.Comm,
) -> tuple[Parameters, FloatArray]:
    rank = comm.Get_rank()
    process_count = comm.Get_size()
    local_batch_size = global_batch_size // process_count
    train_size = features.shape[0]
    batch_count = train_size // global_batch_size
    weights = tuple(parameter.copy() for parameter in params)
    epoch_losses = np.empty(epochs, dtype=np.float64)

    for epoch in range(epochs):
        total_epoch_loss = 0.0
        for batch_index in range(batch_count):
            batch_start = batch_index * global_batch_size
            rank_start = batch_start + rank * local_batch_size
            rank_end = rank_start + local_batch_size
            local_features = features[rank_start:rank_end]
            local_targets = targets[rank_start:rank_end]

            predictions, cache = forward(local_features, weights)
            local_loss = bce_loss(predictions, local_targets)
            gradients = backward(
                local_features, local_targets, predictions, cache, weights
            )

            for gradient in gradients:
                comm.Allreduce(MPI.IN_PLACE, gradient, op=MPI.SUM)
                gradient /= process_count

            weights = tuple(
                parameter - learning_rate * gradient
                for parameter, gradient in zip(weights, gradients)
            )

            batch_loss_sum = comm.allreduce(local_loss, op=MPI.SUM)
            total_epoch_loss += batch_loss_sum / process_count

        epoch_losses[epoch] = total_epoch_loss / batch_count

    return weights, epoch_losses


def train_full_batch_reference(
    features: FloatArray,
    targets: FloatArray,
    initial_params: Parameters,
    epochs: int,
    global_batch_size: int,
    learning_rate: float,
) -> FloatArray:
    weights = tuple(parameter.copy() for parameter in initial_params)
    train_size = features.shape[0]
    batch_count = train_size // global_batch_size
    epoch_losses = np.empty(epochs, dtype=np.float64)

    for epoch in range(epochs):
        total_epoch_loss = 0.0
        for batch_index in range(batch_count):
            batch_start = batch_index * global_batch_size
            batch_end = batch_start + global_batch_size
            batch_features = features[batch_start:batch_end]
            batch_targets = targets[batch_start:batch_end]
            predictions, cache = forward(batch_features, weights)
            total_epoch_loss += bce_loss(predictions, batch_targets)
            gradients = backward(
                batch_features, batch_targets, predictions, cache, weights
            )
            weights = tuple(
                parameter - learning_rate * gradient
                for parameter, gradient in zip(weights, gradients)
            )
        epoch_losses[epoch] = total_epoch_loss / batch_count

    return epoch_losses


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and verify an MPI data-parallel XOR MLP."
    )
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--weight-seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    process_count = comm.Get_size()

    if args.samples <= 0 or args.hidden <= 0 or args.epochs <= 0:
        raise ValueError("samples, hidden size, and epochs must be positive")
    if args.global_batch_size <= 0:
        raise ValueError("global batch size must be positive")
    if args.global_batch_size % process_count != 0:
        raise ValueError(
            "global batch size must be divisible by the MPI rank count"
        )
    if args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")

    features, targets = make_xor_blobs(
        n=args.samples, noise=args.noise, seed=args.data_seed
    )
    train_size = int(0.8 * args.samples)
    train_size -= train_size % args.global_batch_size
    if train_size == 0:
        raise ValueError(
            "training split must contain at least one full global batch"
        )
    train_features = np.ascontiguousarray(features[:train_size])
    train_targets = np.ascontiguousarray(targets[:train_size])

    initial_params = broadcast_params(
        rank, train_features.shape[1], args.hidden, args.weight_seed, comm
    )
    saved_initial_params = tuple(p.copy() for p in initial_params)
    trained_params, distributed_losses = train_distributed(
        train_features,
        train_targets,
        initial_params,
        args.epochs,
        args.global_batch_size,
        args.learning_rate,
        comm,
    )

    w1 = trained_params[0]
    gathered_w1 = (
        np.empty((process_count, *w1.shape), dtype=np.float64)
        if rank == 0
        else None
    )
    comm.Gather(w1, gathered_w1, root=0)

    if rank == 0:
        reference_losses = train_full_batch_reference(
            train_features,
            train_targets,
            saved_initial_params,
            args.epochs,
            args.global_batch_size,
            args.learning_rate,
        )
        assert gathered_w1 is not None
        replica_difference = float(
            np.max(np.abs(gathered_w1 - gathered_w1[0]))
        )
        loss_difference = float(
            np.max(np.abs(distributed_losses - reference_losses))
        )
        print(
            f"Ranks: {process_count}; training samples: {train_size}; "
            f"global batch: {args.global_batch_size}; "
            f"local batch per rank: {args.global_batch_size // process_count}"
        )
        print(
            "Claim (i), identical W1 replicas: max absolute difference = "
            f"{replica_difference:.17g}"
        )
        print(
            "Claim (ii), distributed vs full-batch per-epoch loss: "
            f"max absolute difference = {loss_difference:.17g}"
        )
        print(
            "Unequal-shard note: if shard sizes differ but each rank's mean "
            "gradient is still divided by the rank count, examples receive "
            "unequal weight and the update is not the true global-batch gradient."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
