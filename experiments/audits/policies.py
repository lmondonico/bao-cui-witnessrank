"""Probe orderings and witness-rank computations on finite candidate pools."""

from __future__ import annotations

import numpy as np


TOLERANCE = 1e-12


def ranked_order(scores: np.ndarray, pool: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
    local = np.lexsort((candidate_ids[pool], -scores[pool]))
    return pool[local]


def selection_lookup(tokens: np.ndarray, pool: np.ndarray) -> dict[bytes, int]:
    return {tokens[index].tobytes(): int(index) for index in pool}


def sequence_search(
    scores: np.ndarray,
    tokens: np.ndarray,
    pool: np.ndarray,
    lookup: dict[bytes, int],
    alphabet_size: int,
    budget: int,
    seed: int,
) -> int:
    rng = np.random.default_rng(seed)
    current: int | None = None
    best: int | None = None
    for evaluation in range(budget):
        restart = current is None or evaluation % 32 == 0
        if restart:
            proposal = int(rng.choice(pool))
        else:
            proposal = _mutant_in_pool(tokens[current], lookup, pool, alphabet_size, rng)
        if restart or scores[proposal] >= scores[current] - TOLERANCE:
            current = proposal
        if best is None or scores[proposal] > scores[best] + TOLERANCE:
            best = proposal
        elif abs(scores[proposal] - scores[best]) <= TOLERANCE and proposal < best:
            best = proposal
    if best is None:
        raise AssertionError("search budget must be positive")
    return best


def _mutant_in_pool(
    sequence: np.ndarray,
    lookup: dict[bytes, int],
    pool: np.ndarray,
    alphabet_size: int,
    rng: np.random.Generator,
) -> int:
    for _ in range(32):
        proposal = sequence.copy()
        position = int(rng.integers(len(sequence)))
        token = int(rng.integers(alphabet_size - 1))
        proposal[position] = token + (token >= proposal[position])
        match = lookup.get(proposal.tobytes())
        if match is not None:
            return match
    return int(rng.choice(pool))


def _nomination_first(nomination: int, order: np.ndarray) -> np.ndarray:
    return np.concatenate(([nomination], order[order != nomination])).astype(np.int64)


def top_order(
    nomination: int,
    prediction: np.ndarray,
    pool: np.ndarray,
    candidate_ids: np.ndarray,
) -> np.ndarray:
    return _nomination_first(nomination, ranked_order(prediction, pool, candidate_ids))


def local_order(
    nomination: int,
    prediction: np.ndarray,
    tokens: np.ndarray,
    pool: np.ndarray,
    candidate_ids: np.ndarray,
) -> np.ndarray:
    distance = np.count_nonzero(tokens[pool] != tokens[nomination], axis=1)
    local = np.lexsort((candidate_ids[pool], -prediction[pool], distance))
    return _nomination_first(nomination, pool[local])


def random_order(nomination: int, pool: np.ndarray, seed: int) -> np.ndarray:
    remaining = pool[pool != nomination]
    return np.concatenate(([nomination], np.random.default_rng(seed).permutation(remaining)))


def disagreement_order(
    nomination: int,
    prediction: np.ndarray,
    other_prediction: np.ndarray,
    pool: np.ndarray,
    candidate_ids: np.ndarray,
) -> np.ndarray:
    disagreement = np.abs(prediction[pool] - other_prediction[pool])
    local = np.lexsort((candidate_ids[pool], -prediction[pool], -disagreement))
    return _nomination_first(nomination, pool[local])


def diversity_order(
    nomination: int,
    prediction: np.ndarray,
    tokens: np.ndarray,
    pool: np.ndarray,
    candidate_ids: np.ndarray,
    maximum_budget: int,
) -> np.ndarray:
    proxy_order = ranked_order(prediction, pool, candidate_ids)
    shortlist_size = min(len(proxy_order), max(4096, 16 * maximum_budget))
    shortlist = proxy_order[:shortlist_size]
    selected = [nomination]
    available = shortlist[shortlist != nomination]
    minimum_distance = np.count_nonzero(tokens[available] != tokens[nomination], axis=1)
    while len(selected) < maximum_budget and len(available):
        local = np.lexsort((candidate_ids[available], -prediction[available], -minimum_distance))
        choice_position = int(local[0])
        choice = int(available[choice_position])
        selected.append(choice)
        available = np.delete(available, choice_position)
        minimum_distance = np.delete(minimum_distance, choice_position)
        if len(available):
            distance = np.count_nonzero(tokens[available] != tokens[choice], axis=1)
            minimum_distance = np.minimum(minimum_distance, distance)
    remainder = proxy_order[~np.isin(proxy_order, np.asarray(selected), assume_unique=False)]
    return np.concatenate((np.asarray(selected, dtype=np.int64), remainder))


def audit_gaps(
    reference: np.ndarray,
    prediction: np.ndarray,
    pool: np.ndarray,
    candidate_ids: np.ndarray,
    approximate: int,
    hard: int,
) -> dict[str, object]:
    proxy_order = ranked_order(prediction, pool, candidate_ids)
    true_order = ranked_order(reference, pool, candidate_ids)
    selected = int(proxy_order[0])
    true_best = int(true_order[0])
    epsilon = float(np.max(np.abs(reference[pool] - prediction[pool])))
    better = reference[proxy_order] > reference[selected] + TOLERANCE
    witness_positions = np.flatnonzero(better)
    witness_rank = int(witness_positions[0] + 1) if len(witness_positions) else None
    threshold = prediction[selected] - 2.0 * epsilon
    n_2eps = int(np.count_nonzero(prediction[pool] >= threshold - TOLERANCE))
    g_search = float(prediction[selected] - prediction[approximate])
    g_verifier = float(reference[true_best] - reference[selected])
    g_total = float(reference[true_best] - reference[approximate])
    harm = float(reference[approximate] - reference[selected])
    g_hard = float(reference[true_best] - reference[hard])
    if g_verifier > TOLERANCE:
        if witness_rank is None or witness_rank > n_2eps:
            raise AssertionError("witness-rank certification-budget theorem failed")
    elif witness_rank is not None:
        raise AssertionError("witness exists when verifier gap is zero")
    return {
        "x_true_index": true_best,
        "x_sel_index": selected,
        "x_approx_index": int(approximate),
        "x_hard_index": int(hard),
        "g_hard": g_hard,
        "g_verifier": g_verifier,
        "g_search": g_search,
        "g_total": g_total,
        "H": harm,
        "eps_inf": epsilon,
        "witness_rank_R": witness_rank,
        "n_2eps": n_2eps,
        "n_2eps_over_R": n_2eps / witness_rank if witness_rank else None,
        "r_le_n_2eps": witness_rank <= n_2eps if witness_rank else None,
        "gap_identity_error": abs(g_total - (g_verifier - harm)),
    }


def certificate_rows(
    dataset: str,
    model: str,
    seed: int,
    reference: np.ndarray,
    nomination: int,
    order: np.ndarray,
    policy: str,
    budgets: list[int],
    g_verifier: float,
    witness_rank: int | None,
    repetition: int = 0,
) -> list[dict[str, object]]:
    improvements = reference[order] > reference[nomination] + TOLERANCE
    first = np.flatnonzero(improvements)
    first_positive = int(first[0] + 1) if len(first) else None
    rows: list[dict[str, object]] = []
    for budget in budgets:
        actual = min(int(budget), len(order))
        lower = max(0.0, float(np.max(reference[order[:actual]]) - reference[nomination]))
        if lower < -TOLERANCE or lower > g_verifier + 1e-10:
            raise AssertionError("finite noiseless certificate violated its one-sided guarantee")
        certified = lower > TOLERANCE
        row = {
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "policy": policy,
            "repetition": repetition,
            "budget": int(budget),
            "actual_probe_count": actual,
            "g_verifier": g_verifier,
            "l_b": lower,
            "l_b_over_g_verifier": lower / g_verifier if g_verifier > TOLERANCE else None,
            "genuine_failure": g_verifier > TOLERANCE,
            "certified_failure": certified,
            "trusted_evaluations_to_first_positive": first_positive if first_positive is not None and first_positive <= actual else None,
            "trusted_evaluations_to_first_positive_full_order": first_positive,
            "witness_rank_R": witness_rank,
        }
        if policy == "top_b" and g_verifier > TOLERANCE:
            expected = bool(witness_rank is not None and witness_rank <= budget)
            row["top_b_recall_equals_r_cdf"] = bool(certified == expected)
            if certified != expected:
                raise AssertionError("Top-B certificate does not equal witness-rank CDF indicator")
        else:
            row["top_b_recall_equals_r_cdf"] = None
        rows.append(row)
    return rows
