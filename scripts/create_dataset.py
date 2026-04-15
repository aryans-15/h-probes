#!/usr/bin/env python3
"""Generate node-to-node traversal prompts for complete binary trees.

This script builds JSONL examples where the task is to output the shortest path
between two nodes in a complete binary tree labeled in breadth-first order.
Each record includes the prompt (with an ASCII tree), the ground-truth path,
and metadata such as depth and sampling configuration. Sampling is applied
globally across depths.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable, List, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cutter.utils.shared import paths as path_utils
from cutter.utils.shared.paths import dataset_tag, gsm8k_dataset_tag, parse_gsm8k_dataset_tag, parse_tree_dataset_tag
from cutter.utils.math.datasets import build_gsm8k_records
from cutter.utils.shared.basic import parse_float_range_arg, parse_range_arg

# -----------------------------------------------------------------------------
# Core data structures


@dataclass
class TraversalExample:
    depth: int
    source: int
    target: int
    waypoints: List[int]
    path: List[int]
    prompt: str
    num_samples: int
    num_steps: int
    label_mapping: List[int]
    canonical_source: int
    canonical_target: int
    canonical_waypoints: List[int]
    canonical_path: List[int]
    sparsity_sampled: float
    num_nodes: int
    max_possible_nodes: int
    is_sparse: bool

    def to_json(self) -> str:
        record = {
            "depth": self.depth,
            "source": self.source,
            "target": self.target,
            "waypoints": self.waypoints,
            "path": self.path,
            "path_text": " ".join(str(node) for node in self.path),
            "prompt": self.prompt,
            "num_samples": self.num_samples,
            "num_steps": self.num_steps,
            "label_mapping": self.label_mapping,
            "canonical_source": self.canonical_source,
            "canonical_target": self.canonical_target,
            "canonical_waypoints": self.canonical_waypoints,
            "canonical_path": self.canonical_path,
            "canonical_path_text": " ".join(str(node) for node in self.canonical_path),
            "sparsity_sampled": self.sparsity_sampled,
            "num_nodes": self.num_nodes,
            "max_possible_nodes": self.max_possible_nodes,
            "is_sparse": self.is_sparse,
        }
        return json.dumps(record, ensure_ascii=True)


# -----------------------------------------------------------------------------
# Tree helpers


def node_count_for_depth(depth: int) -> int:
    """Number of nodes in a full binary tree for a given depth."""

    return (1 << (depth + 1)) - 1


def build_tree_diagram(
    depth: int,
    label_mapping: Sequence[int] | None = None,
    included_nodes: Set[int] | None = None,
) -> str:
    """Return a simple ASCII diagram for a complete binary tree.

    If label_mapping is provided, it should map each structural node id to a
    shuffled label to render in the diagram.
    """

    nodes_by_depth: List[List[str]] = []
    cursor = 0
    for level in range(depth + 1):
        level_count = 1 << level
        labels: List[str] = []
        for idx in range(cursor, cursor + level_count):
            if included_nodes is not None and idx not in included_nodes:
                labels.append("")
                continue
            if label_mapping is None:
                labels.append(str(idx))
                continue
            mapped = int(label_mapping[idx])
            labels.append("" if mapped < 0 else str(mapped))
        nodes_by_depth.append(labels)
        cursor += level_count

    max_label_len = max((len(label) for level in nodes_by_depth for label in level if label), default=1)
    unit = max(3, max_label_len + 2)
    width = (1 << (depth + 1)) * unit

    lines = ["Tree Graph:"]
    for level, labels in enumerate(nodes_by_depth):
        line = [" "] * width
        slots = 1 << level
        segment = width // slots
        center_offset = segment // 2
        for idx, label in enumerate(labels):
            if not label:
                continue
            center = idx * segment + center_offset
            start_pos = int(max(0, center - len(label) // 2))
            end_pos = min(width, start_pos + len(label))
            if end_pos - start_pos < len(label):
                start_pos = max(0, end_pos - len(label))
                end_pos = start_pos + len(label)
            for offset, ch in enumerate(label):
                pos = start_pos + offset
                if 0 <= pos < width:
                    line[pos] = ch
        lines.append("".join(line).rstrip())
    return "\n".join(lines)


def ancestors_to_root(node: int) -> List[int]:
    """Return node plus its ancestors up to the root."""

    result: List[int] = []
    current = node
    while True:
        result.append(current)
        if current == 0:
            break
        current = (current - 1) // 2
    return result


def shortest_path(source: int, target: int) -> List[int]:
    """Compute the unique shortest path between two nodes in a binary tree."""

    if source == target:
        raise ValueError("source and target must differ")

    ancestors_source = ancestors_to_root(source)
    ancestors_target = ancestors_to_root(target)
    target_set = set(ancestors_source)

    lca = None
    for node in ancestors_target:
        if node in target_set:
            lca = node
            break
    if lca is None:
        raise RuntimeError("No common ancestor found; tree construction is invalid.")

    to_lca = ancestors_source[: ancestors_source.index(lca) + 1]
    to_target = ancestors_target[: ancestors_target.index(lca)]
    full_path = to_lca + list(reversed(to_target))
    return full_path


def target_node_count_for_sparsity(depth: int, sparsity: float) -> int:
    """Target node count for a sampled sparsity, clamped to a feasible depth-d tree."""

    full_nodes = node_count_for_depth(depth)
    raw_target = int(math.ceil(sparsity * full_nodes))
    min_for_depth = min(full_nodes, depth + 1)
    return max(min_for_depth, min(full_nodes, raw_target))


def sample_sparse_node_ids(depth: int, target_nodes: int, rng: Random) -> List[int]:
    """Sample an ancestor-closed node set with exact size and max depth exactly `depth`."""

    full_nodes = node_count_for_depth(depth)
    if target_nodes < 1 or target_nodes > full_nodes:
        raise ValueError(f"target_nodes must be in [1, {full_nodes}] for depth={depth}")
    if depth == 0:
        return [0]

    nodes: Set[int] = {0}
    frontier: Set[int] = set()

    def add_children(parent: int) -> None:
        level = int(math.floor(math.log2(parent + 1)))
        if level >= depth:
            return
        left = 2 * parent + 1
        right = left + 1
        if left < full_nodes and left not in nodes:
            frontier.add(left)
        if right < full_nodes and right not in nodes:
            frontier.add(right)

    # Force at least one node at depth exactly `depth`.
    first_leaf = (1 << depth) - 1
    last_leaf = (1 << (depth + 1)) - 2
    forced_leaf = rng.randint(first_leaf, last_leaf)
    current = forced_leaf
    forced_path: List[int] = []
    while True:
        forced_path.append(current)
        if current == 0:
            break
        current = (current - 1) // 2
    for nid in reversed(forced_path):
        nodes.add(nid)

    for nid in list(nodes):
        add_children(nid)
    for nid in nodes:
        frontier.discard(nid)

    while len(nodes) < target_nodes:
        if not frontier:
            raise RuntimeError(f"Unable to expand sparse tree to {target_nodes} nodes at depth {depth}")
        candidate = rng.choice(tuple(frontier))
        frontier.remove(candidate)
        nodes.add(candidate)
        add_children(candidate)
        frontier.discard(candidate)

    if len(nodes) != target_nodes:
        raise RuntimeError(f"Sparse tree size mismatch: expected {target_nodes}, got {len(nodes)}")
    return sorted(nodes)


def sample_waypoint_sequence(node_ids: Sequence[int], num_steps: int, rng: Random) -> List[int]:
    """Sample waypoints from available nodes without adjacent repeats."""

    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if len(node_ids) < 2:
        raise ValueError("Need at least two nodes to sample waypoint transitions")

    waypoints = [rng.choice(node_ids)]
    for _ in range(num_steps):
        next_node = rng.choice(node_ids)
        while next_node == waypoints[-1]:
            next_node = rng.choice(node_ids)
        waypoints.append(next_node)
    return waypoints


# -----------------------------------------------------------------------------
# Example construction


def all_pairs(depth: int) -> List[tuple[int, int]]:
    """Enumerate ordered source/target pairs for the given depth."""

    n_nodes = node_count_for_depth(depth)
    pairs: List[tuple[int, int]] = []
    for src in range(n_nodes):
        for dst in range(n_nodes):
            if src == dst:
                continue
            pairs.append((src, dst))
    return pairs


def all_waypoint_sequences(depth: int, num_steps: int) -> List[tuple[int, ...]]:
    """Enumerate waypoint sequences with no adjacent repeats."""

    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    n_nodes = node_count_for_depth(depth)
    sequences: List[tuple[int, ...]] = []

    def backtrack(prefix: List[int]) -> None:
        if len(prefix) == num_steps + 1:
            sequences.append(tuple(prefix))
            return
        for node in range(n_nodes):
            if prefix and node == prefix[-1]:
                continue
            prefix.append(node)
            backtrack(prefix)
            prefix.pop()

    backtrack([])
    return sequences


def build_prompt(
    depth: int,
    waypoints: Sequence[int],
    tree_diagram: str,
    *,
    is_sparse: bool,
    num_nodes: int,
    max_possible_nodes: int,
    sparsity_sampled: float,
) -> str:
    """Format the prompt for a single traversal task."""

    waypoint_text = " to ".join(str(node) for node in waypoints)
    if is_sparse:
        tree_header = (
            "You are a path-finding assistant for a sparse, possibly unbalanced binary tree.\n"
            f"Tree max depth: {depth} (root depth 0).\n"
            f"Tree sparsity: {sparsity_sampled:.4f} ({num_nodes}/{max_possible_nodes} nodes retained).\n"
        )
    else:
        tree_header = (
            "You are a path-finding assistant for a complete binary tree.\n"
            f"Tree depth: {depth} (root depth 0; leaves depth {depth}).\n"
        )
    return (
        tree_header +
        "Nodes are labeled uniquely as shown in the diagram (root at the top, then left to right by level).\n\n"
        f"{tree_diagram}\n\n"
        f"Task: Provide the shortest path from node {waypoint_text}, moving only along tree edges.\n"
        "Guidelines:\n"
        "- You may reason step by step, but end with a single line that contains only the path.\n"
        "- Format the final line as `PATH: n0 n1 ... nk` using space-separated node ids.\n"
        "- Use only node numbers from the tree diagram and do not add extra text after the final line.\n"
        "Example format (depth 1): PATH: 1 0 2\n"
    )


def _select_candidates(
    candidates: List[tuple[int, tuple[int, ...]]],
    target_size: int,
    rng: Random,
    step_label: int,
) -> List[tuple[int, tuple[int, ...]]]:
    total_possible = len(candidates)
    if target_size <= 0:
        return []
    if target_size > total_possible:
        print(
            f"Requested num-samples={target_size} exceeds unique pairs {total_possible} "
            f"for num-steps={step_label}; sampling in reshuffled rounds with replacement across rounds."
        )

    selected: List[tuple[int, tuple[int, ...]]] = []
    while len(selected) < target_size:
        pool = list(candidates)
        rng.shuffle(pool)
        remaining = target_size - len(selected)
        selected.extend(pool[:remaining])
    return selected


def _targets_by_step(steps: Sequence[int], num_samples: int, candidates_by_step: dict[int, List[tuple[int, tuple[int, ...]]]] | None = None) -> Tuple[dict[int, int], int]:
    """Compute per-step sampling targets and total expected examples."""

    num_steps_values = len(steps)
    if num_steps_values == 0:
        return {}, 0

    if num_samples <= 0:
        if candidates_by_step is None:
            raise ValueError("num_samples=0 is only supported when candidate sets are explicitly enumerable")
        per_step_target = min(len(candidates) for candidates in candidates_by_step.values())
        targets = {step: per_step_target for step in steps}
        total_target = per_step_target * num_steps_values
    else:
        base = int(math.floor(num_samples / num_steps_values))
        remainder = int(num_samples % num_steps_values)
        targets = {
            step: base + (1 if idx < remainder else 0)
            for idx, step in enumerate(steps)
        }
        total_target = num_samples
    return targets, total_target


def _weighted_depth_choice(depths: Sequence[int], num_steps: int, rng: Random) -> int:
    """Sample depths with weights matching full-tree waypoint-sequence counts."""

    weights = []
    for depth in depths:
        n_nodes = node_count_for_depth(depth)
        if n_nodes < 2:
            weights.append(0)
            continue
        weights.append(n_nodes * ((n_nodes - 1) ** num_steps))
    if not any(weights):
        raise ValueError("No valid depth has at least two nodes for waypoint sampling")
    return int(rng.choices(list(depths), weights=weights, k=1)[0])


def generate_examples(
    depths: Iterable[int],
    num_samples: int,
    seed: int,
    steps_range: Sequence[int],
    sparsity_range: Tuple[float, float] = (1.0, 1.0),
) -> List[TraversalExample]:
    """Construct traversal examples for the requested depths and steps."""

    rng = Random(seed)
    steps = list(steps_range)
    if not steps:
        return []
    depths_list = list(depths)
    if not depths_list:
        return []

    sparsity_min, sparsity_max = sparsity_range
    use_sparse = not (math.isclose(sparsity_min, 1.0) and math.isclose(sparsity_max, 1.0))

    if use_sparse:
        targets, total_target = _targets_by_step(steps, num_samples, candidates_by_step=None)
        examples: List[TraversalExample] = []
        for step in steps:
            for _ in range(targets[step]):
                depth = _weighted_depth_choice(depths_list, step, rng)
                max_possible_nodes = node_count_for_depth(depth)
                sampled_sparsity = rng.uniform(sparsity_min, sparsity_max)
                num_nodes = target_node_count_for_sparsity(depth, sampled_sparsity)
                canonical_nodes = sample_sparse_node_ids(depth, num_nodes, rng)
                canonical_set = set(canonical_nodes)

                label_values = list(range(num_nodes))
                rng.shuffle(label_values)
                label_mapping = [-1] * max_possible_nodes
                for node_id, label in zip(canonical_nodes, label_values):
                    label_mapping[node_id] = label

                canonical_waypoints = sample_waypoint_sequence(canonical_nodes, step, rng)
                mapped_waypoints = [label_mapping[node] for node in canonical_waypoints]

                canonical_path: List[int] = []
                for idx in range(len(canonical_waypoints) - 1):
                    segment = shortest_path(canonical_waypoints[idx], canonical_waypoints[idx + 1])
                    if idx > 0:
                        segment = segment[1:]
                    for node in segment:
                        if node not in canonical_set:
                            raise RuntimeError("Generated path exits sparse tree; ancestor-closure invariant violated")
                    canonical_path.extend(segment)
                mapped_path = [label_mapping[node] for node in canonical_path]

                tree_diagram = build_tree_diagram(
                    depth,
                    label_mapping=label_mapping,
                    included_nodes=canonical_set,
                )
                prompt = build_prompt(
                    depth,
                    mapped_waypoints,
                    tree_diagram,
                    is_sparse=True,
                    num_nodes=num_nodes,
                    max_possible_nodes=max_possible_nodes,
                    sparsity_sampled=sampled_sparsity,
                )
                examples.append(
                    TraversalExample(
                        depth=depth,
                        source=mapped_waypoints[0],
                        target=mapped_waypoints[-1],
                        waypoints=mapped_waypoints,
                        path=mapped_path,
                        prompt=prompt,
                        num_samples=total_target,
                        num_steps=step,
                        label_mapping=label_mapping,
                        canonical_source=canonical_waypoints[0],
                        canonical_target=canonical_waypoints[-1],
                        canonical_waypoints=list(canonical_waypoints),
                        canonical_path=canonical_path,
                        sparsity_sampled=sampled_sparsity,
                        num_nodes=num_nodes,
                        max_possible_nodes=max_possible_nodes,
                        is_sparse=True,
                    )
                )
        return examples

    candidates_by_step: dict[int, List[tuple[int, tuple[int, ...]]]] = {}
    for num_steps in steps:
        candidates: List[tuple[int, tuple[int, ...]]] = []
        for depth in depths_list:
            waypoints = all_waypoint_sequences(depth, num_steps)
            for seq in waypoints:
                candidates.append((depth, seq))
        candidates_by_step[num_steps] = candidates

    targets, total_target = _targets_by_step(steps, num_samples, candidates_by_step=candidates_by_step)

    selected_by_step: dict[int, List[tuple[int, tuple[int, ...]]]] = {}
    for step in steps:
        selected_by_step[step] = _select_candidates(
            candidates_by_step[step],
            targets[step],
            rng,
            step_label=step,
        )

    examples: List[TraversalExample] = []
    for step in steps:
        for depth, waypoints in selected_by_step[step]:
            n_nodes = node_count_for_depth(depth)
            label_mapping = rng.sample(range(n_nodes), n_nodes)
            mapped_waypoints = [label_mapping[node] for node in waypoints]

            structural_path: List[int] = []
            for idx in range(len(waypoints) - 1):
                segment = shortest_path(waypoints[idx], waypoints[idx + 1])
                if idx > 0:
                    segment = segment[1:]
                structural_path.extend(segment)
            mapped_path = [label_mapping[node] for node in structural_path]
            tree_diagram = build_tree_diagram(depth, label_mapping=label_mapping)
            prompt = build_prompt(
                depth,
                mapped_waypoints,
                tree_diagram,
                is_sparse=False,
                num_nodes=n_nodes,
                max_possible_nodes=n_nodes,
                sparsity_sampled=1.0,
            )
            examples.append(
                TraversalExample(
                    depth=depth,
                    source=mapped_waypoints[0],
                    target=mapped_waypoints[-1],
                    waypoints=mapped_waypoints,
                    path=mapped_path,
                    prompt=prompt,
                    num_samples=total_target,
                    num_steps=step,
                    label_mapping=list(label_mapping),
                    canonical_source=waypoints[0],
                    canonical_target=waypoints[-1],
                    canonical_waypoints=list(waypoints),
                    canonical_path=structural_path,
                    sparsity_sampled=1.0,
                    num_nodes=n_nodes,
                    max_possible_nodes=n_nodes,
                    is_sparse=False,
                )
            )
    return examples


# -----------------------------------------------------------------------------
# CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate datasets for tree traversal or GSM8K math problems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--setting",
        choices=("tree", "math"),
        default="tree",
        help="Which setting to generate: tree traversal or GSM8K math.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset folder tag to create (e.g., depth1-2_n1000_steps1-2_spars0.5-1 or gsm8k_all_nall_seed0).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument(
        "--depth-range",
        type=int,
        nargs="+",
        default=[1, 2],
        help="One or two integers: depth or [min max] depth range (inclusive).",
    )
    parser.add_argument(
        "--steps-range",
        type=int,
        nargs="+",
        default=[1, 2],
        help="One or two integers: steps or [min max] steps range (inclusive).",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        nargs="+",
        default=[1.0],
        help="One or two floats: sparsity or [min max] sparsity range in (0, 1].",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of examples to sample across all depths (0 = sample uniformly across steps).",
    )
    parser.add_argument(
        "--gsm8k-split",
        type=str,
        default="all",
        help="GSM8K split to use for math setting: train, test, or all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.setting == "math":
        if args.dataset:
            split, num_samples, seed = parse_gsm8k_dataset_tag(args.dataset)
            tag = args.dataset
        else:
            split, num_samples, seed = args.gsm8k_split, args.num_samples, args.seed
            tag = gsm8k_dataset_tag(num_samples, seed, split)
        records = build_gsm8k_records(
            num_samples=num_samples,
            seed=seed,
            split=split,
        )
        output_path = path_utils.gsm8k_dataset_path_from_tag(tag)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(record.to_json() + "\n")
        print(f"Wrote {len(records)} examples to {output_path} (tag: {tag})")
        return

    if args.dataset:
        min_depth, max_depth, num_samples, steps_range, sparsity_range = parse_tree_dataset_tag(args.dataset)
        min_steps, max_steps = steps_range
        tag = args.dataset
    else:
        min_depth, max_depth = parse_range_arg(args.depth_range, "depth-range", min_value=0)
        min_steps, max_steps = parse_range_arg(args.steps_range, "steps-range", min_value=1)
        sparsity_range = parse_float_range_arg(args.sparsity, "sparsity", min_value=0.0, max_value=1.0, min_inclusive=False)
        steps_tag = (min_steps, max_steps) if min_steps != max_steps else min_steps
        num_samples = args.num_samples
        tag = dataset_tag(min_depth, max_depth, num_samples, steps_tag, sparsity=sparsity_range)
    depths = list(range(min_depth, max_depth + 1))
    steps = list(range(min_steps, max_steps + 1))

    examples = generate_examples(
        depths=depths,
        num_samples=num_samples,
        seed=args.seed,
        steps_range=steps,
        sparsity_range=sparsity_range,
    )

    # Store dataset under a tag-specific folder so downstream steps can infer paths from dataset/model ids.
    output_path = path_utils.dataset_path_from_tag(tag)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(example.to_json() + "\n")
    print(f"Wrote {len(examples)} examples to {output_path} (tag: {tag})")


if __name__ == "__main__":
    main()
