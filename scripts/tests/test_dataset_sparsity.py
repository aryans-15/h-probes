from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cutter.scripts.create_dataset import generate_examples
from cutter.utils.shared.paths import dataset_tag, parse_tree_dataset_tag


def _node_depth(node_id: int) -> int:
    return int(math.floor(math.log2(node_id + 1))) if node_id >= 0 else 0


def test_sparse_dataset_tag_roundtrip() -> None:
    tag = dataset_tag(1, 3, 1000, (1, 2), sparsity=(0.5, 1.0))
    assert tag == "depth1-3_n1000_steps1-2_spars0.5-1"

    dmin, dmax, nsamp, steps, sparsity = parse_tree_dataset_tag(tag)
    assert (dmin, dmax, nsamp) == (1, 3, 1000)
    assert steps == (1, 2)
    assert sparsity == (0.5, 1.0)


def test_tree_dataset_tag_backward_compatible_defaults() -> None:
    tag = "depth1-2_n1000_steps1-2"
    dmin, dmax, nsamp, steps, sparsity = parse_tree_dataset_tag(tag)
    assert (dmin, dmax, nsamp) == (1, 2, 1000)
    assert steps == (1, 2)
    assert sparsity == (1.0, 1.0)

    rebuilt = dataset_tag(1, 2, 1000, (1, 2), sparsity=(1.0, 1.0))
    assert rebuilt == tag


def test_generate_sparse_examples_invariants() -> None:
    examples = generate_examples(
        depths=[3, 4],
        num_samples=16,
        seed=7,
        steps_range=[1],
        sparsity_range=(0.5, 1.0),
    )
    assert len(examples) == 16

    for ex in examples:
        full_nodes = (1 << (ex.depth + 1)) - 1
        expected_nodes = max(min(full_nodes, ex.depth + 1), int(math.ceil(ex.sparsity_sampled * full_nodes)))

        assert ex.is_sparse is True
        assert 0.5 <= ex.sparsity_sampled <= 1.0
        assert ex.max_possible_nodes == full_nodes
        assert ex.num_nodes == expected_nodes

        assert len(ex.label_mapping) == full_nodes
        active_nodes = [idx for idx, label in enumerate(ex.label_mapping) if label >= 0]
        active_labels = [label for label in ex.label_mapping if label >= 0]
        active_set = set(active_nodes)

        assert len(active_nodes) == ex.num_nodes
        assert len(set(active_labels)) == ex.num_nodes
        assert max(_node_depth(nid) for nid in active_nodes) == ex.depth

        for node_id in active_nodes:
            current = node_id
            while current > 0:
                current = (current - 1) // 2
                assert current in active_set

        assert len(ex.canonical_waypoints) == ex.num_steps + 1
        for i in range(len(ex.canonical_waypoints) - 1):
            assert ex.canonical_waypoints[i] != ex.canonical_waypoints[i + 1]

        for node_id in ex.canonical_path:
            assert node_id in active_set
