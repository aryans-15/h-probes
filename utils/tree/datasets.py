"""Traversal dataset loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class TraversalRecord:
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

    @classmethod
    def from_json(cls, row: Dict[str, Any]) -> "TraversalRecord":
        path_vals = [int(x) for x in row["path"]]
        depth_val = int(row["depth"])
        max_nodes = int(row.get("max_possible_nodes", (1 << (depth_val + 1)) - 1))
        mapping_vals = [int(x) for x in row.get("label_mapping", [])]
        inferred_num_nodes = (
            sum(1 for x in mapping_vals if x >= 0)
            if mapping_vals
            else max_nodes
        )
        return cls(
            depth=depth_val,
            source=int(row["source"]),
            target=int(row["target"]),
            waypoints=[int(x) for x in row.get("waypoints", [row["source"], row["target"]])],
            path=path_vals,
            prompt=str(row["prompt"]),
            num_samples=int(row.get("num_samples", row.get("sample_rate", -1))),
            num_steps=int(row.get("num_steps", 1)),
            label_mapping=mapping_vals,
            canonical_source=int(row.get("canonical_source", row.get("source", -1))),
            canonical_target=int(row.get("canonical_target", row.get("target", -1))),
            canonical_waypoints=[int(x) for x in row.get("canonical_waypoints", row.get("waypoints", []))],
            canonical_path=[int(x) for x in row.get("canonical_path", path_vals)],
            sparsity_sampled=float(row.get("sparsity_sampled", 1.0)),
            num_nodes=int(row.get("num_nodes", inferred_num_nodes)),
            max_possible_nodes=max_nodes,
            is_sparse=bool(row.get("is_sparse", False)),
        )


def load_traversal_dataset(path: Path) -> List[TraversalRecord]:
    rows: List[TraversalRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(TraversalRecord.from_json(json.loads(line)))
    return rows
