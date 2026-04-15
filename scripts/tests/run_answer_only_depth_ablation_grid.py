#!/usr/bin/env python3
"""Run answer-only causal ablations for depth probe setups on tree tasks.

This script evaluates depth setup causal importance by ablating 1D directions
at a fixed transformer layer and measuring answer-token logit changes using
`cutter/scripts/intervene.py` internals.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# This script lives under cutter/scripts/tests/, one level deeper than scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import cutter.scripts.intervene as intervene
from cutter.scripts import evaluate_probe as eval_probe
from cutter.utils.tree.encoding import DEVICE, load_reasoning_model, set_global_seed
from cutter.utils.shared import paths as path_utils
from cutter.utils.shared.paths import embeddings_path, intervention_output_dir, model_tag, responses_path
from cutter.utils.shared.models import resolve_single_model_id
from cutter.utils.shared.basic import split_balanced
from cutter.utils.shared.embeddings_cache import load_embedding_payload


@dataclass
class ExampleLogitCache:
    example_id: int
    prompt_text: str
    prefix_text: str
    answer_text: str
    answer_token_ids: List[int]
    baseline_logits: torch.Tensor


def _invfreq_weights(values: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(values, return_counts=True)
    inv = {val: 1.0 / max(int(cnt), 1) for val, cnt in zip(unique, counts)}
    return np.asarray([inv[val] for val in values], dtype=np.float64)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _extract_linear_direction(model: Pipeline) -> np.ndarray:
    named = getattr(model, "named_steps", {})
    scaler = named.get("scale")
    reg = named.get("reg")
    if reg is None or not hasattr(reg, "coef_"):
        raise ValueError("Expected Pipeline with 'reg' step exposing coef_.")
    coef = np.asarray(reg.coef_, dtype=np.float32).reshape(-1)
    if scaler is not None and hasattr(scaler, "scale_"):
        scale = np.asarray(scaler.scale_, dtype=np.float32).reshape(-1)
        scale = np.where(scale == 0.0, 1.0, scale)
        coef = coef / scale
    return coef.astype(np.float32)


def _normalize_direction(direction: np.ndarray) -> np.ndarray:
    vec = np.asarray(direction, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError("Direction norm is zero.")
    return (vec / norm).astype(np.float32)


def _build_bucket(records: Iterable[Any], bucket: str) -> List[Any]:
    if bucket == "all":
        return list(records)
    selected: List[Any] = []
    for rec in records:
        if rec.exact_match:
            key = "exact"
        elif rec.partial_score > 0:
            key = "partial"
        else:
            key = "zero"
        if key == bucket:
            selected.append(rec)
    return selected


def _build_example_cache(
    records: Sequence[Any],
    tokenizer,
    model,
) -> List[ExampleLogitCache]:
    cache: List[ExampleLogitCache] = []
    for rec in records:
        prompt_text = intervene._render_prompt(tokenizer, rec.prompt)
        prefix_text = ""
        raw_response = str(getattr(rec, "raw_response", getattr(rec, "raw", "")))
        if rec.parsed_text:
            start_idx = raw_response.rfind(rec.parsed_text)
            if start_idx != -1:
                prefix_text = raw_response[:start_idx]
        answer_text = intervene._build_answer_text(rec.ground_truth_path)
        answer_token_ids = tokenizer.encode(answer_text, add_special_tokens=False)
        if not answer_token_ids:
            continue
        baseline_logits = intervene._compute_answer_logits(
            prompt_text=prompt_text,
            prefix_text=prefix_text,
            answer_text=answer_text,
            tokenizer=tokenizer,
            model=model,
            ablation_basis=None,
            target_layer=None,
        )
        cache.append(
            ExampleLogitCache(
                example_id=int(rec.example_id),
                prompt_text=prompt_text,
                prefix_text=prefix_text,
                answer_text=answer_text,
                answer_token_ids=list(answer_token_ids),
                baseline_logits=baseline_logits,
            )
        )
    return cache


def _mean_signed_gold_logit_drop(
    baseline_logits: torch.Tensor,
    variant_logits: torch.Tensor,
    answer_token_ids: Sequence[int],
) -> float:
    if baseline_logits.shape != variant_logits.shape:
        return float("nan")
    if not answer_token_ids:
        return float("nan")
    if baseline_logits.shape[0] != len(answer_token_ids):
        return float("nan")
    token_ids = torch.tensor(answer_token_ids, dtype=torch.long, device=baseline_logits.device)
    positions = torch.arange(len(answer_token_ids), device=baseline_logits.device)
    baseline_vals = baseline_logits[positions, token_ids]
    variant_vals = variant_logits[positions, token_ids]
    return float((baseline_vals - variant_vals).mean().item())


def _evaluate_basis_answer_only(
    basis_hidden: np.ndarray,
    *,
    example_cache: Sequence[ExampleLogitCache],
    target_layer: torch.nn.Module,
    tokenizer,
    model,
    model_dtype: torch.dtype,
) -> Dict[str, Any]:
    basis = _normalize_direction(basis_hidden).reshape(-1, 1)
    basis_t = torch.tensor(basis, dtype=torch.float32, device=model.device).to(dtype=model_dtype)

    abs_diffs: List[float] = []
    signed_drops: List[float] = []
    per_example: List[Dict[str, Any]] = []

    for item in example_cache:
        variant_logits = intervene._compute_answer_logits(
            prompt_text=item.prompt_text,
            prefix_text=item.prefix_text,
            answer_text=item.answer_text,
            tokenizer=tokenizer,
            model=model,
            ablation_basis=basis_t,
            target_layer=target_layer,
        )
        abs_diff = intervene._mean_abs_answer_logit_diff(
            item.baseline_logits,
            variant_logits,
            item.answer_token_ids,
        )
        signed_drop = _mean_signed_gold_logit_drop(
            item.baseline_logits,
            variant_logits,
            item.answer_token_ids,
        )
        abs_diffs.append(abs_diff)
        signed_drops.append(signed_drop)
        per_example.append(
            {
                "example_id": item.example_id,
                "abs_answer_logit_diff": abs_diff,
                "signed_gold_logit_drop": signed_drop,
            }
        )

    abs_arr = np.asarray(abs_diffs, dtype=float)
    signed_arr = np.asarray(signed_drops, dtype=float)
    return {
        "abs_answer_logit_diff_mean": float(np.nanmean(abs_arr)) if abs_arr.size else float("nan"),
        "abs_answer_logit_diff_std": float(np.nanstd(abs_arr, ddof=1)) if abs_arr.size > 1 else 0.0,
        "signed_gold_logit_drop_mean": float(np.nanmean(signed_arr)) if signed_arr.size else float("nan"),
        "signed_gold_logit_drop_std": float(np.nanstd(signed_arr, ddof=1)) if signed_arr.size > 1 else 0.0,
        "n_examples": int(len(per_example)),
        "per_example": per_example,
    }


def _train_method_direction_pc(
    *,
    method: str,
    Z: np.ndarray,
    depths: np.ndarray,
    train_idx: np.ndarray,
    example_ids: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    Z_train = Z[train_idx]
    y_train = depths[train_idx]
    sw = _invfreq_weights(y_train)

    diagnostics: Dict[str, Any] = {"method": method}

    if method == "ridge":
        ridge = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("reg", Ridge(alpha=1e-2)),
            ]
        )
        ridge.fit(Z_train, y_train, reg__sample_weight=sw)
        direction = _extract_linear_direction(ridge)
        y_hat_train = ridge.predict(Z_train).astype(np.float32)
        diagnostics["train_depth_corr"] = _safe_corr(y_train, y_hat_train)
        return direction, diagnostics

    if method == "logreg_depth_class":
        clf = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
            ]
        )
        y_class = y_train.astype(int)
        clf.fit(Z_train, y_class)
        proba = clf.predict_proba(Z_train)
        cls = clf.named_steps["lr"].classes_.astype(np.float32)
        y_hat_train = (proba @ cls).astype(np.float32)
    elif method == "mlp_64":
        mlp = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(64,),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=800,
                        early_stopping=True,
                        random_state=seed,
                    ),
                ),
            ]
        )
        mlp.fit(Z_train, y_train)
        y_hat_train = mlp.predict(Z_train).astype(np.float32)
    elif method == "rf_300":
        rf = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(Z_train, y_train)
        y_hat_train = rf.predict(Z_train).astype(np.float32)
    else:
        raise ValueError(f"Unsupported method '{method}'")

    diagnostics["train_depth_corr"] = _safe_corr(y_train, y_hat_train)

    # Distill non-linear/classifier predictions into a linear direction.
    surrogate = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("reg", Ridge(alpha=1e-2)),
        ]
    )
    surrogate.fit(Z_train, y_hat_train, reg__sample_weight=sw)
    surrogate_pred = surrogate.predict(Z_train).astype(np.float32)
    diagnostics["distill_corr_method_vs_surrogate"] = _safe_corr(y_hat_train, surrogate_pred)
    direction = _extract_linear_direction(surrogate)
    return direction, diagnostics


def _summarize_seed_blocks(
    seed_blocks: Sequence[Dict[str, Any]],
    *,
    methods: Sequence[str],
    pcs: Sequence[int],
) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    variants = list(methods) + ["top1pc_baseline", "random_baseline", "zero_baseline"]
    for pc in pcs:
        pc_key = str(pc)
        aggregate[pc_key] = {}
        for variant in variants:
            abs_vals: List[float] = []
            signed_vals: List[float] = []
            for sb in seed_blocks:
                pc_block = sb.get("pcs", {}).get(pc_key, {})
                var_block = pc_block.get("variants", {}).get(variant, {})
                abs_val = var_block.get("abs_answer_logit_diff_mean")
                signed_val = var_block.get("signed_gold_logit_drop_mean")
                if abs_val is not None and np.isfinite(abs_val):
                    abs_vals.append(float(abs_val))
                if signed_val is not None and np.isfinite(signed_val):
                    signed_vals.append(float(signed_val))
            abs_arr = np.asarray(abs_vals, dtype=float)
            signed_arr = np.asarray(signed_vals, dtype=float)
            aggregate[pc_key][variant] = {
                "n_seeds": int(len(abs_vals)),
                "abs_answer_logit_diff_mean_over_seeds": float(np.nanmean(abs_arr)) if abs_arr.size else float("nan"),
                "abs_answer_logit_diff_std_over_seeds": float(np.nanstd(abs_arr, ddof=1)) if abs_arr.size > 1 else 0.0,
                "signed_gold_logit_drop_mean_over_seeds": float(np.nanmean(signed_arr)) if signed_arr.size else float("nan"),
                "signed_gold_logit_drop_std_over_seeds": float(np.nanstd(signed_arr, ddof=1)) if signed_arr.size > 1 else 0.0,
            }
    return aggregate


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run answer-only causal ablations across depth setups.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=path_utils.DEFAULT_TREE_DATASET_TAG,
        help="Traversal dataset folder tag.",
    )
    parser.add_argument(
        "--reasoning-models",
        nargs="+",
        default=["7B"],
        help="Reasoning parameter counts. Use one value (e.g., 7B).",
    )
    parser.add_argument(
        "--chat-models",
        nargs="+",
        default=["none"],
        help="Chat parameter counts. Keep 'none' for this script by default.",
    )
    parser.add_argument("--layer", type=int, default=21, help="Transformer layer to ablate.")
    parser.add_argument("--pcs", nargs="+", type=int, default=[10, 20, 50, 100], help="PCA dimensions to evaluate.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ridge", "logreg_depth_class", "mlp_64", "rf_300"],
        choices=("ridge", "logreg_depth_class", "mlp_64", "rf_300"),
        help="Depth setup methods.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46], help="Run seeds.")
    parser.add_argument("--train-split", type=float, default=0.5, help="Train fraction for balanced split.")
    parser.add_argument(
        "--bucket",
        type=str,
        default="exact",
        choices=("all", "exact", "partial", "zero"),
        help="Evaluation bucket from test split.",
    )
    parser.add_argument("--num-random", type=int, default=1, help="Number of random 1D baselines per setup.")
    parser.add_argument("--max-examples", type=int, default=0, help="Optional cap on selected test examples (0=all).")
    parser.add_argument("--seed", type=int, default=42, help="Global seed for reproducibility.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output NPZ path.")
    parser.add_argument(
        "--save-json-summary",
        action="store_true",
        help="Also save <output>.summary.json with aggregate metrics.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_global_seed(args.seed)

    model_id = resolve_single_model_id(args.reasoning_models, args.chat_models)
    dataset_tag = args.dataset

    responses_fp = responses_path(dataset_tag, model_id)
    embeddings_fp = embeddings_path(dataset_tag, model_id)

    responses = eval_probe.load_responses(responses_fp)  # includes label_mapping needed by build_layer_from_cache
    emb_cache, _ = load_embedding_payload(embeddings_fp)

    tokenizer, model = load_reasoning_model(model_id, device=DEVICE, use_half_precision=True)
    model.eval()
    model_dtype = next(model.parameters()).dtype
    target_layer = intervene._resolve_layer_module(model, args.layer)

    hidden_dim: Optional[int] = None
    seed_blocks: List[Dict[str, Any]] = []

    for run_seed in args.seeds:
        print(f"\n=== Seed {run_seed} ===")
        train_records, test_records = split_balanced(responses, args.train_split, run_seed, exact_attr="exact_match")
        selected = _build_bucket(test_records, args.bucket)
        if args.max_examples and args.max_examples > 0:
            selected = selected[: args.max_examples]
        if not selected:
            print(f"Seed {run_seed}: no selected test records for bucket '{args.bucket}', skipping.")
            continue

        example_cache = _build_example_cache(selected, tokenizer=tokenizer, model=model)
        if not example_cache:
            print(f"Seed {run_seed}: no usable examples with answer tokens, skipping.")
            continue
        print(f"Seed {run_seed}: using {len(example_cache)} examples from bucket '{args.bucket}'.")

        layer_data = eval_probe.build_layer_from_cache(train_records, test_records, args.layer, emb_cache)
        if layer_data is None:
            print(f"Seed {run_seed}: missing layer data, skipping.")
            continue

        X_hidden = np.asarray(layer_data["X"], dtype=np.float32)
        depths = np.asarray(layer_data["depth"], dtype=np.float32)
        train_idx = np.asarray(layer_data["train_idx"], dtype=int)
        example_ids = np.asarray(layer_data["example_ids"], dtype=np.int64)
        if hidden_dim is None:
            hidden_dim = int(X_hidden.shape[1])

        seed_result: Dict[str, Any] = {"seed": int(run_seed), "n_examples": len(example_cache), "pcs": {}}

        for pc in args.pcs:
            ncomp = min(int(pc), int(X_hidden.shape[1]), int(train_idx.size))
            if ncomp <= 0:
                continue
            print(f"Seed {run_seed} | PC={pc}: fitting PCA and depth setups ...")
            pca = PCA(n_components=ncomp, svd_solver="auto", random_state=run_seed)
            pca.fit(X_hidden[train_idx])
            Z = pca.transform(X_hidden).astype(np.float32)
            components = np.asarray(pca.components_, dtype=np.float32)  # [pc, hidden_dim]

            pc_variants: Dict[str, Any] = {}

            # Baseline: top-1 PC direction (in hidden space).
            top1_hidden = _normalize_direction(components[0])
            pc_variants["top1pc_baseline"] = _evaluate_basis_answer_only(
                top1_hidden,
                example_cache=example_cache,
                target_layer=target_layer,
                tokenizer=tokenizer,
                model=model,
                model_dtype=model_dtype,
            )

            # Baseline: random 1D direction(s) in hidden space.
            random_runs: List[Dict[str, Any]] = []
            for ridx in range(int(args.num_random)):
                rng = np.random.default_rng(run_seed + 10_000 * (ridx + 1) + pc)
                rand_vec = rng.standard_normal(size=(components.shape[1],)).astype(np.float32)
                random_runs.append(
                    _evaluate_basis_answer_only(
                        rand_vec,
                        example_cache=example_cache,
                        target_layer=target_layer,
                        tokenizer=tokenizer,
                        model=model,
                        model_dtype=model_dtype,
                    )
                )
            if random_runs:
                abs_vals = np.asarray([r["abs_answer_logit_diff_mean"] for r in random_runs], dtype=float)
                signed_vals = np.asarray([r["signed_gold_logit_drop_mean"] for r in random_runs], dtype=float)
                pc_variants["random_baseline"] = {
                    "n_random": int(len(random_runs)),
                    "abs_answer_logit_diff_mean": float(np.nanmean(abs_vals)),
                    "abs_answer_logit_diff_std": float(np.nanstd(abs_vals, ddof=1)) if len(random_runs) > 1 else 0.0,
                    "signed_gold_logit_drop_mean": float(np.nanmean(signed_vals)),
                    "signed_gold_logit_drop_std": float(np.nanstd(signed_vals, ddof=1)) if len(random_runs) > 1 else 0.0,
                    "runs": random_runs,
                }
            else:
                pc_variants["random_baseline"] = {
                    "n_random": 0,
                    "abs_answer_logit_diff_mean": float("nan"),
                    "abs_answer_logit_diff_std": float("nan"),
                    "signed_gold_logit_drop_mean": float("nan"),
                    "signed_gold_logit_drop_std": float("nan"),
                    "runs": [],
                }

            # Baseline: zero-ablation hook.
            zero_abs: List[float] = []
            zero_signed: List[float] = []
            for item in example_cache:
                zero_logits = intervene._compute_answer_logits(
                    prompt_text=item.prompt_text,
                    prefix_text=item.prefix_text,
                    answer_text=item.answer_text,
                    tokenizer=tokenizer,
                    model=model,
                    target_layer=target_layer,
                    ablation_zero=True,
                )
                zero_abs.append(
                    intervene._mean_abs_answer_logit_diff(
                        item.baseline_logits,
                        zero_logits,
                        item.answer_token_ids,
                    )
                )
                zero_signed.append(
                    _mean_signed_gold_logit_drop(
                        item.baseline_logits,
                        zero_logits,
                        item.answer_token_ids,
                    )
                )
            zero_abs_arr = np.asarray(zero_abs, dtype=float)
            zero_signed_arr = np.asarray(zero_signed, dtype=float)
            pc_variants["zero_baseline"] = {
                "abs_answer_logit_diff_mean": float(np.nanmean(zero_abs_arr)),
                "abs_answer_logit_diff_std": float(np.nanstd(zero_abs_arr, ddof=1)) if zero_abs_arr.size > 1 else 0.0,
                "signed_gold_logit_drop_mean": float(np.nanmean(zero_signed_arr)),
                "signed_gold_logit_drop_std": float(np.nanstd(zero_signed_arr, ddof=1)) if zero_signed_arr.size > 1 else 0.0,
                "n_examples": int(len(example_cache)),
            }

            # Method directions in PCA space, mapped back to hidden space.
            for method in args.methods:
                dir_pc, diagnostics = _train_method_direction_pc(
                    method=method,
                    Z=Z,
                    depths=depths,
                    train_idx=train_idx,
                    example_ids=example_ids,
                    seed=run_seed,
                )
                # Align sign with train-depth correlation.
                train_score = _safe_corr(Z[train_idx] @ dir_pc, depths[train_idx])
                if np.isfinite(train_score) and train_score < 0:
                    dir_pc = -dir_pc
                    train_score = -train_score
                diagnostics["train_direction_depth_corr"] = train_score

                dir_hidden = components.T @ dir_pc.reshape(-1, 1)
                dir_hidden = _normalize_direction(dir_hidden.reshape(-1))
                metrics = _evaluate_basis_answer_only(
                    dir_hidden,
                    example_cache=example_cache,
                    target_layer=target_layer,
                    tokenizer=tokenizer,
                    model=model,
                    model_dtype=model_dtype,
                )
                metrics["diagnostics"] = diagnostics
                pc_variants[method] = metrics

            seed_result["pcs"][str(pc)] = {
                "n_components_fit": int(ncomp),
                "variants": pc_variants,
            }

        seed_blocks.append(seed_result)

    aggregate = _summarize_seed_blocks(seed_blocks, methods=args.methods, pcs=args.pcs)

    out_meta = {
        "dataset": dataset_tag,
        "model_id": model_id,
        "model_tag": model_tag(model_id),
        "responses_path": str(path_utils.repo_relative(responses_fp)),
        "embeddings_path": str(path_utils.repo_relative(embeddings_fp)),
        "layer": int(args.layer),
        "pcs": [int(x) for x in args.pcs],
        "methods": list(args.methods),
        "seeds": [int(x) for x in args.seeds],
        "train_split": float(args.train_split),
        "bucket": str(args.bucket),
        "num_random": int(args.num_random),
        "max_examples": int(args.max_examples),
        "answer_only": True,
    }

    default_output = intervention_output_dir(dataset_tag, model_id) / f"answeronly_depth_grid_layer{args.layer}.npz"
    output_path = args.output or default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        meta=np.array(out_meta, dtype=object),
        seed_results=np.array(seed_blocks, dtype=object),
        aggregate=np.array(aggregate, dtype=object),
    )

    if args.save_json_summary:
        summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump({"meta": out_meta, "aggregate": aggregate}, fh, indent=2)

    print(f"\nSaved answer-only depth ablation grid to {output_path}")
    print("Aggregate preview:")
    for pc in args.pcs:
        pc_key = str(pc)
        if pc_key not in aggregate:
            continue
        row = aggregate[pc_key]
        ridge = row.get("ridge", {})
        rf = row.get("rf_300", {})
        top1 = row.get("top1pc_baseline", {})
        rand = row.get("random_baseline", {})
        print(
            f"PC={pc}: "
            f"ridge drop={ridge.get('signed_gold_logit_drop_mean_over_seeds', float('nan')):.4f}, "
            f"rf drop={rf.get('signed_gold_logit_drop_mean_over_seeds', float('nan')):.4f}, "
            f"top1 drop={top1.get('signed_gold_logit_drop_mean_over_seeds', float('nan')):.4f}, "
            f"rand drop={rand.get('signed_gold_logit_drop_mean_over_seeds', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
