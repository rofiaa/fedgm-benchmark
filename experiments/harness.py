"""Generic Monte Carlo harness: scenarios × estimator configs × reps -> one CSV row per
(rep, config, estimator).

A **scenario** generates a data frame (X1..Xd, H, W, Y) from a seed and knows its true
ATE. A **config** runs on one data frame and returns {estimator label: result dict};
every result honours the frozen keys of src/estimators/common.py (tau, se, se_method,
max_smd, runtime). A config may share nuisance fits across several estimators
(fed_ps.estimate_multi) and so emit several rows, all tagged with its config_id.
Adding an estimator (e.g. GenMatch) = adding a config.

Seeds are derived from (base_seed, scenario name, rep[, config_id]) with
np.random.SeedSequence spawn keys, so any subset of reps (e.g. one SLURM array task)
reproduces exactly the rows of a full run.
"""

import json
import os
import time
import traceback
import warnings
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.estimators.common import REQUIRED_KEYS, check_result


@dataclass
class Scenario:
    """generate(seed) -> data frame; truth(n, seed) -> (tau, Monte Carlo SE)."""
    name: str
    generate: Callable
    truth: Callable
    truth_n: int = 1_000_000


@dataclass
class Config:
    """run(df, seed, rep) -> {estimator label: result dict}. `labels` lists the
    estimators it emits (used for error rows)."""
    config_id: str
    run: Callable
    labels: list
    meta: dict = field(default_factory=dict)  # free-form, copied into each row


def single(config_id, estimate, label=None, bootstrap=0, bootstrap_reps=None, **kwargs):
    """Config around `estimate(df, seed=..., **kwargs)` (one row per rep).
    bootstrap: B, applied only for reps in `bootstrap_reps` (None = all reps)."""
    label = label or config_id

    def run(df, seed, rep):
        boot = bootstrap if bootstrap_reps is None or rep in bootstrap_reps else 0
        kw = dict(kwargs, bootstrap=boot) if bootstrap else kwargs
        return {label: estimate(df, seed=seed, **kw)}

    return Config(config_id, run, [label], dict(kwargs))


def multi(config_id, estimate_multi, estimands, bootstrap=0, bootstrap_reps=None,
          bootstrap_for=None, **kwargs):
    """Config around `estimate_multi(df, estimands, seed=..., **kwargs)`: one shared
    fit, one row per estimand. bootstrap: B for the estimands in `bootstrap_for`, only
    for reps in `bootstrap_reps` (None = all reps)."""

    def run(df, seed, rep):
        boot = bootstrap if bootstrap_reps is None or rep in bootstrap_reps else 0
        return estimate_multi(df, estimands, seed=seed, bootstrap=boot,
                              bootstrap_for=bootstrap_for, **kwargs)

    return Config(config_id, run, list(estimands), dict(kwargs))


def _key(text):
    return zlib.crc32(text.encode())


def data_seed(base_seed, scenario, rep):
    seq = np.random.SeedSequence(base_seed, spawn_key=(_key(scenario), rep))
    return int(seq.generate_state(1)[0])


def config_seed(base_seed, scenario, rep, config_id):
    seq = np.random.SeedSequence(base_seed, spawn_key=(_key(scenario), rep, _key(config_id)))
    return int(seq.generate_state(1)[0])


def flatten(result):
    """Frozen keys first, then scalar extras; 2-tuples -> _lo/_hi, dicts of scalars ->
    prefix_key, lists of str -> ';'-joined. Arrays, Series and frames are dropped."""
    row = {k: result[k] for k in REQUIRED_KEYS}
    for key, value in result.items():
        if key in row:
            continue
        if value is None or isinstance(value, (bool, int, float, str, np.integer, np.floating)):
            row[key] = value
        elif isinstance(value, tuple) and len(value) == 2 and all(
                isinstance(v, (int, float, np.integer, np.floating)) for v in value):
            row[f"{key}_lo"], row[f"{key}_hi"] = value
        elif isinstance(value, dict) and all(
                isinstance(v, (bool, int, float, str, np.integer, np.floating, type(None)))
                for v in value.values()):
            row.update({f"{key}_{k}": v for k, v in value.items()})
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            row[key] = ";".join(value)
    return row


def true_ate(scenario, base_seed, cache_dir):
    """Monte Carlo true ATE, cached as JSON (written atomically: array tasks may race)."""
    path = Path(cache_dir) / f"{scenario.name}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["truth_n"] == scenario.truth_n and cached["base_seed"] == base_seed:
            return cached["tau"], cached["mc_se"]
    seed = int(np.random.SeedSequence(base_seed, spawn_key=(_key(scenario.name),
                                                            2**31)).generate_state(1)[0])
    tau, mc_se = scenario.truth(scenario.truth_n, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(dict(tau=tau, mc_se=mc_se, truth_n=scenario.truth_n,
                                   base_seed=base_seed, seed=seed)))
    tmp.replace(path)
    return tau, mc_se


def run_rep(scenario, configs, rep, base_seed, tau_true):
    """All configs on one dataset; returns a list of row dicts."""
    seed = data_seed(base_seed, scenario.name, rep)
    df = scenario.generate(seed)
    rows = []
    for config in configs:
        c_seed = config_seed(base_seed, scenario.name, rep, config.config_id)
        base = dict(scenario=scenario.name, rep=rep, data_seed=seed,
                    config_id=config.config_id, config_seed=c_seed, tau_true=tau_true,
                    **{f"cfg_{k}": v for k, v in config.meta.items()
                       if isinstance(v, (bool, int, float, str))})
        start = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = config.run(df, c_seed, rep)
            config_runtime = time.perf_counter() - start
            for label, result in results.items():
                rows.append(dict(base, estimator=label, config_runtime=config_runtime,
                                 error="", **flatten(check_result(result))))
        except Exception as exc:
            config_runtime = time.perf_counter() - start
            err = f"{type(exc).__name__}: {exc}"
            for label in config.labels:
                rows.append(dict(base, estimator=label, config_runtime=config_runtime,
                                 error=err, traceback=traceback.format_exc(limit=3),
                                 **dict.fromkeys(("tau", "se", "max_smd"), np.nan)))
    return rows


def parse_reps(spec):
    """'0-29' -> range(0, 30); '5' -> range(5, 6); '0-9,20-29' -> list."""
    reps = []
    for part in str(spec).split(","):
        lo, _, hi = part.partition("-")
        reps.extend(range(int(lo), int(hi or lo) + 1))
    return reps


_LOADED = {}


def load_config(path):
    """Import an experiment config file: it defines SCENARIOS (list of Scenario),
    CONFIGS (list of Config) and BASE_SEED (int). Cached per process."""
    import importlib.util

    path = str(Path(path).resolve())
    if path not in _LOADED:
        spec = importlib.util.spec_from_file_location(f"_experiment_{_key(path)}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LOADED[path] = module
    return _LOADED[path]


def select(module, scenario_names=None, config_ids=None):
    """The config file's scenarios / configs, optionally restricted by name."""
    unknown = (set(scenario_names or ()) - {s.name for s in module.SCENARIOS}) | (
        set(config_ids or ()) - {c.config_id for c in module.CONFIGS})
    if unknown:
        raise ValueError(f"unknown scenarios/configs: {sorted(unknown)}")
    scenarios = [s for s in module.SCENARIOS if not scenario_names or s.name in scenario_names]
    configs = [c for c in module.CONFIGS if not config_ids or c.config_id in config_ids]
    return scenarios, configs


def _worker(config_path, scenario_name, config_ids, rep, tau_true):
    """Process-pool entry point: rebuild scenario/configs from the config file (they
    hold closures, which cannot be pickled)."""
    module = load_config(config_path)
    scenarios, configs = select(module, [scenario_name], config_ids)
    return run_rep(scenarios[0], configs, rep, module.BASE_SEED, tau_true)


def run(config_path, reps, out_dir, scenario_names=None, config_ids=None, jobs=1):
    """Run the selected configs on `reps` of each selected scenario of the config file;
    write <out_dir>/rows/<scenario>_reps<first>-<last>.csv per scenario."""
    from concurrent.futures import ProcessPoolExecutor

    module = load_config(config_path)
    scenarios, configs = select(module, scenario_names, config_ids)
    config_ids = [c.config_id for c in configs]
    out_dir = Path(out_dir)
    (out_dir / "rows").mkdir(parents=True, exist_ok=True)
    written = []
    pool = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    try:
        for scenario in scenarios:
            tau_true, mc_se = true_ate(scenario, module.BASE_SEED, out_dir / "truth")
            print(f"[{scenario.name}] true ATE {tau_true:.4f} (MC SE {mc_se:.4f})",
                  flush=True)
            if pool:
                futures = [pool.submit(_worker, str(config_path), scenario.name,
                                       config_ids, rep, tau_true) for rep in reps]
                results = (f.result() for f in futures)
            else:
                results = (run_rep(scenario, configs, rep, module.BASE_SEED, tau_true)
                           for rep in reps)
            args = reps
            rows = []
            for i, rep_rows in enumerate(results, 1):
                rows.extend(rep_rows)
                if i % 10 == 0 or i == len(args):
                    print(f"[{scenario.name}] {i}/{len(args)} reps", flush=True)
            path = out_dir / "rows" / f"{scenario.name}_reps{min(reps)}-{max(reps)}.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            written.append(path)
    finally:
        if pool:
            pool.shutdown()
    return written
