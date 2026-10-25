"""Tiny harness config for tests: one DGP A scenario, cheap estimators, and a config
that always fails."""

from experiments.harness import Config, Scenario, multi, single
from src.dgps.authors import dgp_a_params, generate, true_ate
from src.estimators import fed_ps, meta_sw

BASE_SEED = 7

SCENARIOS = [Scenario(
    name="A_poor_small",
    generate=lambda seed: generate(dgp_a_params("poor", n_k=200), "A", seed)[1],
    truth=lambda n, seed: true_ate(dgp_a_params("poor", n_k=200), "A", n, seed),
    truth_n=30_000,
)]


def _boom(df, seed, rep):
    raise RuntimeError("deliberate failure")


CONFIGS = [
    multi("fed", fed_ps.estimate_multi, ["IPW-HT", "AIPW"], bootstrap=5,
          bootstrap_reps=range(0, 1), bootstrap_for=["IPW-HT"]),
    single("meta_ht", meta_sw.estimate),
    Config("broken", _boom, ["x"]),
]
