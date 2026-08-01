"""Validate conformal p-values and BH on synthetic data with known truth."""
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import stats as sps

from src.stats import (
    conformal_pvalues,
    benjamini_hochberg,
    realized_fdp,
    realized_power,
)

N_CAL, N_TEST, PI1, MU = 20_000, 5_000, 0.02, 4.0


def one_trial(q, seed):
    r = np.random.default_rng(seed)
    cal = r.normal(0, 1, N_CAL)
    is_anom = r.random(N_TEST) < PI1
    test = np.where(is_anom, r.normal(MU, 1, N_TEST), r.normal(0, 1, N_TEST))
    p = conformal_pvalues(cal, test)
    reject, _ = benjamini_hochberg(p, q=q)
    return realized_fdp(reject, is_anom), realized_power(reject, is_anom), p[~is_anom]


def main():
    _, _, p_null = one_trial(0.10, 1)
    ks = sps.kstest(p_null, "uniform")
    print("CHECK 1  null p-values uniform")
    print(f"  mean p  = {p_null.mean():.4f}   (expect ~0.500)")
    print(f"  KS stat = {ks.statistic:.4f}, p = {ks.pvalue:.3f}   (expect p > 0.05)")
    print(f"  {'PASS' if ks.pvalue > 0.05 else 'FAIL'}\n")

    print("CHECK 2  E[FDP] <= q  (200 trials each)")
    print(f"  {'q':>6} {'mean FDP':>10} {'power':>8}  status")
    for q in [0.01, 0.05, 0.10, 0.20, 0.30]:
        res = [one_trial(q, 1000 + i) for i in range(200)]
        fdp = float(np.mean([a for a, _, _ in res]))
        pwr = float(np.mean([b for _, b, _ in res]))
        ok = "PASS" if fdp <= q + 0.01 else "FAIL"
        print(f"  {q:>6.2f} {fdp:>10.4f} {pwr:>8.3f}  {ok}")


if __name__ == "__main__":
    main()