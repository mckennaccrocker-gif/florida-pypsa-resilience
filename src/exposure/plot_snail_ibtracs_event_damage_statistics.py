"""
Graphs and statistical analysis for Snail IBTrACS event damage results.

Inputs:
    data/Exposure/florida_open_gira_tc_event_damage_snail/
        snail_ibtracs_event_network_damage.csv

Outputs:
    data/Exposure/florida_open_gira_tc_event_damage_snail/statistical_graphs/

Advisor-requested outputs:
    1. Event vs. total damage, ranked highest to lowest.
    2. Damage vs. event rank / exceedance curve.
    3. Probability distribution fitted to ranked damage values.
    4. Estimated 10-, 20-, 50-, 100-, 200-, and 500-year damages.
    5. Comparison of estimated 100-year damage with existing STORM/open-gira
       return-period damage.
"""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
EVENT_DIR = PROJECT_DIR / "data" / "Exposure" / "florida_open_gira_tc_event_damage_snail"
OUTPUT_DIR = EVENT_DIR / "statistical_graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_DAMAGE_CSV = EVENT_DIR / "snail_ibtracs_event_network_damage.csv"

EXISTING_STORM_DAMAGE_CSV = (
    PROJECT_DIR
    / "data"
    / "Exposure"
    / "florida_open_gira_tc_exposure"
    / "snail_tc_intersection"
    / "snail_storm_return_period_damage_vs_fred.csv"
)

RETURN_PERIODS = [10, 20, 50, 100, 200, 500]

# The event table includes many events with zero or tiny near-zero damage because
# storms far from Florida still exist in the IBTrACS event stack. A lognormal tail
# fit is only meaningful for damaging events, so this threshold is used for the
# fitted return-period estimates and is written to the output notes.
DAMAGE_FIT_THRESHOLD_USD = 1_000_000


def load_ranked_event_damage() -> pd.DataFrame:
    """Load event damage and rank events from highest to lowest damage."""
    df = pd.read_csv(EVENT_DAMAGE_CSV)
    df = df.sort_values("total_network_damage_usd", ascending=False).reset_index(drop=True)
    df["event_rank"] = np.arange(1, len(df) + 1)
    df["damage_billion_usd"] = df["total_network_damage_usd"] / 1e9
    df["empirical_event_exceedance_probability"] = df["event_rank"] / (len(df) + 1)
    return df


def plot_event_vs_total_damage(ranked: pd.DataFrame) -> None:
    """Ranked event damage plot with a readable top-event panel."""
    top_n = 20
    top_events = ranked.head(top_n).sort_values("damage_billion_usd")

    fig, (ax_top, ax_all) = plt.subplots(
        2,
        1,
        figsize=(11, 8.5),
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    colors = ["#1f77b4"] * len(top_events)
    colors[-1] = "#d62728"
    bars = ax_top.barh(
        top_events["event_id"],
        top_events["damage_billion_usd"],
        color=colors,
    )
    for bar, value in zip(bars, top_events["damage_billion_usd"]):
        ax_top.text(
            value + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"${value:.2f}B",
            va="center",
            fontsize=8,
        )

    ax_top.set_xlabel("Total network damage (billion USD)")
    ax_top.set_ylabel("IBTrACS event ID")
    ax_top.set_title("Top 20 Snail IBTrACS Events by Florida Transmission Damage")
    ax_top.grid(axis="x", alpha=0.25)
    ax_top.set_xlim(0, max(top_events["damage_billion_usd"].max() * 1.18, 0.1))

    ax_all.bar(
        ranked["event_rank"],
        ranked["damage_billion_usd"],
        color="#9ecae1",
        width=0.95,
    )
    ax_all.axvline(top_n, color="#d62728", linestyle="--", linewidth=1)
    ax_all.text(
        top_n + 2,
        ranked["damage_billion_usd"].max() * 0.75,
        "Top 20 shown above",
        color="#d62728",
        fontsize=9,
    )
    ax_all.set_xlabel("Event rank, ordered from highest damage to lowest")
    ax_all.set_ylabel("Damage (B USD)")
    ax_all.set_title("All 175 Events Ranked")
    ax_all.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "event_vs_total_damage_ranked.png", dpi=300)
    plt.close()


def plot_damage_vs_event_rank(ranked: pd.DataFrame) -> None:
    """Exceedance/rank curve requested by advisor."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        ranked["event_rank"],
        ranked["total_network_damage_usd"],
        color="#d62728",
        linewidth=1.6,
        marker="o",
        markersize=3,
    )
    ax.set_xlabel("Event rank")
    ax.set_ylabel("Total network damage (USD)")
    ax.set_title("Damage vs. Event Rank (Exceedance Curve)")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "damage_vs_event_rank_exceedance_curve.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(
        ranked["event_rank"],
        ranked["total_network_damage_usd"].clip(lower=1),
        color="#d62728",
        linewidth=1.6,
        marker="o",
        markersize=3,
    )
    ax.set_xlabel("Event rank")
    ax.set_ylabel("Total network damage (USD, log scale)")
    ax.set_title("Damage vs. Event Rank (Log-Scale Exceedance Curve)")
    ax.grid(alpha=0.25, which="both")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "damage_vs_event_rank_exceedance_curve_log.png", dpi=300)
    plt.close()


def fit_lognormal_tail(ranked: pd.DataFrame) -> dict:
    """
    Fit a lognormal distribution to damaging event values above the threshold.

    Without scipy, the maximum-likelihood lognormal fit is simply the mean and
    standard deviation of log(damage) for positive tail values.
    """
    tail = ranked.loc[
        ranked["total_network_damage_usd"] > DAMAGE_FIT_THRESHOLD_USD,
        "total_network_damage_usd",
    ].copy()
    if len(tail) < 5:
        raise ValueError("Not enough damaging events to fit a lognormal tail.")

    log_damage = np.log(tail.to_numpy())
    years_observed = ranked["event_year"].max() - ranked["event_year"].min() + 1
    tail_event_rate_per_year = len(tail) / years_observed

    return {
        "distribution": "lognormal_tail",
        "damage_threshold_usd": DAMAGE_FIT_THRESHOLD_USD,
        "tail_event_count": len(tail),
        "years_observed": years_observed,
        "tail_event_rate_per_year": tail_event_rate_per_year,
        "mu_log_damage": float(log_damage.mean()),
        "sigma_log_damage": float(log_damage.std(ddof=0)),
    }


def estimate_return_period_damages(fit: dict) -> pd.DataFrame:
    """
    Estimate return-period damages from the fitted damaging-event tail.

    If damaging events above the threshold occur at rate lambda per year, then
    a T-year annual exceedance probability is converted to a conditional event-tail
    exceedance probability:

        p_tail = 1 / (T * lambda)

    The return-period damage is then the lognormal quantile with CDF 1 - p_tail.
    """
    normal = NormalDist()
    records = []

    for return_period in RETURN_PERIODS:
        annual_exceedance_probability = 1 / return_period
        p_tail = annual_exceedance_probability / fit["tail_event_rate_per_year"]

        if p_tail >= 1:
            estimated_damage = np.nan
            note = "Return period is too frequent for selected damaging-event threshold."
        else:
            quantile = 1 - p_tail
            z_value = normal.inv_cdf(quantile)
            estimated_damage = np.exp(
                fit["mu_log_damage"] + fit["sigma_log_damage"] * z_value
            )
            note = ""

        records.append(
            {
                "return_period_years": return_period,
                "annual_exceedance_probability": annual_exceedance_probability,
                "conditional_tail_exceedance_probability": p_tail,
                "estimated_damage_usd": estimated_damage,
                "estimated_damage_billion_usd": estimated_damage / 1e9,
                "fit_distribution": fit["distribution"],
                "fit_damage_threshold_usd": fit["damage_threshold_usd"],
                "tail_event_rate_per_year": fit["tail_event_rate_per_year"],
                "note": note,
            }
        )

    return pd.DataFrame(records)


def compare_with_existing_storm(return_period_estimates: pd.DataFrame) -> pd.DataFrame:
    """Compare fitted historical-event return levels with existing STORM RP damage."""
    output = return_period_estimates.copy()
    if not EXISTING_STORM_DAMAGE_CSV.exists():
        output["existing_storm_damage_usd"] = np.nan
        output["difference_fit_minus_existing_storm_usd"] = np.nan
        return output

    storm = pd.read_csv(EXISTING_STORM_DAMAGE_CSV)
    storm = storm.rename(
        columns={
            "return_period": "return_period_years",
            "storm_damage_usd": "existing_storm_damage_usd",
        }
    )
    output = output.merge(
        storm[["return_period_years", "existing_storm_damage_usd"]],
        on="return_period_years",
        how="left",
    )
    output["existing_storm_damage_billion_usd"] = (
        output["existing_storm_damage_usd"] / 1e9
    )
    output["difference_fit_minus_existing_storm_usd"] = (
        output["estimated_damage_usd"] - output["existing_storm_damage_usd"]
    )
    output["difference_fit_minus_existing_storm_percent"] = np.where(
        output["existing_storm_damage_usd"] > 0,
        output["difference_fit_minus_existing_storm_usd"]
        / output["existing_storm_damage_usd"]
        * 100,
        np.nan,
    )
    return output


def plot_return_period_damage_comparison(comparison: pd.DataFrame) -> None:
    """Plot fitted historical-event return-period damage vs existing STORM estimates."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        comparison["return_period_years"],
        comparison["estimated_damage_billion_usd"],
        marker="o",
        linewidth=2,
        label="Fitted IBTrACS event damage distribution",
    )

    if comparison["existing_storm_damage_billion_usd"].notna().any():
        ax.plot(
            comparison["return_period_years"],
            comparison["existing_storm_damage_billion_usd"],
            marker="s",
            linewidth=2,
            linestyle="--",
            label="Existing STORM/open-gira return-period damage",
        )

    row100 = comparison.loc[comparison["return_period_years"].eq(100)]
    if not row100.empty:
        y_value = float(row100["estimated_damage_billion_usd"].iloc[0])
        ax.scatter([100], [y_value], s=90, color="#d62728", zorder=5)
        ax.text(100, y_value, f"  Fitted 100-yr ${y_value:.2f}B", va="center")

    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Estimated total network damage (billion USD)")
    ax.set_title("Estimated Return-Period Damages from Historical IBTrACS Events")
    ax.set_xticks(RETURN_PERIODS)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "return_period_damage_estimates_vs_storm.png", dpi=300)
    plt.close()


def save_fit_notes(fit: dict) -> None:
    """Write a short plain-text note explaining the distribution choice."""
    notes = [
        "Snail IBTrACS event damage statistical analysis",
        "",
        f"Fitted distribution: {fit['distribution']}",
        f"Damage threshold for fit: ${fit['damage_threshold_usd']:,.0f}",
        f"Tail event count: {fit['tail_event_count']}",
        f"Years observed: {fit['years_observed']}",
        f"Tail event rate per year: {fit['tail_event_rate_per_year']:.4f}",
        f"mu(log damage): {fit['mu_log_damage']:.6f}",
        f"sigma(log damage): {fit['sigma_log_damage']:.6f}",
        "",
        "Interpretation:",
        "The fit is a sensitivity-style return-level estimate based on historical",
        "IBTrACS event damages. Many events have zero or near-zero Florida damage,",
        "so the lognormal fit is applied only to damaging events above the stated",
        "threshold. Return-period estimates should be treated as approximate and",
        "compared against the existing STORM/open-gira return-period dataset.",
    ]
    (OUTPUT_DIR / "distribution_fit_notes.txt").write_text("\n".join(notes))


def main() -> None:
    ranked = load_ranked_event_damage()
    ranked.to_csv(OUTPUT_DIR / "ranked_event_damage.csv", index=False)

    plot_event_vs_total_damage(ranked)
    plot_damage_vs_event_rank(ranked)

    fit = fit_lognormal_tail(ranked)
    save_fit_notes(fit)

    return_period_estimates = estimate_return_period_damages(fit)
    comparison = compare_with_existing_storm(return_period_estimates)
    comparison.to_csv(OUTPUT_DIR / "return_period_damage_estimates.csv", index=False)

    plot_return_period_damage_comparison(comparison)

    print("Saved graphs and statistics in:", OUTPUT_DIR)
    print("\nReturn-period damage estimates:")
    print(
        comparison[
            [
                "return_period_years",
                "estimated_damage_billion_usd",
                "existing_storm_damage_billion_usd",
                "difference_fit_minus_existing_storm_percent",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
