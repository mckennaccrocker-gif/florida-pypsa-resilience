"""
Finalize the large generator-to-bus desktop review.

This script turns the raw large-generator review table into an Excel-friendly
decision table. It does not blindly change every flagged row: radial 230 kV
assignments close to large plants are treated as likely plant switchyards.
Only close low-voltage-to-high-voltage corrections are marked as accepted
overrides.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
REVIEW_DIR = (
    PROJECT_DIR
    / "data"
    / "Electricity"
    / "pypsa_florida_network_county_population_load"
    / "generator_bus_assignment_review"
)
OVERRIDE_NETWORK_DIR = (
    PROJECT_DIR
    / "data"
    / "Electricity"
    / "pypsa_florida_network_county_load_generator_overrides"
)


REVIEW_FILE = REVIEW_DIR / "large_generator_bus_assignment_review.csv"
APPLIED_OVERRIDES_FILE = OVERRIDE_NETWORK_DIR / "applied_generator_bus_overrides.csv"
DECISION_FILE = REVIEW_DIR / "large_generator_bus_assignment_final_decisions.csv"
NOTES_FILE = REVIEW_DIR / "large_generator_bus_assignment_final_notes.md"


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def classify_row(row: pd.Series, applied_override_lookup: dict[str, dict]) -> tuple[str, str, str]:
    generator = row["generator"]
    assigned_voltage = pd.to_numeric(row.get("assigned_bus_v_nom"), errors="coerce")
    assigned_edges = pd.to_numeric(row.get("assigned_bus_connected_edge_count"), errors="coerce")
    nearest_bus_distance_m = pd.to_numeric(row.get("nearest_bus_distance_m"), errors="coerce")
    candidate_distance_m = pd.to_numeric(
        row.get("nearest_high_voltage_candidate_distance_m"), errors="coerce"
    )
    candidate_bus = row.get("nearest_high_voltage_candidate_bus")

    if generator in applied_override_lookup:
        applied = applied_override_lookup[generator]
        return (
            "accepted_override_applied",
            applied["new_bus"],
            (
                "Large generator was assigned below 230 kV and had a nearby 230+ kV "
                "candidate within 5 km; override applied in copied network."
            ),
        )

    if assigned_voltage >= 230 and assigned_edges <= 1 and nearest_bus_distance_m <= 1000:
        return (
            "keep_current_bus_likely_plant_switchyard",
            row["bus"],
            (
                "Assigned bus is 230+ kV and very close to the plant. Although radial, "
                "this is consistent with a plant switchyard or dedicated interconnection."
            ),
        )

    if (
        assigned_voltage < 230
        and pd.notna(candidate_distance_m)
        and candidate_distance_m <= 5000
        and bool_value(row.get("nearby_hv_candidate_better_than_assigned"))
    ):
        return (
            "recommend_override_not_applied",
            candidate_bus,
            (
                "Low-voltage assignment has a close 230+ kV candidate, but it was not "
                "in the auto-applied override list. Review before applying."
            ),
        )

    if assigned_voltage < 230 and pd.isna(candidate_distance_m):
        return (
            "keep_current_bus_no_close_high_voltage_candidate",
            row["bus"],
            (
                "Assigned bus is below 230 kV, but no 230+ kV candidate was found within "
                "the search radius, so there is not enough evidence to move it."
            ),
        )

    if assigned_voltage < 230 and pd.notna(candidate_distance_m) and candidate_distance_m > 5000:
        return (
            "keep_current_bus_candidate_too_far",
            row["bus"],
            (
                "Nearby 230+ kV candidate is more than 5 km away. Moving the generator "
                "would be a stronger assumption than keeping the close assigned bus."
            ),
        )

    return (
        "keep_current_bus_no_strong_evidence",
        row["bus"],
        "No strong evidence that the current assignment should be changed.",
    )


def main() -> None:
    review = pd.read_csv(REVIEW_FILE)
    applied = pd.read_csv(APPLIED_OVERRIDES_FILE) if APPLIED_OVERRIDES_FILE.exists() else pd.DataFrame()
    applied_lookup = {
        row["generator"]: row.to_dict()
        for _, row in applied.iterrows()
    }

    flagged = review[
        (review["large_generator"].map(bool_value))
        & (pd.to_numeric(review["suspicious_assignment_flag_count"], errors="coerce") > 0)
    ].copy()

    decisions = flagged.apply(
        lambda row: classify_row(row, applied_lookup),
        axis=1,
        result_type="expand",
    )
    flagged["codex_review_decision"] = decisions[0]
    flagged["final_recommended_bus"] = decisions[1]
    flagged["codex_review_notes"] = decisions[2]
    flagged["bus_changed_in_override_network"] = flagged["generator"].isin(applied_lookup)

    columns = [
        "generator",
        "plant_name",
        "carrier",
        "p_nom",
        "bus",
        "final_recommended_bus",
        "bus_changed_in_override_network",
        "codex_review_decision",
        "codex_review_notes",
        "assigned_bus_v_nom",
        "assigned_bus_connected_edge_count",
        "nearest_bus_distance_m",
        "nearest_high_voltage_candidate_bus",
        "nearest_high_voltage_candidate_v_nom",
        "nearest_high_voltage_candidate_connected_edges",
        "nearest_high_voltage_candidate_distance_m",
        "nearby_high_voltage_candidates",
        "suspicious_assignment_flag_count",
        "review_priority_score",
    ]
    decision_table = flagged[columns].sort_values(
        ["bus_changed_in_override_network", "review_priority_score"],
        ascending=[False, False],
    )
    decision_table.to_csv(DECISION_FILE, index=False)

    summary = decision_table["codex_review_decision"].value_counts().sort_index()
    notes = [
        "# Large Generator Bus Assignment Final Review",
        "",
        "This is a conservative desktop review based on plant coordinates, assigned bus",
        "voltage, bus connectivity, distance to the assigned bus, and nearby 230+ kV",
        "candidate buses.",
        "",
        "Rules used:",
        "- Keep close 230+ kV radial assignments because they likely represent plant switchyards.",
        "- Apply only close low-voltage-to-230+ kV corrections where the candidate is within 5 km.",
        "- Keep low-voltage assignments when the high-voltage alternative is far away or missing.",
        "",
        "Decision counts:",
    ]
    for decision, count in summary.items():
        notes.append(f"- {decision}: {count}")
    notes.extend(
        [
            "",
            "The copied network with accepted overrides is:",
            str(OVERRIDE_NETWORK_DIR),
            "",
            "The original county-population network was not overwritten.",
        ]
    )
    NOTES_FILE.write_text("\n".join(notes) + "\n", encoding="utf-8")

    print(f"Flagged large generators reviewed: {len(decision_table)}")
    print(summary.to_string())
    print(f"Saved decisions: {DECISION_FILE}")
    print(f"Saved notes: {NOTES_FILE}")


if __name__ == "__main__":
    main()
