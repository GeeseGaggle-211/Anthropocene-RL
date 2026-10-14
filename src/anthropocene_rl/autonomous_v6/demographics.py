"""Focal-resident interval welfare and native demographic ledger decomposition."""

import math


def summarize(trace):
    if not trace:
        raise ValueError("Empty demographic trajectory")
    duration = sum(r["elapsed"] for r in trace)
    person_years = sum(r["population_start"] * r["elapsed"] for r in trace)
    welfare = sum(r["reward"] for r in trace)
    initial, final = trace[0]["population_start"], trace[-1]["population_end"]
    totals = {
        k: 0
        for k in [
            "ResidentGrowth",
            "StarvationDeaths",
            "Immigrants",
            "Emigrants",
            "FissionHeadcount",
            "OtherResidentChange",
        ]
    }
    triggers = {}
    previous = initial
    compiled = []
    for r in trace:
        d = r["demographics"]
        if (
            d["SchemaVersion"] != 1
            or d["OpeningPopulation"] != r["population_start"]
            or d["EndingPopulation"] != r["population_end"]
        ):
            raise ValueError("Native owner demographic schema/receipt mismatch")
        expected = (
            previous
            + d["ResidentGrowth"]
            - d["StarvationDeaths"]
            + d["Immigrants"]
            - d["Emigrants"]
            - d["FissionHeadcount"]
            + d["OtherResidentChange"]
        )
        if (
            previous != r["population_start"]
            or expected != r["population_end"]
            or sum(d["FissionByTrigger"].values()) != d["FissionHeadcount"]
        ):
            raise ValueError("Native resident population ledger does not reconcile")
        if not math.isclose(
            r["reward"],
            r["population_start"] * r["qol"] * r["elapsed"],
            rel_tol=1e-12,
            abs_tol=1e-8,
        ):
            raise ValueError("Focal-only welfare identity violated")
        previous = r["population_end"]
        for k in totals:
            totals[k] += d[k]
        for k, v in d["FissionByTrigger"].items():
            triggers[k] = triggers.get(k, 0) + v
        if sum(r["staffing"]) != r["population_start"]:
            raise ValueError("Compiled staffing does not equal opening residents")
        compiled.append([n / r["population_start"] for n in r["staffing"]])

    def mean(values):
        return sum(v * r["elapsed"] for v, r in zip(values, trace, strict=True)) / duration

    return dict(
        initial_population=initial,
        final_population=final,
        time_mean_population=person_years / duration,
        person_years=person_years,
        time_mean_qol=mean([r["qol"] for r in trace]),
        person_year_weighted_qol=welfare / person_years if person_years else 0,
        raw_welfare=welfare,
        annual_welfare=welfare / duration,
        population_normalized_welfare=welfare / initial if initial else 0,
        outflows=totals,
        unexplained_resident_change=totals["OtherResidentChange"],
        all_resident_changes_explained=all(
            r["demographics"]["OtherResidentChange"] == 0 for r in trace
        ),
        fission_headcount_by_trigger=triggers,
        population_accounting_residual=final
        - initial
        - totals["ResidentGrowth"]
        + totals["StarvationDeaths"]
        - totals["Immigrants"]
        + totals["Emigrants"]
        + totals["FissionHeadcount"]
        - totals["OtherResidentChange"],
        allocation={
            name: dict(
                requested=mean([r["requested_workforce"][i] for r in trace]),
                compiled=mean([row[i] for row in compiled]),
            )
            for name, i in [("Gatherer", 0), ("Farmer", 1), ("Craftsperson", 4)]
        },
        expansion_farmer_share=mean([r["requested_controls"][0] for r in trace]),
        food_reserve_years=mean([r["demographics"]["FoodReserveYears"] for r in trace]),
        mean_field_intensity_q=mean([r["demographics"]["MeanFieldIntensity"] for r in trace]),
        person_year_convention=(
            "opening focal residents x interval years; no daughter future welfare"
        ),
    )
