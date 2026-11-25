"""Generate the fully synthetic regional-economics example dataset (seeded, standard library only).

All values are invented by this script. They do not describe any real region, person or firm.
Usage: python examples/make_synthetic_economics.py > examples/synthetic_regions.csv
"""

from __future__ import annotations

import random
import sys

MACRO_REGIONS = ("Alpha", "Bravo", "Charlie", "Delta", "Şahin")  # non-ASCII label on purpose
N_ROWS = 150


def generate() -> str:
    rng = random.Random(2026)
    lines = [
        "region_id,macro_region,gdp_per_capita,unemployment_rate,education_years,household_income"
    ]
    for i in range(1, N_ROWS + 1):
        macro = MACRO_REGIONS[(i * 7) % len(MACRO_REGIONS)]
        level = rng.gauss(0, 1) + 0.6 * MACRO_REGIONS.index(macro)
        gdp = 14000 + 4200 * level + rng.gauss(0, 900)
        edu = 9.5 + 1.4 * level + rng.gauss(0, 0.8)
        unemployment = 9.0 - 1.1 * level + rng.gauss(0, 1.3)
        income = 0.55 * gdp + rng.gauss(0, 1200)
        fields = [
            f"R{i:03d}",
            macro,
            f"{gdp:.2f}",
            f"{unemployment:.2f}",
            f"{edu:.2f}",
            f"{income:.2f}",
        ]
        if i % 17 == 0:
            fields[3] = ""  # missing unemployment
        if i % 29 == 0:
            fields[4] = ""  # missing education
        lines.append(",".join(fields))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.stdout.write(generate())
