"""Table 2: oracle quality ladder evaluated on the selection pool."""

from __future__ import annotations

from experiments.quality_ladder.common import CALIBRATION, LADDER_ROWS, OUTPUT
from experiments.quality_ladder.oracle_ladder import compute_oracle_ladder
from experiments.quality_ladder.selection_quality import compute_selection_quality, summarize_witnesses


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    calibration, ladder, _, _ = compute_oracle_ladder()
    calibration.to_csv(CALIBRATION, index=False)
    ladder.to_csv(LADDER_ROWS, index=False)
    quality, _ = compute_selection_quality()
    quality.to_csv(OUTPUT / "oracle_selection_quality.csv", index=False)
    summarize_witnesses(quality).to_csv(OUTPUT / "table_2.csv", index=False)


if __name__ == "__main__":
    main()
