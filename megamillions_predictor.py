"""Mega Millions predictor: same framework as lotto_predictor.py, locked to Mega Millions.
Usage: python megamillions_predictor.py [any lotto_predictor flag, e.g. --no-update --backtest --study 2025 2026]
Data: data/megamillions_draws.csv   Workbook: output/megamillions_workbook.xlsx   Study: output/megamillions_study_*.xlsx
"""
import sys
import lotto_predictor

if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    sys.exit(lotto_predictor.main(["--game", "megamillions"] + args))
