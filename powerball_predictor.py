"""Powerball predictor: same framework as lotto_predictor.py, locked to Powerball.
Usage: python powerball_predictor.py [any lotto_predictor flag, e.g. --no-update --backtest --study 2025 2026]
Data: data/powerball_draws.csv   Workbook: output/powerball_workbook.xlsx   Study: output/powerball_study_*.xlsx
"""
import sys
import lotto_predictor

if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    sys.exit(lotto_predictor.main(["--game", "powerball"] + args))
