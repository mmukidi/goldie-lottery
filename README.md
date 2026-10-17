# goldie-lottery

For-fun Powerball and Mega Millions predictor. It is a math experiment; no formula predicts lottery draws.

## Run
```
pip3 install openpyxl      # one time, for the Excel workbook
python3 lotto_predictor.py # update data, learn, predict next draws, log and score guesses
```

## Modes
| Command | What it does |
|---|---|
| `python3 lotto_predictor.py` | Downloads new draws, self-tunes, prints 6 guesses per game, updates the workbook |
| `--remind` | Only acts if a draw is **tomorrow** (ET); prints a short reminder with the guesses |
| `--review [K]` | Shows where the model went wrong on the last K draws and which signal ranked winners best |
| `--audit [N]` | Honest out-of-sample test: does self-tuning beat the default on the last N draws? |
| `--backtest` | Walk-forward test of the five fixed strategies |
| `--date YYYY-MM-DD` | What would it have guessed for that date? |
| `--ingest GAME FILE` | Add draws from a text/JSON file (lines like `2026-10-17 03 28 36 40 66 02`) |
| `--no-update`, `--no-learn`, `--no-workbook`, `--game`, `--days`, `--learn-window` | Options |

## How it learns
Every run replays the last 60 draws using only earlier history, measures how highly the formula ranked the numbers that
actually won (50% = no better than random), and searches a small grid of weights / hot-streak speeds. A new setting is
adopted only if it beats the default by z >= 2.5; otherwise the default stays. This protects against chasing luck.
Lottery draws are random, so expect the result to stay near the 50% random line.

## Files
- `data/*_draws.csv`: draw history (new draws are appended from the NY Open Data feed).
- `data/picks_log.csv`: every guess recorded before its draw, scored on later runs.
- `data/learning_log.csv`: what the tuner chose each run and how close the winners ranked.
- `output/lottery_workbook.xlsx`: regenerated each run (git-ignored).

## Daily reminder
A Claude scheduled task checks every morning and sends a push notification the day before each draw
(Powerball Mon/Wed/Sat, Mega Millions Tue/Fri).
