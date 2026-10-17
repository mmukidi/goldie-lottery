# goldie-lottery

For-fun Powerball and Mega Millions predictor. It is a math experiment; no formula predicts lottery draws.

## Run
```
pip3 install openpyxl      # one time, for the Excel workbook
python3 lotto_predictor.py # update data, predict next draws, log and score guesses
```

Options: `--no-update`, `--game powerball|megamillions`, `--date YYYY-MM-DD`, `--backtest`, `--days 365`.

- `data/*_draws.csv` are the draw history (the script appends new draws from the NY Open Data feed on each run).
- `data/picks_log.csv` records every guess before its draw and is scored on later runs.
- `output/lottery_workbook.xlsx` is regenerated each run (git-ignored).
