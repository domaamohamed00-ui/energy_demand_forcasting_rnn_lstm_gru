# Hourly Energy Demand Forecasting: RNN, LSTM and GRU from Scratch

End-to-end time-series project on the **PJM East (PJME) hourly electricity load** dataset. The recurrent cells (RNN, LSTM, GRU) are implemented **by hand** in PyTorch: explicit weight matrices and gate equations, with autograd used only for the backward pass. They are trained to forecast the next 24 hours of demand and compared against simple baselines.

## Highlights

- Vanilla RNN, LSTM and GRU cells written from scratch (no `nn.RNN`, `nn.LSTM` or `nn.GRU`), all plugged into one shared sequence model.
- Careful time-series hygiene: chronological split, scaling fitted on train only, no leakage.
- Honest evaluation: every model is compared to naive and seasonal baselines.
- Cyclical calendar features (hour, day of week, month) improved the GRU by ~12.6% and the RNN by ~9.4% in MAE.

## Dataset

- `PJME_hourly.csv`: hourly load in MW (`PJME_MW`), from 2002-01-01 to 2018-08-03.
- Raw issues found and fixed in `Cell 2`:
  - rows were not in time order,
  - 4 duplicated timestamps (daylight saving), averaged,
  - 30 missing hours, filled by linear interpolation.
- After cleaning: 145,392 regular hourly rows, no NaNs.

## Method

| Step | Details |
|---|---|
| Split | Chronological. Train: up to 2015-12-31. Validation: 2016. Test: 2017 onward. |
| Scaling | Standardisation with mean and std computed on the train set only. |
| Windows | Input: past **168 hours** (one week). Target: next **24 hours** (direct multi-step). |
| Features | Load (scaled) + sin/cos of hour of day, day of week and month (7 features). |
| Model | Hand-written cell, hidden size 64, last hidden state goes through a `Linear(64, 24)` head. |
| Training | Adam (lr 1e-3), MSE loss, gradient clipping at 1.0, batch size 256, up to 15 epochs, early stopping (patience 3) on validation loss. |
| Metrics | MAE, RMSE and MAPE, computed in real MW after inverse scaling. |

### The cells

```
RNN : h_t = tanh(x_t W_x + h_{t-1} W_h + b)

LSTM: i = σ(...)  f = σ(...)  g = tanh(...)  o = σ(...)
      c_t = f * c_{t-1} + i * g
      h_t = o * tanh(c_t)

GRU : r = σ(...)  z = σ(...)  n = tanh(x W_xn + b_xn + r * (h W_hn + b_hn))
      h_t = (1 - z) * n + z * h_{t-1}
```

The LSTM uses a forget-gate bias of 1. The GRU follows PyTorch's gate ordering `[r, z, n]`.

## Results (test set, 24-hour horizon)

Single run (one seed). Lower is better.

| Model | MAE (MW) | RMSE (MW) | MAPE % |
|---|---|---|---|
| RNN + calendar features | **1440.95** | **1999.04** | **4.58** |
| GRU + calendar features | 1443.57 | 2001.21 | 4.60 |
| LSTM + calendar features | 1482.64 | 2078.94 | 4.71 |
| LSTM | 1496.87 | 2087.95 | 4.73 |
| RNN | 1590.67 | 2217.67 | 5.06 |
| GRU | 1651.59 | 2232.42 | 5.33 |
| Baseline: same day yesterday | 2299.07 | 3134.09 | 7.32 |
| Baseline: same day last week | 3514.86 | 4797.51 | 11.00 |
| Baseline: repeat last value | 4472.55 | 5967.02 | 14.66 |

### Takeaways

- Every recurrent model beats every baseline. The best ones reduce MAE by about 37% compared with "same day yesterday".
- Forecast error grows quickly over the first ~8 hours ahead, then flattens out.
- Calendar features helped the RNN and GRU clearly and the LSTM only slightly.
- The three architectures end up within about 3% of each other once calendar features are added.

### Limitations

- Results come from **one random seed**. Differences of a few percent between the architectures should not be read as a real ranking.
- Only the load history and calendar features are used. Weather and holidays are not available in this dataset and are likely the main source of the remaining error.
- Test numbers are reported for transparency. Model selection for deployment should use the validation set.

## Project structure

```
.
├── PJME_hourly.csv
├── step1_load_clean.py      # loading and cleaning
├── notebook.ipynb           # full Colab walkthrough, cell by cell
├── best_24h.pt              # saved checkpoint (weights + scaling stats + config)
└── README.md
```

## How to run

1. Open the notebook in Google Colab (a GPU runtime is optional but faster).
2. Upload `PJME_hourly.csv` when prompted.
3. Run the cells in order: setup, cleaning, EDA, split and scaling, windows, baselines, RNN, LSTM, GRU, comparison, 24-hour forecasting, calendar features.

Requirements: `python`, `numpy`, `pandas`, `matplotlib`, `torch`.

## Roadmap

- [ ] Repeat training over several seeds and report mean ± std.
- [ ] Add weather and holiday features.
- [ ] Learning-rate scheduler to stabilise late-epoch training.
- [ ] Serve the best model with FastAPI and package it with Docker.
- [ ] Monitoring and deployment.
