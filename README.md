# Timestral playground

A tiny CPU demo for [Mistral's Timestral-5M](https://huggingface.co/mistralai/Timestral-5M).
Feed it past measurements and it predicts future values, including uncertainty.
It needs no training on this example.

```sh
cd tmp/timestral-demo
bash run.sh
```

Requires `uv`. First run installs CPU-only PyTorch and downloads the model.
Dependencies, model weights, and the virtual environment stay in this folder.
Open `forecast.html` in your browser. It is self-contained, with zoom, pan,
an overview slider, and clickable legends. You can also serve it with rala.

The default data simulates three weeks of hourly server traffic, then hides the
following 48 hours from the model. The chart compares its forecast with the real
held-out values and a baseline that repeats the previous day's traffic.
MAE measures forecast error; lower is better. The shaded area is the model's
10th–90th percentile range. Its actual coverage on new data can differ from 80%.
The timing covers one forecast call, excluding model loading and downloads.

```sh
bash run.sh --horizon 96 --noise 12
bash run.sh --csv example.csv --column value --horizon 48
bash run.sh --csv your-data.csv --column temperature --period 24
```

CSV observations must be equally spaced and finite, ordered oldest to newest.
The last `--horizon` rows are withheld for evaluation, so include at least
`horizon + max(32, period)` rows. A step means whatever interval your data uses.
`forecast.csv` contains the held-out actuals, quantiles, and baseline.

Official code: https://github.com/mistralai/timestral
