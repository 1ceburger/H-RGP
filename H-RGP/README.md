# H-RGP

Training and evaluation code for **A differentiable hybrid runoff forecasting framework based on physical process mapping for basins regulated by multiple reservoirs**.

H-RGP combines grid inflow generation, differentiable storage routing and a reservoir release response module. The comparison models are LSTM, LSTM without reservoir inputs, MC-LSTM, differentiable HBV and standard HBV.

[中文说明](README.zh-CN.md) · [Data access](DATA.md) · [Input format](docs/INPUT_FORMAT.md)

## Download and installation

On this repository page, select **Code → Download ZIP** and extract the archive. Alternatively, clone this repository using the URL displayed under **Code**. Open a terminal in the extracted repository directory.

Python 3.10 is supported. Install a PyTorch build suitable for your CPU or CUDA environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
python run.py --action verify
```

Source verification and job planning do not require the study data. Training and numerical evaluation require the model-ready inputs described in [Input format](docs/INPUT_FORMAT.md). The hydrological and reservoir observations are not included or downloadable from this repository. Downloading the public source products alone is insufficient to reconstruct the inputs that depend on those observations.

## Experiments

The lead times are 1, 3 and 7 days, with seeds 0, 1, 2, 3 and 4. H-RGP and the four learning-based comparison models are fitted independently for each lead and seed. The two LSTM models select a 15- or 90-day history by validation NSE. Standard HBV is calibrated once per seed and its continuous simulation is scored on each lead's target dates.

The complete experiment contains 90 model–lead–seed result combinations. Including the LSTM history-window fits gives 110 fits or calibrations across 80 launcher jobs. Settings are recorded in `configs/experiment.json`.

Neural training uses AdamW, a base learning rate of 0.0005, weight decay of 0.01, gradient clipping at 1, a maximum of 160 epochs and early-stopping patience of 20. Rainfall augmentation is disabled. H-RGP uses a 15-day history, eight chronological streams, four-day truncated backpropagation and EMA decay of 0.99. Checkpoint and history-window selection use validation NSE. Differentiable HBV uses CPU state initialization and the supplied HBV parameter initialization.

## Run

Use a separate local directory for authorized input files. The following commands use `data` for inputs and `runs` for generated outputs:

```bash
python run.py --action plan --data data --output runs
python run.py --action check-data --data data
python run.py --action preflight --data data --output preflight --seeds 0 --leads 1 3 7 --device cpu
python run.py --action train --data data --output runs --device cuda
python run.py --action evaluate --evaluate-test --data data --output runs --device cuda
python summarize.py --runs runs --split test --output summary
```

The default action displays the job plan without executing it. Training evaluates validation data by default; evaluation of the study's test dates requires `--evaluate-test`. `--models`, `--leads` and `--seeds` restrict a job plan. Standard HBV always evaluates its continuous simulation at all three leads. Keep preflight and training outputs in separate directories. The summarizer checks common dates and observations and reports means and sample standard deviations over all five seeds.

H-RGP and the LSTM models were fitted with CUDA; MC-LSTM and differentiable HBV use CPU. CPU preflight is available for all models. Numerical reproducibility can depend on the hardware and library build.

## Evaluation scope

The experiments use known historical-date merged rainfall; HBV models also use PET estimated from contemporaneous meteorology. They therefore assess retrospective forecasting under known forcing. H-RGP receives masked observed inflow as an auxiliary training label, whereas the comparison models are supervised by outlet discharge. The evaluation period also informed method development and is not an untouched independent holdout. Five-seed variation measures fitting variability rather than independent generalization.

## Repository contents

- `src/hrgp/`: H-RGP model and training entry point.
- `src/neural_baselines/`: four learning-based comparison models.
- `src/hbv/`: continuous HBV calibration and evaluation.
- `configs/experiment.json`: lead times, seeds and training settings.
- `run.py`: verification, job planning, data checks, preflight, training and evaluation.
- `summarize.py`: metrics and five-seed aggregation.

## License and citation

The source code is provided under the [MIT License](LICENSE). Data products retain their respective providers' terms; the code license does not authorize redistribution of observations. A citation entry is supplied in `CITATION.cff`.
