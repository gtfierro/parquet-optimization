Install the deps

```
pip install optuna joblib pyarrow pandas optuna-dashboard
```

Then you can run the optimization script

```
python optuna_dataset_optimization.py
```

Whiel that's running, you can open the dashboard to look at some cool data about the optimization (run this command from this directory!)

```
optuna-dashboard sqlite:///optuna.sqlite3
```

