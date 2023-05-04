# Bayes ML model to predict SLD distribution

Create the environment:

```
conda env create -f play_env.yml
source activate playground
```

Create a training set:

```
python train.py  -n 100000 -v 1000 -f config-erik.json --create
```


Train the ML model:

```
python train.py  -n 100000 -v 1000 -f config-erik.json
```
