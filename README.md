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

TODO:

 - Add more layers
 - Get rid of post-processing to get proper sld
 - Add full SLD to reconstruction loss
 - Add q resolution
 - Check max thickness and get rid of model and generate another one. 
