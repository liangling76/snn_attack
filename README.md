# snn_attack

This code apllies gradient based SNN attack on MNIST dataset

## Train an SNN model
```
CUDA_VISIBLE_DEVICES=0 python train_mnist_ce.py
```

## Collect samples from each classes (optional)

We have already selected 5 images for each class and saved these samples in file ckpt. The following command is used to select data.
```
python dataset_select.py
```

## Apply SNN attack
```
CUDA_VISIBLE_DEVICES=0 python snn_attack.py
```

In file 'snn_attack.py', 'attack_mode' indicates targeted or untargeted attack, 'thresh_attack_value' controls the firing threshold in pernultimate layer. 

For the boundary attack, please see the original [source code!](https://github.com/greentfrapp/boundary-attack) for more detail. Note that the boundary attack does not have any constrain on the model.

## Results
```
untargeted attack; pernultimate layer firing thresh = original firing threshold = 0.3
'succ': 46, 'fail': 4, 'wrong': 0, 'ratio': 0.92, 'avg_diff': 0.024219149865372026

untargeted attack; pernultimate layer firing thresh = 1.0
'succ': 50, 'fail': 0, 'wrong': 0, 'ratio': 1.00, 'avg_diff': 0.020177705427631736

targeted attack; pernultimate layer firing thresh = original firing threshold = 0.3
'succ': 223, 'fail': 218, 'wrong': 1, 'ratio': 0.5056689342403629, 'avg_diff': 0.03405453556221429

targeted attack; pernultimate layer firing thresh = 1.0
'succ': 413, 'fail': 37,  'wrong': 0, 'ratio': 0.9177777777777778, 'avg_diff': 0.033113396922735266
```

## Paper link
https://ieeexplore.ieee.org/document/9527394

DOI 10.1109/TNNLS.2021.3106961

