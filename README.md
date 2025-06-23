The cityscape dataset is too large and can be downloaded from https://github.com/mcordts/cityscapesScripts
```
1. run train_danet.py to obtain the model weight of the DANet
```

```
2. run train_tanet.py to obtain the model weight of the TANet
```

```
3. run train_dscamsff_tanet.py to obtain the model weight of the proposed model.
```

```
4. run read_data.py to display the trainning logs.
```

```
5. run test.py and modify the weight of different model the see the test results.
modify the weight path in line 95:
example model_path = 'training_data/DSCAMSFF_TANet_QKV_has_VFA/msca_danet_best_epoch_0_mIOU_0.334.pth'
```

