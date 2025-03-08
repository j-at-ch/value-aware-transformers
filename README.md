# value-aware-transformers

This repo contains the code for the paper [Value-aware Transformers for 1.5D Data](paper.pdf).

## Requirements

We've provided a requirements.txt file specifying the basic packages needed to use this repo. We use a conda env.
Note that Anaconda does not provide a distribution for the entmax package. 
This is accessible on pypi. To bring it into a conda-managed venv
you can add pip to your conda venv. Then install entmax into venv directly via pip. E.g.

`conda create --name <vat-env>`

`conda activate <vat-env>`

`conda install --file requirements.txt`

`anaconda3/envs/<vat-env>/bin/pip install entmax`

For visualisation and logging you will need Tensorboard. Make sure that you do not have tensorflow in your venv. This will confuse tensorboard.
If you wish to see the embeddings, make sure to append model_name to logs, as below.
```
tensorboard --logdir <path_to_desired_log_dir>/model_name
```

## Preprocessing

This relies on having access to (and permission for) the MIMIC-III database files which can be found at:
https://physionet.org/content/mimiciii/1.4/

Further details on preprocessing can be found in ```preprocessing/README.md```

## Pretraining, Finetuning, Baselining

### Pretraining commands

The defaults settings in arguments.py specify the pretraining of a value-aware transformer of depth 4 the same as we
trained in the paper. You will need to specify the following required arguments so that the script knows: where to find
your preprocessed data; where to save checkpointed models; where to write logs to; and what your cuda device is called. 
```
python mimic/pretraining.py \
    --data_root <path_to_preprocessed_data_dir>\
    --save_root <path_to_desired_save_dir>\
    --logs_root <path_to_desired_log_dir>\
    --device <your_cuda_device>
```

Provided you specify the correct `--data_root` (which includes correctly formatted data)
and give a valid `--save_root` and `--logs_root`, the model should run straight out of the box.

*HINT*: to do a test run, append the option `--toy_run 1` to the command above.

To see all specifiable options, type:
```
python mimic/pretraining.py --help 
```
and for further details consult `mimic/arguments.py` which shows the default settings.

### Finetuning

To run finetuning you should have a pretrained model and specify the correct arguments so that the base (va-)transformer
architecture matches the weights being loaded.

You will need to specify these required arguments:
```
python mimic/finetuning.py \
    --data_root <path_to_preprocessed_data_dir>\
    --save_root <path_to_desired_save_dir>\
    --logs_root <path_to_desired_log_dir>\
    --pretrained_model <path_to_ckpt_of_output_of_pretraining>
```

The default prediction problem is ```--targets="DEATH<=3D"```. 
Switching in any of the other mortality targets will work without changing any of the defaults.

For the regression problem ```--targets="LOS"```, the following arguments need to be set:
```--clf_or_reg="reg"```, ```--num_classes=1```.


*HINT*: to do a test run, append the option `--toy_run True` to the command above.

### Baselining1D

To run the baseline1D models you will need 

You will need to specify these required arguments.
```
python mimic/baselining1D.py \
    --data_root <path_to_preprocessed_data_dir>\
    --save_root <path_to_desired_save_dir>\
    --logs_root <path_to_desired_log_dir>
```

The default prediction problem is ```--targets="DEATH<=3D"```. 
Changing in any of the other mortality targets will work without changing any of the default setting.

For the regression problem ```--targets="LOS"```, the following arguments need to be set:
```--clf_or_reg="reg"```, ```--num_classes=1```.

*HINT*: to do a test run, append the option `--toy_run True` to the command above.

### Final Note
Any questions, reach out. I'd be happy to help!
