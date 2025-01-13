#!/bin/bash

echo '------- Delete Incomplete on CranfieldU --------'

python -m main_sweep\
       --command delete_incomplete\
       --command_launcher plain\
       --n_trials 3\
       --n_hparams 20\
       --datasets CranfieldU\
       --label_noise_type sym\
       --label_noise_rate 0.2\
       --data_dir=./datasets/CranfieldU\
       --algorithms ERM Mixup IRM  EIRM\
       --erm_losses CELoss NCELoss\
       --skip_model_save

echo '------- Launch on CranfieldU --------'

python -m main_sweep\
       --command launch\
       --command_launcher plain\
       --n_trials 3\
       --n_hparams 20\
       --datasets CranfieldU\
       --label_noise_type sym\
       --label_noise_rate 0.2\
       --data_dir=./datasets/CranfieldU\
       --algorithms ERM Mixup IRM EIRM\
       --erm_losses CELoss NCELoss\
       --skip_model_save

echo '-------  Done on CranfieldU --------'
