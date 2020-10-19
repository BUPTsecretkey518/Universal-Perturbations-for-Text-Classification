Code and data for the paper `Generating Natural Adversarial Examples with Universal Perturbations for Text Classification`

Experimental environment:

```shell
Ubuntu=18.04.1 LTS
GeForce GTX 1080ti
```

Requirements:

```shell
python=3.6
torch=1.5.1
```



Running Instructions:
1.) [Optional] Train and test the target model
`python train_baseline.py` and `python test_baseline.py`

you can also use our pre-trained model in `./models/baseline`

2.) [Optional] Train the ARAE
`python train.py --data_path ./data --update_base --convolution_enc --classifier_path ./models`
you can also use our pre-trained model in `./output/1593075369`

3.) [Optional] Train the Inverter
`python train.py --data_path ./data --load_pretrained <pretrain_exp_ID> --classifier_path ./models`
you can also use our pre-trained model in `./output/1593075369`

4.) Generate adversarial examples
`python attack.py`

If you need other target models, such as BiLSTM or EMB, please contact us `haorangao@bupt.edu.cn`

#### Acknowledgment
Initial code is based on  [Zhengli Zhao et al., 2018](https://github.com/zhengliz/natural-adversary) and [Zhao et al., 2017](https://github.com/jakezhaojb/ARAE)