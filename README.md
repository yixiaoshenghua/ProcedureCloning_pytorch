# Chain of thought Imitation with Procedure Cloning (Pytorch Version)

This repository contains pytorch code for ``Chain of Thought Imitation with Procedure Cloning''.

Please cite this work upon using this library.

## Setup

Install the discrete maze environment:

    cd dice_rl
    pip install -e ./dice_rl

## Run procedure cloning

Run procedure cloning with default maze size:

    source run_train_eval.sh

Change `algo_name` in `run_train_eval.sh` from `pc` (procedure cloning) to `bc` (vanilla behavioral cloning).

## Create additional datasets
To test procecure cloning on addition datasets (e.g., maze of different sizes), create data by changing `create_dataset.sh` and create more discrete maze data:

    source create_dataset.sh

## Tensorboard Results

![](assets/train.png)

![](assets/eval.png)

## Citation

If you find this repository useful, please cite us and the original paper as:

```
@misc{pc_torch24wansh,
  author       = {Shenghua Wan},  
  title        = {ProcedureCloning_pytorch},  
  year         = {2024},  % 年份
  howpublished = {\url{https://github.com/yixiaoshenghua/ProcedureCloning_pytorch}}, 
  note         = {Accessed: YYYY-MM-DD}  % accessed date
}

@article{yang2022chain,
  title={Chain of thought imitation with procedure cloning},
  author={Yang, Mengjiao Sherry and Schuurmans, Dale and Abbeel, Pieter and Nachum, Ofir},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={36366--36381},
  year={2022}
}
```