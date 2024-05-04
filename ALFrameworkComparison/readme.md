# Getting Started

The argument list:

||name|descrption|default|
|----------|----------|----------|----------|
1|`--dataset`|What dataset to train on||
2|`--algorithm`|What active learning algorithm to evaluate from the values {random, bald, coreset, entropy}||
3|`--random_seed`|What random seed to use (int)|42|
4|`--val_share`|What share of unviolated labeled instances to use for validation (float)|0.25|
5|`--n_initially_labeled`|What number of labeled instances to start with (int)|20
6|`-hpo_mode`|from the values {constant, online}|
7|`--split`|{whole, initial, static, dynamic}|

When `-hpo_mode` is constant, `--split` must be whole or initial and When `-hpo_mode` is online, `--split` must be static or dynamic.

