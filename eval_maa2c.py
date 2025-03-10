import os

Ns = [5, 10, 15, 20, 25, 30]
Ms = [5, 10, 20, 30, 40, 50]
lambdas = [1e-2, 2e-2, 3e-2, 4e-2]
recvs = [5, 10, 15, 20, 25, 30]

default_N = 5
default_M = 20
default_lambda = 2e-2
default_recv = 10

seed = 1002

# mode
train_default = True
train_no_interruption = True
train_N = True
train_M = True
train_lambda = True
train_recv = True

# training with default parameters
if train_default:
    os.system(
        f"python train_maa2c.py --N {default_N} --M {default_M} --lambda_I {default_lambda} --recv {default_recv} --seed {seed} --training_steps 100000 --eval_steps 3000"
    )
    if train_no_interruption:
        os.system(
            f"python train_maa2c.py --N {default_N} --M {default_M} --lambda_I {0} --recv {default_recv} --seed {seed} --training_steps 100000 --eval_steps 3000"
        )

# training with different N
if train_N:
    for N in Ns:
        if N == default_N:
            continue
        os.system(
            f"python train_maa2c.py --N {N} --M {default_M} --lambda_I {default_lambda} --recv {default_recv} --seed {seed} --training_steps 100000 --eval_steps 3000"
        )

# training with different M
if train_M:
    for M in Ms:
        if M == default_M:
            continue
        os.system(
            f"python train_maa2c.py --N {default_N} --M {M} --lambda_I {default_lambda} --recv {default_recv} --seed {seed} --training_steps 100000 --eval_steps 3000"
        )

# training with different lambda
if train_lambda:
    for lambda_I in lambdas:
        if lambda_I == default_lambda:
            continue
        os.system(
            f"python train_maa2c.py --N {default_N} --M {default_M} --lambda_I {lambda_I} --recv {default_recv} --seed {seed} --training_steps 100000 --eval_steps 3000"
        )

# # training with different recv
if train_recv:
    for recv in recvs:
        if recv == default_recv:
            continue
        os.system(
            f"python train_maa2c.py --N {default_N} --M {default_M} --lambda_I {default_lambda} --recv {recv} --seed {seed} --training_steps 100000 --eval_steps 3000"
        )
