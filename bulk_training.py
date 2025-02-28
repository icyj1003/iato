import os

Ns = [5, 10, 15, 20]
Ms = [10, 20, 30, 40]
lambdas = [1e-3, 3e-3, 5e-3, 7e-3]
recvs = [1, 3, 5, 7, 9]

default_N = 10
default_M = 20
default_lambda = 3e-3
default_recv = 3

seed = 10003

# training with default parameters
os.system(
    f"python train_maa2c.py --N {default_N} --M {default_M} --lambda_I {default_lambda} --recv {default_recv} --seed {seed}"
)

# training with different N
for N in Ns:
    if N == default_N:
        continue
    os.system(
        f"python train_maa2c.py --N {N} --M {default_M} --lambda_I {default_lambda} --recv {default_recv} --seed {seed}"
    )

# training with different M
for M in Ms:
    if M == default_M:
        continue
    os.system(
        f"python train_maa2c.py --N {default_N} --M {M} --lambda_I {default_lambda} --recv {default_recv} --seed {seed}"
    )

# training with different lambda
for lambda_I in lambdas:
    if lambda_I == default_lambda:
        continue
    os.system(
        f"python train_maa2c.py --N {default_N} --M {default_M} --lambda_I {lambda_I} --recv {default_recv} --seed {seed}"
    )

# training with different recv
for recv in recvs:
    if recv == default_recv:
        continue
    os.system(
        f"python train_maa2c.py --N {default_N} --M {default_M} --lambda_I {default_lambda} --recv {recv} --seed {seed}"
    )
