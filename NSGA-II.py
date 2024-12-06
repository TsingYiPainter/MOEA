# 决策变量的下界和上界
import evox
from evox import algorithms, problems, workflows, monitors, use_state
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from entry import EA

lb = jnp.full(shape=(12,), fill_value=0)  # 下界：维度为2，值为 -32
ub = jnp.full(shape=(12,), fill_value=1)   # 上界：维度为2，值为 32


# 初始化 NSGA2
nsga2 = algorithms.NSGA2(
    lb=lb,              # 下界数组
    ub=ub,              # 上界数组
    n_objs=3,           # 三目标优化问题
    pop_size=100,       # 种群大小为 100
)

TestBench = [problems.numerical.MaF1(),
    problems.numerical.MaF2(),
    problems.numerical.MaF3(),
    problems.numerical.MaF4(),
    problems.numerical.MaF5(),
    problems.numerical.MaF6(),
    problems.numerical.MaF7(),
    problems.numerical.MaF8(),
    problems.numerical.MaF9(),
    problems.numerical.MaF10(),
    problems.numerical.MaF11(),
    problems.numerical.MaF12(),
    problems.numerical.MaF13(),
    problems.numerical.MaF14(),
    problems.numerical.MaF15(),
]

for i in range(6):
    EA(TestBench[i],nsga2,"NSGA-II","MaF{}".format(i+1))