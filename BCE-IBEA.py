# 决策变量的下界和上界
import evox
from evox import algorithms, problems, workflows, monitors, use_state
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from entry import EA

lb1_6 = jnp.full(shape=(12,), fill_value=0)  # 下界：维度为2
ub1_6 = jnp.full(shape=(12,), fill_value=1)   # 上界：维度为2

lb7 = jnp.full(shape=(22,), fill_value=0)  # 下界：维度为2
ub7 = jnp.full(shape=(22,), fill_value=1)   # 上界：维度为2

lb8_9 = jnp.full(shape=(2,),fill_value=-10000)
ub8_9 = jnp.full(shape=(2,),fill_value=10000)

lb10_12 = jnp.zeros(shape=(12,))  # 下界：全为 0
ub10_12 = jnp.array([2 * (i + 1) for i in range(12)])  # 上界：每个变量 [0, 2*(i+1)]

# 定义下界和上界
lb_13 = jnp.array([0, 0, -2, -2, -2])  # 下界：前两个变量是 [0, 0]，后三个是 [-2, -2, -2]
ub_13 = jnp.array([1, 1,  2,  2,  2])  # 上界：前两个变量是 [1, 1]，后三个是 [ 2,  2,  2]

lb14 = jnp.full(shape=(60,), fill_value=0)  # 下界：维度为2
ub14 = jnp.full(shape=(60,), fill_value=1)   # 上界：维度为2

lb15 = jnp.full(shape=(60,), fill_value=0)  # 下界：维度为2
ub15 = jnp.full(shape=(60,), fill_value=1)   # 上界：维度为2
# 初始化 NSGA2
bce_ibea_1_6 = algorithms.BCEIBEA(
    lb=lb1_6,              # 下界数组
    ub=ub1_6,              # 上界数组
    n_objs=3,           # 三目标优化问题
    pop_size=100,       # 种群大小为 100
)

# 初始化 NSGA2
bce_ibea_7 = algorithms.BCEIBEA(
    lb=lb7,              # 下界数组
    ub=ub7,              # 上界数组
    n_objs=3,           # 三目标优化问题
    pop_size=100,       # 种群大小为 100
)

bce_ibea_8_9 = algorithms.BCEIBEA(
    lb=lb8_9,              # 下界数组
    ub=ub8_9,              # 上界数组
    n_objs=3,           # 三目标优化问题
    pop_size=100,       # 种群大小为 100
)

bce_ibea_10_12 = algorithms.BCEIBEA(
    lb=lb10_12,              # 下界数组
    ub=ub10_12,              # 上界数组
    n_objs=3,           # 三目标优化问题
    pop_size=100,       # 种群大小为 100
)

bce_ibea_13= algorithms.BCEIBEA(
    lb=lb_13,              # 下界数组
    ub=ub_13,              # 上界数组
    n_objs=3,           # 三目标优化问题
    pop_size=100,       # 种群大小为 100
)

bce_ibea_14= algorithms.BCEIBEA(
    lb=lb14,              # 下界数组
    ub=ub14,              # 上界数组
    n_objs=3,           # 三目标优化问题
    pop_size=100,       # 种群大小为 100
)

bce_ibea_15= algorithms.BCEIBEA(
    lb=lb15,              # 下界数组
    ub=ub15,              # 上界数组
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

for i in range(15):
    if i <= 5:
        #pass
        EA(TestBench[i],bce_ibea_1_6,"BCE-IBEA","MaF{}".format(i+1))
    elif i == 6:
        #pass
        EA(TestBench[6],bce_ibea_7,"BCE-IBEA","MaF7")
    elif i == 7 and i ==8 :
        #pass
        EA(TestBench[i],bce_ibea_8_9,"BCE-IBEA","MaF{}".format(i+1))
    elif i>=9 and i<=11:
        #pass
        EA(TestBench[i],bce_ibea_10_12,"BCE-IBEA","MaF{}".format(i+1))
    elif i == 12:
        #pass
        EA(TestBench[12],bce_ibea_13,"BCE-IBEA","MaF13")
    elif i == 13:
        #pass
        EA(TestBench[i],bce_ibea_14,"BCE-IBEA","MaF{}".format(i+1))
    else:
        #pass
        EA(TestBench[i],bce_ibea_15,"BCE-IBEA","MaF{}".format(i+1))
