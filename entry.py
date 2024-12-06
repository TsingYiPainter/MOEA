import evox
from evox import algorithms, problems, workflows, monitors, use_state

import jax.numpy as jnp
from jax import random
import re
import io
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def EA(mo_problem,mo_algorithm,algo_name,benchName):
    key = random.PRNGKey(42)
    monitor = monitors.EvalMonitor(multi_obj = True)
    workflow = workflows.StdWorkflow(
        mo_algorithm,
        mo_problem,
        monitors=[monitor],
    )

    # init the workflow
    state = workflow.init(key)
    # run the workflow for 5 steps
    for i in range(3001):
        state = workflow.step(state)
    print("over")

    buffer = io.StringIO()
    sys.stdout = buffer  # 重定向标准输出到 buffer
    print((state.query_state("algorithm")))
    #print(use_state(monitor.get_pf_fitness)(state))
    sys.stdout = sys.__stdout__
    state_output = buffer.getvalue()
    # 提取 fitness 部分的内容
    fitness_data = re.search(r"'fitness': Array\((\[\[.*?\]\])", state_output, re.S)
    if fitness_data:
        fitness_str = fitness_data.group(1)  # 提取出 fitness 的字符串部分
        #print("Extracted Fitness String:\n", fitness_str)

        # 将字符串转换为 NumPy 数组
        fitness_array = np.array(eval(fitness_str))
        print("\nFitness as NumPy Array:\n", fitness_array)
    else:
        print("Fitness data not found.")


    # 提取 x, y, z 坐标
    x = fitness_array[:, 0]
    y = fitness_array[:, 1]
    z = fitness_array[:, 2]

    # 创建三维图形
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    points = ax.scatter(xs=x,    # x 轴坐标
                    ys=y,    # y 轴坐标
                    zs=z,    # z 轴坐标
                    zdir='z',    # 
                    c='black',    # color
                    s=70,    # size
                    )

    # 设置坐标轴标题和刻度
    ax.set(xlabel='X',
        ylabel='Y',
        zlabel='Z',
        xticks=np.arange(0,math.ceil(max(x)),max(math.ceil(max(x))/4,0.2)),
        yticks=np.arange(0,math.ceil(max(y)),max(math.ceil(max(y))/4,0.2)),
        zticks=np.arange(0,math.ceil(max(z)),max(math.ceil(max(z))/4,0.2))
        )

    # 调整视角
    angle = 45
    if benchName=="MaF6":
        angle =75
    ax.view_init(elev=20,    # 仰角
                azim=angle    # 方位角
            )
    ax.set_title('3D Visualization of Points')

    plt.savefig("./res/"+algo_name+"_"+benchName+".png")
    # 显示图形
    plt.show()






    


