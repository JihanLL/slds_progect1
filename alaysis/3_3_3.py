import numpy as np

import matplotlib.pyplot as plt

# 1. 初始化三个数组 (Initialize three arrays)
# 假设我们创建一些示例数据 (Assuming we create some sample data)
x = np.arange(10)  # X轴数据 (X-axis data)
A = x * 2          # 数组 A (Array A)
B = x ** 2         # 数组 B (Array B)
C = 10 * np.sin(x) # 数组 C (Array C)

# 2. 在同一个图中绘制三个数组 (Plot the three arrays on the same graph)
plt.figure(figsize=(8, 6)) # 创建图形 (Create a figure)

plt.plot(x, A, label='Array A (2*x)', marker='o')  # 绘制 A (Plot A)
plt.plot(x, B, label='Array B (x^2)', marker='s')  # 绘制 B (Plot B)
plt.plot(x, C, label='Array C (10*sin(x))', marker='^') # 绘制 C (Plot C)

# 添加图例、标签和标题 (Add legend, labels, and title)
plt.legend()
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Plot of Arrays A, B, and C")
plt.grid(True) # 添加网格线 (Add grid lines)

# 显示图形 (Display the plot)
plt.show()