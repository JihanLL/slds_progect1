import numpy as np

import matplotlib.pyplot as plt
from readjason import readjosn

address = []
Key_word = "test_accuracies"

# address.append("../results/3.3.3/2/2025-04-30_22-57-31 2layer/metrics.json")
# address.append("../results/3.3.3/2/2025-04-30_22-16-39 3layer/metrics.json")
# address.append("../results/3.3.3/2/2025-04-30_22-44-50 4layer/metrics.json")

address.append("../results/3.3.3/1/2025-04-30_22-57-31 2layer/metrics.json")
address.append("../results/3.3.3/1/2025-04-30_22-16-39 3layer/metrics.json")
address.append("../results/3.3.3/1/2025-04-30_22-44-50 4layer/metrics.json")



test_accuracies = []

for i in range(1, 4):
    # 读取 JSON 文件 (Read JSON file)
    test_accuracies.append(readjosn(Key_word, address[i - 1]))

# 1. 初始化三个数组 (Initialize three arrays)
# 假设我们创建一些示例数据 (Assuming we create some sample data)
x = np.arange(len(test_accuracies[0]))  # X轴数据 (X-axis data)
acc_1 = test_accuracies[0]  # 数组 A (Array A)
acc_2 = test_accuracies[1]  # 数组 B (Array B)
acc_3 = test_accuracies[2]  # 数组 C (Array C)

# 2. 在同一个图中绘制三个数组 (Plot the three arrays on the same graph)
plt.figure(figsize=(8, 6))  # 创建图形 (Create a figure)

plt.plot(x, acc_1, label="2 layer", marker="o")  
plt.plot(x, acc_2, label="3 layer", marker="s") 
plt.plot(x, acc_3, label="4 layer", marker="^")  

# 添加图例、标签和标题 (Add legend, labels, and title)
plt.legend()
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Plot of Arrays A, B, and C")
plt.grid(True)  # 添加网格线 (Add grid lines)

# 显示图形 (Display the plot)
plt.show()
