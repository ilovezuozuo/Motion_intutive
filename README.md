1. 训练主代码：/Motion_intutive_CPU/2024.12.14_main.py，直接运行这个开始训练。
2. 数据集主要在data/output_dataset1.py  这里面有6w组左右的数据，是以tensor形式存储的py文件。
   由于比较大，训练直接加载有可能现存不够。我现在做法是取前1w组，保存成.pt文件，训练时用torch加载。
3. 数据集构成
![image](https://github.com/user-attachments/assets/4e73a72e-b69d-4e91-b41b-835d7c87c80e)
4. 可微计算都写在differentiable_computation_engine下了
5. rrt-algorithms是我改的别人的rrt算法，它本身是个仓库我都删掉了，但是目前push不上来，我查查为什么
