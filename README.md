1. 训练主代码：/Motion_intutive_CPU/2024.12.14_main.py，直接运行这个开始训练。
2. 数据集主要在data/output_dataset1.py  这里面有6w组左右的数据，是以tensor形式存储的py文件。
   由于比较大，训练直接加载有可能现存不够。我现在做法是取前1w组，保存成.pt文件，训练时用torch加载。
3. 数据集构成
![image](https://github.com/user-attachments/assets/068239e3-8a75-4fd3-8b47-7c3ec9bbbaaf)
