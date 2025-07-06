# 前言
使用dir函数可以看到package里面有什么东西哦
使用help（）函数可以看到怎么使用这些东西
jupyter notebook启动jupyter 没啥用
# 关于os
os全称为Operating System，这个模块提供了与操作系统交互的各种函数，我们可以通过这些函数调用计算机底层操作系统的部分功能来快速、高效地管理文件和目录。os 库在数据处理中一个比较常见的功能是批量读取文件路径。
os.path.join(self.data_root,self.data_label)手动拼接路径
当我们得到一个数据集时，Dataset类可以帮我们提取我们需要的数据，我们用子类继承Dataset类，我们先给每个数据一个编号（idx），在后面的神经网络中，初始化Dataset子类实例后，就可以通过这个编号去实例对象中读取相应的数据，会自动调用__getitem__方法，同时子类对象也会获取相应真实的Label（人为去复写即可）
# 关于dataset类
ants_dataset = DataTest("hymenoptera_data/train","ants")
bees_dataset = DataTest("hymenoptera_data/train","bees")
traindataset = ants_dataset + bees_dataset
image, label = traindataset[123]
数据集也可以相加，相当于数组拼接
# 关于tensorboard
SummaryWriter 是 TensorBoard 在 PyTorch 中的接口，它能够将训练过程中的数据转化为 TensorBoard 支持的格式进行可视化。首先，需要创建 SummaryWriter 的实例，指定日志文件的存储路径：

添加conda终端到vscode     
"Anaconda Prompt": {
            "path": [
                "${env:windir}\\Sysnative\\cmd.exe",
                "${env:windir}\\System32\\cmd.exe"
            ],
            "args": ["%windir%\\System32\\cmd.exe", "/K", "E:\\miniconda3\\Scripts\\activate.bat","E:\\miniconda3"],
            "icon": "terminal-cmd"
        },
生成文件后想可视化，会采用命令，tensorboard --logdir=logs --port = 6006，logdir的名字必须和你SummaryWriter这个类的名字对应，port是端口号避免冲突
## add_scalar
```python
     writer = SummaryWriter("logs") # 日志文件存储位置
    for i in range(100):
     writer.add_scalar("y = x", i, i) #前一个i是y 后一个i是横坐标
    writer.close()
 ```
## add_scalars 
用来add_scalars() 允许你在 同一个图表（plot） 中记录多个相关的标量值，这些值会共享同一个 global_step（通常是训练步数或 epoch 数）。
在 TensorBoard 中，这些值会显示为 多条曲线，方便对比它们的趋势。
    ```python3
    import numpy as np  # 添加这行导入语句
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter("adada")
    r = 5
    for i in range(100):
    writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                                'tanx': np.tan(i/r)}, i)
    writer.close()
    ```
## 打印图片格式
image_array = np.array(image)
print(image_array.shape)