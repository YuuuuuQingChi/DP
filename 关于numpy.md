# 1.创建数组
# 1.1使用np.array()
```py
    a = np.array([1,2,4,5])
```
这是生成一维数组
```py
    a = np.array(
        [[1,2,4,5]
        ,[2,3,4,5]]
    )
```
这是生成二维数组
## 1.1.1ndarray的属性
```py
    a.ndim()
```
返回数组的维度，例如返回a是二维数组
## 1.1.2shape
```py
    b=a.shape()
```
返回一个元组，返回数组的维度或者形状，例如返回a是2行4列(b=(2,4))
## 1.1.3dtype
```py
    b=a.dtype()
```
返回数组中的数据类型
## 1.1.4size
```py
    b=a.size()
```
返回数组中的所有元素的个数，例如a中的size应该是8个
## 1.1.5itemsize
```py
    b=a.itemsize()
```
返回数组中的单个元素的所占的字节数

# 1.2使用np.arange——生成等差数列，默认储存一维数组
method:np.arange(start,stop,step,dtype)
start:起始值（默认为0）
stop：终止值
step:步长（默认为1）
dtype:你要的数据类型
## 简单说说
```py
    b=a.arange(10)
```
这个是只有传入stop值，生成从0到10，截至到10之前，即是0-9
## reshape
可以用reshape函数，将数组变化形式
```py
    a = np.arange(1,10).reshape(2，5)
```
这样数组就变成了2维5列

# 1.3使用np.linspace——生成指定数量的等差数列，默认储存一维数组
method:np.linspace(start,stop,num,endpoint)
start:起始值（默认为0）
stop：终止值
num
endpoint:步长（默认为1）
```py
a = np.linspace(1,10,5)
```
这个意思是从1到10生成个等差数列，要求一共有5个数字

# 1.4使用ones/ones_like生成特定数组

## ones
```py
    a = ones(10)
    a = ones((2,5))
```
可以指定数组的形式
## ones_like
```py
    b = ones_like(a) 
```
可以生成与指定数组类型相同的数组

# 1.5使用zeros/zeros_like生成零数组
与ones同理

# 1.6使用empty/empty_like生成未初始化的数组
注意未初始化，意味着值是随机的，并不是0
与ones同理

# 1.7使用full/full_like生成未初始化的数组
```py
    a = np.full((3,4),666)
```
这样会生成3维4列全是666的数组

# 1.8使用random生成随机数组

```py
    a = np.random.rand(5)#生成一维5列的数组，范围是0~1
```
括号里给的参数是生成的数量，默认的数字的生成范围是0~1
当然也可以指定shape
```py
    a = np.random.rand(5，4)#生成一维5列的数组，范围是0~1
```
# 1.9使用randint生成整数数组

``` py
    np.random.randint(2,5)
```
生成一个随机整数  区间范围：[2,5) 左闭右开  包含2不包含5
如果只给定一个参数,那就是从0-参数，左闭右开

``` py
np.random.randint(2,5,(3,5))
```
指定shape,生成3行5列个2·5左闭右开的整数

# 1.10使用uniform生成均匀的数字
```py
np.random.uniform(1,10,指定的shape)
```

# 1.11使用randn生成标准正态分布的数据
```py
np.random.randn(指定的shape)
```

# 1.12使用normal生成标准正态分布的数据
```py
np.random.normal(均值，标准差，指定的shape)
```

# 2.数组的索引
## 2.1一维数组
单值和C++一样
还可以
```py
a = array[2:8]
```
这样可以取到一个区间的，还可以使用-1.-2代表倒数第几个，但是依旧左闭右开
## 2.2二维数组
```py
x[1,1]
```
这个是拿到1行1列的
```py
x[1]
```
这个取一整行
```py
x[:,1]
```
这个取一整列

# 3.数组运算
## 3.1基本运算
A+1
A*3
np.sin(A)
np.exp(A)
**A+B和A*B是对应位置的数相运算，不是线性代数**

## 3.2现成函数
np.sum(A)全部元素求和
np.prod(A)全部元素求积
np.min(A)最小值
np.max(A)最大值
np.median(A)中位数
np.mean(A)平均值
np.std(A)标准差
np.var(A)方差

