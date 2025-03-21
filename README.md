## RSA_factor_search

这是一个基于RSA质因数搜索算法设计的Python程序。

> 1.0版本：小搜索函数采用的是穷举暴力搜索，耗时较长。
>
> 2.0版本：优化小搜索函数，将主搜索方向改为随机漫步搜索，并实现第二阶段的并行化计算。
>
> 3.0版本：优化CPU使用，使其至少保留一个核心用于其他任务。

3.0版本的部分测试数据：

![](/img/1.jpg)

算法的大致内容：（更多内容由于专利版权问题，不便展示）

![](/img/2.png)

> warning: 代码处于开发测试阶段，谨慎使用！！出问题概不负责！！
