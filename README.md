#### 风速区间预测



1. 使用MSA(Modified Scaling Approach)构造区间，(1-beta, 1+alpha) * 原始数据直接构造区间

2. 人工构造beta、alpha存在误差，使用多目标优化算法构造区间

3. 优化预测PICP、PINAW(预测的宽度尽量小，覆盖率尽量大，同时让覆盖率尽量大于90%，修改多目标优化算法，使得PICP尽量大于90%，如果PICP不大于90%，直接返回1，否则返回1-PICP，稍微构造了一下)

4. 修改了GRU，使得输出直接就是预测区间，同时修改了loss函数，loss函数为mean((low - low_pre)^2 + (high - high_pre)^2)

   

最后的效果不错，反正，比直接GRU和quantile效果好

