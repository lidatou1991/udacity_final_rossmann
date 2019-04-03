# udacity_final_rossmann
# 项目简介
graduation project for Udacity machine learning engineer nanodegree  
实际上是 Udacity 采用 Kaggle 的比赛，作为 MLND 纳米学位的毕业项目  
- https://www.kaggle.com/c/rossmann-store-sales  

# 使用的库  
- 数据处理：Pandas，Numpy
- 特征工程：Featuretools
- 算法包装：sk-learn
- 自动化 Pipeline: TPOT

# file description

- /GCP：.ipynb 表示在谷歌云上运行的 notebook；.csv 表示对应模型的结果
- 其余 .ipynb 则是运行在本地 MBP


# 机器硬件及系统  
本地： 2017 MBP 8G
谷歌云： 30 GB RAM

# 运行时间

- TPOT 优化模型时间：在对应 notebook 中的 max_mins 参数中（e.g 120= 2h)
- 本地运行 GradientDescentBoost，＜20 min
- 本地训练 xgboost 时间 154.61秒，GCP 中训练时间在48~62秒
- 对应 notebook：https://github.com/lidatou1991/udacity_final_rossmann/blob/master/%08Auto-Feature-MBP-xgboost.ipynb

# 得分及排名

- 现在得分最高为 pulic score = 0.146, 对应 notebook ：https://github.com/lidatou1991/udacity_final_rossmann/blob/master/%08Auto-Feature-MBP.ipynb

- 增加 feature 后，得分 1.2 非常疑惑， 对应 notebook ：https://github.com/lidatou1991/udacity_final_rossmann/blob/master/GCP/XG-TPOT-GCP-2h-test-Add-Feature.ipynb

- 使用 featurestools + xgboost，但是没有保留date/month 等时间窗 features 的得分 **public score = 0.11937**,基本满足0.1175要求
- 4月2日至今最高得分 **public score = 0.11888**，对应 notebook ：https://github.com/lidatou1991/udacity_final_rossmann/blob/master/%08Auto-Feature-MBP-xgboost-keep-date.ipynb

![Kaggle](https://github.com/lidatou1991/udacity_final_rossmann/blob/master/screenshots/best-score.png)

- 4月3日 private 最高得分 **private score = 0.12794**,对应的 [notebook](https://github.com/lidatou1991/udacity_final_rossmann/blob/master/%08Auto-Feature-MBP-xgboost.ipynb)
![Kaggle](https://github.com/lidatou1991/udacity_final_rossmann/blob/master/screenshots/best-private.png)

# 疑惑与问题

- ~~没有保留时间窗 features 的模型，为什么比保留这些date features的表现反而好这么多~~

- https://github.com/lidatou1991/udacity_final_rossmann/blob/master/xgbm-add-feature.ipynb 为什么得分只有0.58
