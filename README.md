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

# 对 reviewer 建议的回应
> 实际上你这里的模型还是可以继续优化的，主要是在xgboost的调参这一块和特征这块，你可以进一步的提升：

>- 利用特征 CompetitionOpenSinceYear、CompetitionOpenSinceMonth 以及 新特征 Year、Month 构建新的特征 CompetitionOpen，表示最近竞争对手开业有 多少个月的时长。 利用特征 Promo2SinceYear、Promo2SinceWeek 以及新特征 Year、 WeekOfYear 构造新特征 PromoOpen，表示持续促销开始有多少个月的时长。

在最后的 [notebook](https://github.com/lidatou1991/udacity_final_rossmann/blob/master/%08xgboost-af-final.ipynb)中增加了这些 features，其实在[第一次提交](https://github.com/lidatou1991/udacity_final_rossmann/blob/master/xgbm-add-feature.ipynb)的时候就添加了这些features
>- 模型的调参，尝试使用更小的学习率来学习，比如你可以设置 0.003 左右。

在最后的 [notebook](https://github.com/lidatou1991/udacity_final_rossmann/blob/master/%08xgboost-af-final.ipynb) 也减小了 eta，分数确实有提高，但是训练时间也长了很多。

>- train test 建议先拼到一起做特征工程，这样就会避免一些对齐的问题了。

其实在 notebook 中写明了遇到的这个困难，也写了建议的方法。参考[其他资料](https://github.com/Wang-Shuo/Kaggle-Rossman-Store-Sales/blob/master/preprocess.py)也留意到了这个问题。但是本项目对于新手来说，本来就是踩坑的过程，所以我还是保留了我现在的代码方式，其实本质是记录踩坑的过程。否则，基本就和参考的资料一样了，以后回头看都是完美的结果，但是都是别人的经历

> - 注意public和private leaderboard的区别，public是较近日期的数据集，而private是靠后日期的数据集，我们是要求private leaderboard 的top 10%，对于测试集rmpse为0.11773。
之前确实没有注意到这个区别。我以为 private 是用来 validate，而 public 才是用来排名的。现在最好的 private = 0.12794，稍微还差一点 

> - 另外你使用到了：
'store.MEAN(train.Sales)',
'store.MEAN(train.Customers)',
建议你思考一些，该任务的特殊性，我们是要求预测未来六周的sales，你的特征用到了label相关的信息，这会有一个很严重的问题，相当于你使用未来的信息去训练模型，因此会造成过拟合。（因为你在做label encoding的时候是使用的整个的原始的训练集，而不是先划分出验证集的训练集）
这种类型的label encoding 特征建议初学者先不要使用，这是非常危险的，也造成了你的public和private有较大的gap。

'store.MEAN(train.Sales)',
'store.MEAN(train.Customers)',
不就是单个商店的 average sales/customers 么，确实 sales 和 customers 都是 test 中不包含的 features，但是我看到基本每个 solution 都有添加这些 features。并且有的称之为 proxy features。

[例如](https://github.com/unkn0wnxx/kaggle-rossmann-store-sales/blob/master/train.py) 中的 averagesales

[还有](https://solgirouard.github.io/Rossmann_CS109A/notebooks/feature_engineering.html) 称之为 proxy features

