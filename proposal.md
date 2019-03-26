# Machine Learning Engineer Nanodegree
## 毕业项目 Rossmann Store Sales
李懋  
2019年3月

## 开题报告

### 背景介绍
项目作为2018年秋季至2019年春季，Udacity 优达学城机器学习工程师（进阶）毕业项目。实际上，
该项目源自 2015 年 Kaggle 的比赛——Rossmann Store Sales。Rossmann 是一家来自于德国，商店遍布多个欧洲国家的一家大型连锁药店。
在该比赛中，Rossmann 提供了有关店铺的历史数据，希望参赛者可以提出自己的模型，帮助 Rossmann 管理人员对各个门店未来
六周的销售额进行预测，因为，对药店管理者而言，准确可靠的销售额预测能让公司提前进行合理的资源分配，从而提高资源的配置效率。
而门店的销售额，本身会受很多因素的影响，该项目的目标就是要基于 Rossmann 各个门店的信息（比如日期/星期、竞争者远近/存在时长、国家/学校节假日等）
以及过往的销售情况来建模预测未来六周的销售额。项目使用到的数据集为编号 ID 为 1-1115的共 1115 个 Rossmann 门店的历史销售记录和这些门店的相关信息。

### 问题称述
项目要求预测的是对 Rossmann 部分门店未来六周的销售额。销售额是一系列连续数值，因而按照机器学习对问题的分类方法，该项目属于回归问题，那么项目要完成的任务就是从所给的数据中提取出（或者构造出）可能对销售额有影响的特征，建立有效的回归模型进行预测。同时，历史数据中是带有日期，需要预测的是历史数据后连续的一段时间销售量。因此具体来说，该项目又可以细分为时间序列回归问题，与其他机器学习问题类似，完成本项目主要可以分为以下几步：

- **数据探索性分析（Exploratory Data Analysis）**：通过对数据进行探索性分析，得到对问题提供的历史数据的第一印象。在此过程中可以解数据的一些基本情况，像数据的分布情况、缺失情况等，为后面的数据预处理（Data Preprocessing）以及特征工程（Feature - Engineering）做准备准备和提供一些参考；
- **数据预处理（Data Preprocessing）**: 此过程是在数据探索性分析的基础上，来对数据进行清洗，以便提取特征以及训练模型。具体来讲，就是需要在此步骤中来处理异常值、缺失值，以及诸如类别信息、时间序列信息等非数值信息等，便于后续的特征提取；
- **通过特征工程（Feature Engineering）**：因为在写开题报告的时候，已经对问题进行了基本完整的尝试，并且已经在 Kaggle 上提交过测试数据，得到了0.128的分数。因此，回头看来，个人认为此步骤是一个机器学习问题中最为关键的一步（虽然有争议，有部分观点认为模型选择以及调整参数才是最关键），并且这一观点在阅读 Kaggle 优胜者的 discussion 后也得到了证实，因为基本上每位排名靠前的选手，都在已有 feature 基础上，构造出了非常多的新的特征，这直接决定了排名先后[^1]。因为，在最初的尝试中，只局限于训练数据本身带有的 feature，没有对这些feature 进行 aggregation 以及 transformation 等深度挖掘，而是把重心放在了模型选择以及模型调整参数上。实践证明，得到的结果并不是最好，还有很大的改进空间。后来发现，模型选择甚至于模型调参，都已经准备（半）自动化的基础了，很大部分的精力都可以省下来。而自动化工具不能（或者不能完全）代替的这一部分，恰恰就是特征工程，这也是数据科学家的价值所在[^2]。虽然 featuretools 等工具也可以完成部分特征工程自动化，但是本项目中也会测试探讨，这些现阶段企图在特征工程部分完成自动化的工具，效果并不是太好，至少针对本项目而言是的（因为训练数据并不是非常复杂的多 table 数据，只是简单的 train 与 store 两个 table）；
- **模型选择（Model Selection）**： 在完成最重要的特征工程后，相当于准备好了喂给机器学习模型的料。我们需要选择一个合适的模型，具体到本问题中就是利用提取到的特征建立回归模型。选择好模型后，我们就需要训练该模型，其目标是在测试集上能够有良好的表现。
- **模型调参（Model Tunning)**:训练好模型后，我们可以在 validation set 上验证模型的表现。然后，通过调整选择模型的参数，提高在 validation set 上的表现。同时，还可以根据模型结果的 feature importance 等，对特征工程中的特征依据对模型表现的重要性，进行选择。最终期望达成的结果是，训练好的模型在 test set 上，也就是根据门店的相关信息（比如促销，竞争对手，位置等）和预测日当天以及前后一段时期的节假日等信息，能相对准确地对预测日的销售额进行预测。
### 数据集与输入
>在开题报告阶段充分理解你的数据是非常必要的，包括以下几个方面：

>1. 对于数据表项目进行具体的介绍；
>2. 对于缺省值以及异常值的分析；
>3. 对于数据集的一个整体的介绍，例如训练集、测试集大小，时间跨度等等；
>4. 对于你的预测标签进行一个简单的分析，例如可视化标签的分布；

本项目所有数据均可以在 [Kaggle 比赛](https://www.kaggle.com/c/rossmann-store-sales/data)数据介绍页面下载，其中也对数据含义做了介绍。在本项目中，我们可以拿到以csv格式呈现的四张数据表单，它们分别是：

1. train.csv - 包含具体销售额的历史数据训练集；
2. test.csv - 不包含销售额的历史数据测试集，**尤其还要注意不包含 Customers,因为顾客数量也无法事先预测**；
3. sample_submission.csv - 提交数据预测结果的正确格式样本，**其中的 Id 是为了比对结果方便，并不是特征**；
4. store.csv - 门店的额外补充信息；

> 训练集包含 1017209 行数据，测试集包含 41088 行数据。训练集的时间跨度为   ，一共   天；测试集的时间跨度为   ，一共   天。

为了完成项目，需要运用 train.csv 和 store.csv 所提供的信息来对模型进行训练，test.csv 是测试集文件，在预测模型训练完成后，我们需要对 test.csv 中的样本进行预测，并将预测结果依照 sample_submission.csv 中的格式进行整理。最终的结果评价交到 Kaggle, 系统自动进行评分。train.csv 以及store.csv 数据集中提供的补充信息的数据域含义如下：

- Id - 一例特定日期和特定门店的样本。
- Store - 各门店唯一的编号
- Sales - 销售额（本项目的预测内容）。
- Customers - 日客流量。
- Open - 用来表征商店开张或闭店的数据，0 表示闭店，1 表示开张。
- StateHoliday - 用来表征法定假期。除了少数例外，通常所有门店都会在节假日关闭。值得注意的是，所有学校在法定假期以及周末都会关闭。数据 a 表示公共假期，b 表示复活节，c 表示圣诞节，0 则意味着不是假期。
- SchoolHoliday - 用来表征当前样本是否被学校的关闭所影响，也可以理解为学校放假。
- StoreType - 使用 a,b,c,d 四个值来表征四种不同类型的商店
- Assortment - 表征所售商品品类的等级，a 为基础型，b 为大型，c 为特大型。
- CompetitionDistance - 距离最近竞争商家的距离（m）。
- CompetitionOpenSince[Month/Year] - 距离最近竞争商家的开业时间。
- Promo - 表征某天是否有促销活动。
- Promo2 - 表征门店是否在持续推出促销活动
- Promo2Since[Year/Week] - 以年和年中周数表征该门店参与持续促销的时间。
- PromoInterval - 周期性推出促销活动的月份，例如 "Feb,May,Aug,Nov" 表示该门店在每年的 2 月 5 月 8 月和 11 月会周期性的推出促销活动。

>训练集中，销售额的分布如下图所示。可以看到包含了很多 open =1，但是 sales =0 的数据，认为这些数据是异常值，在训练集中予以剔除。
![训练集销售额/顾客分布](https://github.com/lidatou1991/udacity_final_rossmann/blob/master/fig/train-hist.png）

>训练集中，一共剔除了 条销售额为 0 的数据。对于其他异常值，按照以下方式处理：
>1. StateHoliday 中的数字 0 与字符’0‘，全部更改为字符'0’；
>2. store.csv 中的 NaN 全部填充为0，表示不存在竞争者，当然 CompetitionDisdance 等相关 feature 也为 0；
>3. test.csv 中的 NaN 全部集中在 Open 字段，因而全部填充为 1，不然不开门的商店没有预测的意义。

>异常值与空缺值处理后，训练集的销售额的分布如下图所示：




### 方案称述

作为机器学习工程师纳米学位的毕业项目，本项目中，我不仅希望对所学知识进行回归，选择到解决方案以期达到项目要求。同时，还希望能够尽量吸收新知识，
对 Udacity 视频材料中没有提到的方法、工具可行探索。具体到解决本项目的解决方案，主要包括以下几点：

1. 特征工程中，学习使用 featuretools 库。featuretools 是一个开源的机器学习特征提取、构造库，能够发掘出许多深度的数据特征。尤其是，当机器学习工程师，对所解决的问题领域不熟悉的时候，很难深度发掘训练数据的特征，featuretools 就大有用武之地。本项目中，使用该工具是为了探索是否能够让特征工程的流程自动化，以及能否有效提高本项目的得分；

2. 模型选择中，学习使用 TPOT 自动化 pipeline 过程。 TPOT 是一种基于遗传算法的机器学习模型选择、调参自动化工具，在此方案中选择此方法的目的是，优化模型选择以及调参过程。

此外，本项目解决方案中的方法与其余大多数 Kaggle 参赛者相同。在最初的时候，使用了决策树（因为训练数据中有许多 categorical feature，自觉决策树应该效果不错）来尝试性打通项目的 pipeline，然后又尝试了在 adaboost 模型上的表现。最后，采用了在众多 Kaggle 优胜案例中使用的 xgboost.

### 基准模型

为了检验选择模型的表现，也就是在测试集上，预测模型是否达标，应当建立一个基准模型，通过是否满足基准指标来判断我有没有完成一个较为合理的预测。本项目中，预测值得正确结果类似于一个“黑箱”，也就是所有参赛队员都不知道最终的正确结果。在提交个人结果后，系统对自动比对与正确值的偏差，然后返回得分。很方便的是，本项目在 Kaggle 上的竞赛已经完成，Kaggle 通过测试集 RMPSE （Root Mean Percent Square Error) 的大小对参赛个人和团队进行了排名（Rossmann Store Sales Leaderboard）。从排名表中，我们可以看到，共有 3303 名参赛队伍进行了预测，第一名的得分为 0.10021（越低越好。根据 udacity 的要求，我将以 leaderboard 的 top 10% 作为基准，也就是对于测试集的评分达到 0.11773。


### 评价指标

在机器学习问题中，学习模型算法的选择以及算法参数的调试对结果的影响是极大的，但这并不意味着我们只需要完成算法相关的工作。在研究中，我们会选择数据集整体的一部分作为训练集 (training set)，另一部分作为测试集 (validation set)。我们在训练集上应用我们的算法训练我们选择的模型，而测试集则作为”期末考试”来对研究成果进行测试和评价。那么算法运行到什么程度以及我们最终结果是否理想都需要一个量化指标来体现，这也就是评价指标。不同的机器学习任务有着不同的性能评价指标。例如，在垃圾邮件检测系统中，它本身是一个二分类问题（垃圾邮件 vs 正常邮件），可以使用准确率 (Accuracy)。本项目实质上是一个时间序列数据预测问题，由于误差判断过程中使用的是百分比，因此相对与更加常用的 RMSE，RMSPE 对于数值的绝对大小不敏感，更加适合于多尺度规模的序列评测，也就是本项目的任务。

在本项目中，我们使用 RMSPE 也就是 Root Mean Square Percentage Error 作为模型的评价指标。RMSPE 的计算方式如下： 
$$ RMSPE=\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\frac{y_i-y_\hat{i}}{y_i}\right)^2} $$

### 项目流程设计
>非常棒，给出了宏观上的一个流程图，不过这里你需要更加详细点：

>1. 你准备采用的数据集划分方式，是随机划分数据集还是按照日期时间顺序。
>2. 你准备采用的模型，例如可以使用 xgboost、lightgbm；
>3. 模型优化部分，准备从什么角度入手（调参，特征工程，使用模型融合等）
>4. 模型评价部分，除了测试集的 rmspe 值，还可以从特征重要性表等角度出发；

本项目流程图如下所示，其中最关键的步骤就是特征工程以及根据模型测试结果，调整特征工程中选择的特征以及调整选择的模型及参数。这两部分的精力应该分别至少花到50%及40%。但是，遗憾的是，由于最初的重心放错以及代码工程能力的不足，有非常多的时间花费在了 pipeline 打通这些本不应该花费太多时间的事情上。

![流程图](https://github.com/lidatou1991/udacity_final_rossmann/blob/master/fig/Untitled%20Diagram.png)

1. 对于数据集划分方式，由于本问题明显是时间序列问题，因此准备参考[^3]采用日期时间顺序;
2. 因为 xgboost 框架在 Kaggle 比赛中表现非常优异，因而准备尝试 xgboost 框架。其实之前在尝试中使用 TPOT 就发现，gradient descent boost 计算量非常大，TPOT 模型优化需要非常多的时间，很有可能在 max_min 时间内并不能找到最优化的模型。因此，这里准备直接使用 xgboost 而不是计算速度更慢的 gradient descent boost. 网上查阅资料后发现，lightgbm 的计算速度更快，因此，也希望能够在本项目中进行尝试；
3. 因为在 TPOT 模型的尝试中，已经意识到特征工程的重要性，并且在撰写本开题报告的时候，已经做了足够多的特征工程。因此，在截止期前，希望更多的精力放在模型调参上面。毕竟，之前也没有使用过 xgbm，对相关参数都还比较陌生。
4. 特征工程中，添加了很多 feature，但是现在并不知道是不是添加的 feature 都对模型表现有重要作用。因此，研究 feature importance 十分必要。删除掉部分不重要的 feature，很有可能还会提高模型的计算速度以及得分表现。

### 项目总结及疑问

撰写本开题报告的时候，已经基本对该项目有一次完整的探索了，虽然最后的结果还暂时没有达到要求。但是，希望在做本次提交的时候，提出以下问题，希望可以得到助教的解答，以便能够在规定期限内完成本纳米学位：

1. 在这份[笔记本](https://github.com/lidatou1991/udacity_final_rossmann/blob/master/%08Auto-Feature-MBP.ipynb)中，使用featuretool只是做了简单的特征工程，模型未做任何的调参，分数已经可以得到0.14，我是否应该在这个笔记本基础上继续优化，完成项目？

2. 为了提高分数，我在上面笔记本的基础上，增加了很多“时间窗”特征，这些特征应该都是保险的、对模型预测结果有益的，但是非常诡异，得分居然只有1.2！在adaboost 模型上，不做任何优化的得分都有0.16的，所以肯定有很明显的错误。
希望助教能够看一下这份 [笔记本](https://github.com/lidatou1991/udacity_final_rossmann/blob/master/GCP/XG-TPOT-GCP-2h-test.ipynb)，其他的 gradient descent regressor 的参数，来自[笔记本](https://github.com/lidatou1991/udacity_final_rossmann/blob/master/Rossmann-Auto-ML-TPOT-Full.ipynb),由于计算资源问题，TPOT 只运行了2小时，所以我怀疑是不是 gradient descent regressor 没有输出最优化的参数（如果特征工程没有错误的话，肯定问题就在这里了）

3. 在第二步的基础上，最初是怀疑特征工程中，增加单一商店的（weekday，avgsales）特征有误，后面发现并不是。我对 TPOT 输出的模型产生了怀疑，因为我修改 TPOT 数据的模型中的一些参数后，在相同特征上的测试分数，提高一倍。

### 参考资料
[^1]:https://www.kaggle.com/c/rossmann-store-sales/discussion/17896#101318
[^2]:https://colab.research.google.com/drive/1CIVn-GoOyY3H2_Bv8z09mkNRokQ9jlJ-
[^3]:http://imagine4077.github.io/Hogwarts/machine_learning/2016/05/13/TIME_SERIES_FORECASTING_-_TAKING_KAGGLE_ROSSMANN_CHALLENGE_AS_EXAMPLE.html

### 数据量较大，因而训练一次，debug 的等待时间确实很长。所以希望助教指出问题在哪。

第一次使用 github，所以 repo 比较乱，请见谅。 

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
