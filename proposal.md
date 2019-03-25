# Machine Learning Engineer Nanodegree
## 毕业项目 Rossmann Store Sales
李懋  
2019年3月

## 开题报告
_(approx. 2-3 pages)_

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
- **通过特征工程（Feature Engineering）**：因为在写开题报告的时候，已经对问题进行了基本完整的尝试，并且已经在 Kaggle 上提交过测试数据，得到了0.128的分数。因此，回头看来，个人认为此步骤是一个机器学习问题中最为关键的一步（虽然有争议，有部分观点认为模型选择以及调整参数才是最关键）。因为，在最初的尝试中，只局限于训练数据本身带有的 feature，没有对这些feature 进行 aggregation 以及 transformation 等深度挖掘，而是把重心放在了模型选择以及模型调整参数上。实践证明，得到的结果并不是最好，还有很大的改进空间。后来发现，模型选择甚至于模型调参，都已经准备（半）自动化的基础了，很大部分的精力都可以省下来。而自动化工具不能（或者不能完全）代替的这一部分，恰恰就是特征工程，这也是数据科学家的价值所在。虽然 featuretools 等工具也可以完成部分特征工程自动化，但是本项目中也会测试探讨，这些现阶段企图在特征工程部分完成自动化的工具，效果并不是太好，至少针对本项目而言是的（因为训练数据并不是非常复杂的多 table 数据，只是简单的 train 与 store 两个 table）；
- **模型选择（Model Selection）**： 在完成最重要的特征工程后，相当于准备好了喂给机器学习模型的料。我们需要选择一个合适的模型，具体到本问题中就是利用提取到的特征建立回归模型。选择好模型后，我们就需要训练该模型，其目标是在测试集上能够有良好的表现。
- **模型调参（Model Tunning)**:训练好模型后，我们可以在 validation set 上验证模型的表现。然后，通过调整选择模型的参数，提高在 validation set 上的表现。同时，还可以根据模型结果的 feature importance 等，对特征工程中的特征依据对模型表现的重要性，进行选择。最终期望达成的结果是，训练好的模型在 test set 上，也就是根据门店的相关信息（比如促销，竞争对手，位置等）和预测日当天以及前后一段时期的节假日等信息，能相对准确地对预测日的销售额进行预测。
### 数据集与输入
本项目所有数据均可以在 [Kaggle 比赛](https://www.kaggle.com/c/rossmann-store-sales/data)数据介绍页面下载，其中也对数据含义做了介绍。在本项目中，我们可以拿到以csv格式呈现的四张数据表单，它们分别是：

1. train.csv - 包含具体销售额的历史数据训练集；
2. test.csv - 不包含销售额的历史数据测试集；
3. sample_submission.csv - 提交数据预测结果的正确格式样本；
4. store.csv - 门店的额外补充信息；


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

### Solution Statement
_(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### 基准模型
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### 评价指标
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### 项目流程设计
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
