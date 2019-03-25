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

- **数据探索性分析（Exploratory Data Analysis）**：通过来尝试了解数据的一些基本情况，像数据的分布情况、缺失情况等，为后面的特征工程（Feature - Engineering）做准备和提供一些参考；
- **数据预处理（Data Preprocessing）**: 此过程来处理诸如类别信息、缺省值、时间序列信息等，便于后续的特征提取；
- **通过特征工程（Feature Engineering）**：来最大限度地提取出特征供后续的模型使用；
- **模型选择（Model Selection）**： 利用提取到的特征建立回归模型，在建模的过程中可以尝试通过特征选择和模型融合（feature selection and model ensemble）来争取达到比较好的预测效果。
-**模型调参（Model Tunning)**:最终期望达成的结果是，训练好的模型根据门店的相关信息（比如促销，竞争对手，位置等）和预测日当天以及前后一段时期的节假日等信息，能相对准确地对预测日的销售额进行预测。
### 数据集与输入
_(approx. 2-3 paragraphs)_

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

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
