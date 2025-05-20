# EDI Datamapping and Processing

# ⚠️ 现实中会遇到的两个核心问题

## ① 找不到数据（Data Discovery 问题）

### ❗问题表现：
业务想查客户A最近一次退货是什么产品，退货金额是多少，但：
- SAP 里查不到退货原因
- Oracle 里只记录了发货，没有退货信息
- CRM 里记录了退货，但用的是客户邮箱，不是客户ID

### ✅ 你该做的：
- 建立统一的数据目录（Data Catalog），比如：
  - 客户主数据：统一一个客户ID（Customer_ID）
  - 产品主数据：建立产品统一映射表（Product Mapping）
- 使用数据探查工具来发现所有数据落点，例如：
  - Power BI
  - Azure Data Catalog
  - Dataiku
  - Alation

---

## ② 数据无法整合（Data Integration 问题）

### ❗问题表现：
想分析客户从下单到退货的完整路径，但订单在 SAP，发货在 Oracle，售后在 CRM，没法连起来。

### ✅ 你该做的：
1. **构建主数据统一映射关系（Master Data Mapping）**
   - 建立 Customer_ID、Product_Code 的对应关系表
   - 建立统一的“时间戳”字段以关联不同系统的事件流

2. **建立中间层数据仓库或数据湖**
   - 用 Azure Synapse / Snowflake / Databricks 构建一个统一的数据整合层（Data Lake 或 Data Warehouse）
   - 每个系统的数据通过 ETL/ELT 工具（如 ADF / Fivetran / Informatica）导入并进行清洗转换

3. **做 Customer Journey Mapping**
   - 整合事件流（event stream）：订单 → 付款 → 发货 → 客户投诉 → 退货


## Overview

This repository is designed to handle Electronic Data Interchange (EDI) mapping and processing tasks, with a specific focus on integrating AI-powered tools for data manipulation. The project leverages multiple components for different functionalities, including language graph processing and advanced data handling via libraries like PandasAI.

## Structure

### `data`

This folder contains all the data files used by the system. The data is crucial for mapping and processing tasks.

### `demo`

The `demo` folder contains the main code, which is split into two sections:
1. **Langraph**: This section utilizes a local Large Language Model (LLM) for specific processing tasks.
2. **PandasAI_3 (now replaced by `pandas-ai`)**: Originally empowered by PandasAI_3, this section has transitioned to using the updated `pandas-ai` library (v3 beta) to perform advanced data manipulation tasks. Note that `Pandas_ai` is a replacement for `demo/pandasai_3`.

### `pandas-ai-v3`

This folder contains the beta version of the `pandas-ai` library. The open-source library enables seamless integration with pandas DataFrames and allows users to apply AI techniques for data processing.

### `pandas_ai`

This folder is a replacement for the deprecated `demo/pandasai_3` directory. It contains the updated version of the codebase using `pandas-ai`.

## Requirements

To run the project, you'll need the following dependencies:
- Python 3.10 or less than that
- Required packages listed in `requirements.txt`

## Installation

To install the dependencies, simply run the following command:
```bash
pip install -r requirements.txt
