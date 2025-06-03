# EDI Datamapping and Processing  
# EDI 数据映射与处理
#### powered by [pandas-ai](https://github.com/sinaptik-ai/pandas-ai)
---

# ⚠️ 现实中会遇到的两个核心问题  
# ⚠️ Two Core Issues Commonly Encountered in Real-World Scenarios

## ① 找不到数据（Data Discovery 问题）  
## ① Cannot Find the Data (Data Discovery Issue)

### ❗问题表现：  
### ❗Problem Description:
- SAP 里查不到退货原因  
  - SAP doesn't show the return reason  
- Oracle 里只记录了发货，没有退货信息  
  - Oracle only records the shipment, no return data  
- CRM 里记录了退货，但用的是客户邮箱，不是客户ID  
  - CRM has the return record, but uses customer email instead of customer ID  

### ✅ 应该做的：  
### ✅ What You Should Do:
- 建立统一的数据目录（Data Catalog），比如：  
  - Create a unified **Data Catalog**, for example:
  - 客户主数据：统一一个客户ID（Customer_ID）  
    - Customer Master Data: unify using a single `Customer_ID`  
  - 产品主数据：建立产品统一映射表（Product Mapping）  
    - Product Master Data: create a **Product Mapping Table**

- 使用数据探查工具来发现所有数据落点，例如：  
  - Use **Data Discovery Tools** to find all data sources, such as:
  - Power BI  
  - Azure Data Catalog  
  - Dataiku  
  - Alation

---

## ② 数据无法整合（Data Integration 问题）  
## ② Data Cannot Be Integrated (Data Integration Issue)

### ❗问题表现：  
### ❗Problem Description:
- 想分析客户从下单到退货的完整路径，但订单在 SAP，发货在 Oracle，售后在 CRM，没法连起来。  
  - You want to analyze the full path from order to return, but the order is in SAP, shipping is in Oracle, and after-sales is in CRM — they can't be connected.

### ✅ 你该做的：  
### ✅ What You Should Do:
1. **构建主数据统一映射关系（Master Data Mapping）**  
   1. **Create Unified Master Data Mapping Relationships**
   - 建立 Customer_ID、Product_Code 的对应关系表  
     - Build relationship tables for `Customer_ID` and `Product_Code`  
   - 建立统一的“时间戳”字段以关联不同系统的事件流  
     - Create a unified **timestamp** field to relate events across systems  

2. **建立中间层数据仓库或数据湖**  
   2. **Build an Intermediate Data Warehouse or Data Lake**
   - 用 Azure Synapse / Snowflake / Databricks 构建统一数据整合层  
     - Use **Azure Synapse**, **Snowflake**, or **Databricks** to build a unified data integration layer  
   - 每个系统的数据通过 ETL/ELT 工具导入并清洗转换  
     - Ingest and clean data using ETL/ELT tools like **ADF**, **Fivetran**, or **Informatica**

3. **做 Customer Journey Mapping**  
   3. **Build a Customer Journey Mapping**
   - 整合事件流：订单 → 付款 → 发货 → 客户投诉 → 退货  
     - Combine event streams: Order → Payment → Shipment → Complaint → Return

---

## Overview  
## 项目概述

This repository is designed to handle Electronic Data Interchange (EDI) mapping and processing tasks, with a specific focus on integrating AI-powered tools for data manipulation.  
该项目旨在处理电子数据交换（EDI）中的数据映射和处理任务，重点是集成 AI 驱动的数据处理工具。

The project leverages multiple components for different functionalities, including language graph processing and advanced data handling via libraries like PandasAI.  
该项目使用多个组件来实现不同的功能，包括语言图处理（Langraph）以及通过 PandasAI 等库实现高级数据处理。

---

## Structure  
## 项目结构

### `data`  
This folder contains all the data files used by the system.  
该文件夹包含系统使用的所有数据文件。  
The data is crucial for mapping and processing tasks.  
这些数据对于映射和处理任务至关重要。

---

### `demo`  
The `demo` folder contains the main code, which is split into two sections:  
该文件夹包含主代码，分为两个部分：

1. **Langraph**: This section utilizes a local Large Language Model (LLM) for specific processing tasks.  
   **Langraph**：该部分使用本地的大语言模型（LLM）进行特定处理任务。  

2. **PandasAI_3** (now replaced by `pandas-ai`):  
   Originally empowered by PandasAI_3, this section has transitioned to using the updated `pandas-ai` library (v3 beta).  
   原先由 PandasAI_3 支持的部分已替换为最新版的 `pandas-ai`（v3 beta），用于高级数据处理。

---

### `pandas-ai-v3`  
This folder contains the beta version of the `pandas-ai` library.  
该文件夹包含 `pandas-ai` 库的测试版本。  
The open-source library enables seamless integration with pandas DataFrames and allows users to apply AI techniques for data processing.  
该开源库支持与 pandas DataFrame 的无缝集成，并支持应用 AI 技术进行数据处理。

---

### `pandas_ai`  
This folder is a replacement for the deprecated `demo/pandasai_3` directory.  
该文件夹替代了已弃用的 `demo/pandasai_3` 目录。  
It contains the updated version of the codebase using `pandas-ai`.  
它包含使用新版本 `pandas-ai` 的代码库。

---

## Requirements  
## 环境要求

To run the project, you'll need the following dependencies:  
运行本项目需要以下依赖：

- Python 3.10 或更低版本  
  - Python 3.10 or lower  
- Required packages listed in `requirements.txt`  
  - `requirements.txt` 中列出的依赖包

---

## Installation  
## 安装方式

To install the dependencies, simply run the docker:  
要安装依赖项，只需运行 Docker：

```bash
# 示例命令略去，视项目实际提供的 dockerfile 为准
