from pandasai.core.prompts.base import BasePrompt
import pandasai as pai
import pandas as pd
from pathlib import Path
import os
import re
import logging
from pandasai.dataframe.base import DataFrame as PaiDataFrame
from Readfile import read_file

# 设置环境变量启用调试模式
os.environ["PANDASAI_VERBOSE"] = "1"

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('pandasai')
logger.setLevel(logging.DEBUG)

def print_query_info(response):
    """打印查询信息，包括生成的代码"""

    print("\n查询结果:")
    print(response)

# 创建配置
config = {
    "verbose": True,
    "save_logs": True,
    "enable_cache": False
}


pai.api_key.set("PAI-a4927d55-bcb4-4e97-8cef-4250564d1f69")
# Load your data
# df = pai.DataFrame(pd.read_csv("../../data/TBC Lille.csv"))
# 用 pandas 讀 CSV，並加上 encoding 參數



# ------------------------------------------------------------------------------------
# 多表讀取
all_sheets = pd.read_excel("data/HUAWEI - Export inventaire GenateQ(1).xlsx", sheet_name=None)

pai_dfs = []

for sheet_name, dfs in all_sheets.items():
    df = pai.DataFrame(dfs)
    pai_dfs.append(df)
    print(f"已轉換：{sheet_name} -> pai.DataFrame")


print(pai_dfs[1])

# 多表詢問(多表)
response = pai.chat( "show me all item which localisation is STOCK_CARVIN in the dataset", *pai_dfs)
print("First query result:")
print(response.value)
response = pai.DataFrame(response.value)
result = pai.chat( "According to the previous results, show me all item which famille is SDH in the dataset", response)
print("Second query result:")
print(result)
#------------------------------------------------------------------------------------

#多表詢問(單一表)
# all_sheets = pd.read_excel("data/HUAWEI - Export inventaire GenateQ(1).xlsx", sheet_name=None)

# # 转换为 PandasAI DataFrame 并设置配置
# pai_dfs = pai.DataFrame(all_sheets['STOCK_TIBCO_HUAWEI'])
# pai_dfs._config = config

# # 第一次查询
# print("\n执行第一次查询...")
# response = pai_dfs.chat("show me all records which localisation is STOCK_CARVIN in the dataset")
# print_query_info(response)

# # 第二次查询
# print("\n执行第二次查询...")
# result = pai_dfs.chat('find the rest records Modèle is "02120110-SSEE1FAN" in the dataset')
# print_query_info(result)



#------------------------------------------------------------------------------------
#單一表持續詢問
# df_raw = pd.read_csv("data/TBC Lille.csv", encoding="latin1")
# # 然後轉成 pandasai 的 DataFrame
# df = pai.DataFrame(df_raw)


# response = df.chat('show me all item which "storage days" is more than 200 in the dataset')
# print("First query result:")
# print_query_info(response)
# print(response.value)

# df = pai.DataFrame(response.value)
# response2 = df.chat('find all records where hazardcategory equals "Normal Cargo"')
# # 使用第一次查询的结果进行第二次查询
# print("Second query result:")
# print_query_info(response2)


#------------------------------------------------------------------------------------























# Example usage
# if __name__ == "__main__":
#     agent = Agent()
#     file_path = "../../data/"
#     file_name = "TBC Lille.xlsx"
    
#     # Read the file
#     agent.readFile(file_path, file_name)
    
#     # Make a request
#     request = '''show me all the columns in the dataset'''
#     result = agent.llm.chat(request,*agent.df_list)
    
#     print(result)

# if __name__ == "__main__":
#     agent = Agent()

#     # 取得絕對路徑：從當前檔案往上兩層，再到 data 資料夾下的檔案
#     file_path = Path(__file__).resolve().parents[2] / "data" / "TBC Lille.xlsx"

#     # Debug：檢查檔案是否存在
#     if not file_path.exists():
#         print(f"❌ File not found: {file_path}")
#     else:
#         print(f"✅ Found file: {file_path}")

#         # Read the file
#         df = agent.readFile(str(file_path), "tbc")

#         # Make a request
      
#         # request = '''show me all the columns in the dataset'''
#         # result = agent.llm.chat(request, *agent.df_list)\
        
#         # llm.chat()

# #多文檔讀取操作

#         # Load existing datasets
#         # stocks = pai.load("organization/coca_cola_stock")

#         # # Query using multiple datasets 
#         # result = pai.chat("Compare the revenue between Coca Cola and Apple", stocks, companies)

# #与多个 DataFrame 聊天

#         # df_customers = pai.load("company/customers")
#         # df_orders = pai.load("company/orders")
#         # df_products = pai.load("company/products")

#         # response = pai.chat('Who are our top 5 customers and what products do they buy most frequently?', df_customers, df_orders, df_products)


#         myquery = '''show me all item which storage day is more than 200 days in the dataset'''
#         result = agent.llm.chat(myquery, *agent.df_list)
#         print(result)
#         myquery_2 = '''find all records where hazardcategory is "Non DG"'''
#         result = agent.llm.chat(myquery_2, *agent.df_list)

        
        

#         print(result)