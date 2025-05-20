from pandasai.core.prompts.base import BasePrompt
import pandasai as pai
import pandas as pd
from pathlib import Path
import os
import re
from pandasai.dataframe.base import DataFrame as PaiDataFrame
from Readfile import read_file

# class Agent():
#     def __init__(self):
#         # Set the API key
#         pai.api_key.set("PAI-a4927d55-bcb4-4e97-8cef-4250564d1f69")
#         # Initialize the LLM
#         self.llm = pai
#         self.df_list = None
#         self.request = None
#         self._current_agent = None  # 添加这行来跟踪当前的对话

#     def readFile(self, file_path, file_name):
#         self.df_list = read_file(pai_=self.llm, file_path=file_path, file_name=file_name)
#         return self.df_list

#     def chat(self, query, df):
#         """开始新的对话"""
#         result = self.llm.chat(query, df)
#         self._current_agent = result  # 保存对话状态
#         return result

#     def follow_up(self, query):
#         """继续现有对话"""
#         if self._current_agent is None:
#             raise ValueError("No existing conversation. Please use chat() first.")
#         return self.llm.follow_up(query)


# # Example usage
# # if __name__ == "__main__":
# #     agent = Agent()
# #     file_path = "../../data/"
# #     file_name = "TBC Lille.xlsx"
    
# #     # Read the file
# #     agent.readFile(file_path, file_name)
    
# #     # Make a request
# #     request = '''show me all the columns in the dataset'''
# #     result = agent.llm.chat(request,*agent.df_list)
    
# #     print(result)

# if __name__ == "__main__":
#     agent = Agent()
#     file_path = Path(__file__).resolve().parents[2] / "data" / "TBC Lille.xlsx"

#     if not file_path.exists():
#         print(f"❌ File not found: {file_path}")
#     else:
#         print(f"✅ Found file: {file_path}")

#         df_list = agent.readFile(str(file_path), "tbc")
        
#         if df_list and len(df_list) > 0:
#             df = df_list[0]
            
#             # 打印数据框信息
#             print("\nDataFrame info:")
#             print("Columns:", df.columns.tolist())
            
#             # 第一次查询 - 使用 chat
#             result = agent.chat("show me all item which storage day is more than 200 days in the dataset", df)
#             print("\nFirst query result:")
#             print(result)
            
#             # 第二次查询 - 使用 follow_up
#             result = agent.follow_up('find all records where hazardcategory equals "Non DG"')
#             print("\nSecond query result:")
#             print(result)


class Agent():
    def __init__(self):
        # Set the API key
        pai.api_key.set("PAI-a4927d55-bcb4-4e97-8cef-4250564d1f69")
        # Initialize the LLM
        self.llm = pai
        self.df_list = None
        self.request = None
        self.current_df = None  # 当前数据框

    def readFile(self, file_path, file_name):
        """读取文件并保存为当前数据框"""
        self.df_list = read_file(pai_=self.llm, file_path=file_path, file_name=file_name)
        if self.df_list and len(self.df_list) > 0:
            self.current_df = self.df_list[0]
            # 确保 storage_days 列为数值类型
            if 'storage_days' in self.current_df.columns:
                self.current_df['storage_days'] = pd.to_numeric(self.current_df['storage_days'], errors='coerce')
        return self.df_list

    def execute_query(self, query, df=None):
        """执行单个查询，返回结果"""
        df_to_use = df if df is not None else self.current_df
        if df_to_use is None:
            raise ValueError("No DataFrame available")
        
        try:
            # 重命名列以匹配查询
            column_mapping = {
                'Storage Days': 'storage_days',
                'HazardCategory': 'hazardcategory'
            }
            df_to_use = df_to_use.rename(columns=column_mapping)
            
            # 确保 storage_days 列为数值类型
            if 'storage_days' in df_to_use.columns:
                df_to_use['storage_days'] = pd.to_numeric(df_to_use['storage_days'], errors='coerce')
            
            # 打印当前数据框信息
            print(f"\n当前数据框信息:")
            print(f"形状: {df_to_use.shape}")
            print(f"列名: {df_to_use.columns.tolist()}")
            
            # 使用更明确的查询语句
            if "storage day" in query.lower():
                print("\n执行存储天数查询...")
                # 直接使用 pandas 操作进行筛选
                if 'storage_days' in df_to_use.columns:
                    result = df_to_use[df_to_use['storage_days'] > 200].copy()
                    print(f"找到 {len(result)} 条记录")
                    if len(result) > 0:
                        print("\n数据预览:")
                        print(result[['Item', 'storage_days', 'hazardcategory']].head())
                    return result
                else:
                    print("错误：未找到 storage_days 列")
                    print("可用的列:", df_to_use.columns.tolist())
                    return None
            elif "hazardcategory" in query.lower():
                print("\n执行危险品类别查询...")
                # 直接使用 pandas 操作进行筛选
                if 'hazardcategory' in df_to_use.columns:
                    print("\nhazardcategory 列的唯一值:")
                    print(df_to_use['hazardcategory'].unique())
                    result = df_to_use[df_to_use['hazardcategory'] == "Non DG"].copy()
                    print(f"找到 {len(result)} 条记录")
                    if len(result) > 0:
                        print("\n数据预览:")
                        print(result[['Item', 'storage_days', 'hazardcategory']].head())
                    return result
                else:
                    print("错误：未找到 hazardcategory 列")
                    print("可用的列:", df_to_use.columns.tolist())
                    return None
            else:
                # 对于其他查询，使用 PandasAI
                result = self.llm.chat(query, df_to_use)
            
            # 如果结果是 DataFrame，更新当前数据框
            if isinstance(result, pd.DataFrame):
                self.current_df = result
                print(f"查询返回 DataFrame，形状: {result.shape}")
            else:
                print(f"查询返回类型: {type(result)}")
                
            return result
        except Exception as e:
            print(f"查询执行错误: {e}")
            print("错误类型:", type(e).__name__)
            return None

if __name__ == "__main__":
    agent = Agent()
    file_path = Path(__file__).resolve().parents[2] / "data" / "TBC Lille.xlsx"

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
    else:
        print(f"✅ Found file: {file_path}")

        df_list = agent.readFile(str(file_path), "tbc")
        
        if df_list and len(df_list) > 0:
            # 打印原始数据框信息
            print("\nOriginal DataFrame info:")
            print("Columns:", agent.current_df.columns.tolist())
            print("Shape:", agent.current_df.shape)
            
            try:
                # 第一次查询 - 查找存储天数超过200天的记录
                print("\n执行第一次查询 - 存储天数超过200天的记录...")
                result1 = agent.execute_query("show me all item which storage day is more than 200 days in the dataset")
                
                if isinstance(result1, pd.DataFrame) and not result1.empty:
                    print("\n第一次查询结果:")
                    print("形状:", result1.shape)
                    print("\n数据预览:")
                    print(result1[['item', 'storage_days', 'hazardcategory']].head())
                    
                    # 第二次查询 - 在结果中查找 Non DG 记录
                    print("\n执行第二次查询 - 查找 Non DG 记录...")
                    result2 = agent.execute_query('find all records where hazardcategory equals "Non DG"', result1)
                    
                    if isinstance(result2, pd.DataFrame) and not result2.empty:
                        print("\n最终查询结果:")
                        print("形状:", result2.shape)
                        print("\n数据预览:")
                        print(result2[['item', 'storage_days', 'hazardcategory']].head())
                    else:
                        print("\n第二次查询未返回有效结果")
                else:
                    print("\n第一次查询未返回有效结果")
                    print("在原始数据框上执行第二次查询...")
                    result2 = agent.execute_query('find all records where hazardcategory equals "Non DG"', agent.current_df)
                    if isinstance(result2, pd.DataFrame) and not result2.empty:
                        print("\n查询结果:")
                        print("形状:", result2.shape)
                        print("\n数据预览:")
                        print(result2[['item', 'storage_days', 'hazardcategory']].head())
                    else:
                        print("\n查询未返回有效结果")
            except Exception as e:
                print(f"执行错误: {e}")