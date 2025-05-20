import sys
import os
from pathlib import Path

# 添加 pandas-ai 目录到 Python 路径
current_dir = Path(__file__).resolve().parent
pandas_ai_path = current_dir.parent.parent / "pandas-ai-3.0.0a16"
sys.path.append(str(pandas_ai_path))

import pandasai as pai
import pandas as pd
from pandasai.dataframe.base import DataFrame as PaiDataFrame
from pandasai.agent.base import Agent as BaseAgent
from Readfile import read_file, sanitize_column_name

class Agent(BaseAgent):
    def __init__(self):
        # 设置 API key
        pai.api_key.set("PAI-a4927d55-bcb4-4e97-8cef-4250564d1f69")
        # 初始化 LLM
        self.llm = pai
        # 初始化数据框列表为空
        self.df_list = []
        # 初始化父类，传入空的DataFrame列表
        super().__init__(dfs=[], config=None, memory_size=10)
        
    def readFile(self, file_path, file_name):
        """读取文件并返回处理后的数据框列表"""
        try:
            # 构建完整文件路径
            full_path = os.path.join(file_path, file_name)
            # 读取文件并获取数据框列表
            self.df_list = read_file(pai_=self.llm, file_path=full_path, file_name=file_name)
            # 更新父类的数据框列表
            self._state.dfs = self.df_list
            return self.df_list
        except Exception as e:
            print(f"读取文件时发生错误: {e}")
            return []
            
    def chat(self, prompt):
        """封装聊天功能，确保使用当前加载的数据框"""
        if not self.df_list:
            raise ValueError("请先使用 readFile() 方法加载数据")
        return super().chat(prompt)



if __name__ == "__main__":
    try:
        # 设置文件路径
        file_path = "../../data/"
        file_name = "TBC Lille.xlsx"
        
        # 创建 Agent 实例
        agent = Agent()
        
        # 读取文件
        df_list = agent.readFile(file_path, file_name)
        
        if df_list:
            # 示例查询
            queries = [
                "show me all item which storage day is more than 200 days in the dataset",
                'find all records where hazardcategory is "Non DG"'
            ]
            
            for query in queries:
                try:
                    result = agent.chat(query)
                    print(f"\n查询: {query}")
                    print("结果:", result)
                except Exception as e:
                    print(f"执行查询时发生错误: {e}")
        else:
            print("未能成功读取数据文件")
            
    except Exception as e:
        print(f"程序执行过程中发生错误: {e}")