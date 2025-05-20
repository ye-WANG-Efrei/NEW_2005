from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pandasai.config import Config, ConfigManager
from pandasai.constants import DEFAULT_CHART_DIRECTORY
from pandasai.data_loader.semantic_layer_schema import is_schema_source_same
from pandasai.exceptions import InvalidConfigError
from pandasai.helpers.folder import Folder
from pandasai.helpers.logger import Logger
from pandasai.helpers.memory import Memory
from pandasai.llm.bamboo_llm import BambooLLM
from pandasai.vectorstores.vectorstore import VectorStore

if TYPE_CHECKING:
    from pandasai.dataframe import DataFrame, VirtualDataFrame
    from pandasai.llm.base import LLM


@dataclass
class AgentState:
    """
    Context class for managing pipeline attributes and passing them between steps.
    """
    #-------------------------------------------------------------
    query_history: List[Union[DataFrame, VirtualDataFrame]] = field(default_factory=list)
    #-------------------------------------------------------------
    dfs: List[Union[DataFrame, VirtualDataFrame]] = field(default_factory=list)
    _config: Union[Config, dict] = field(default_factory=dict)
    memory: Memory = field(default_factory=Memory)
    vectorstore: Optional[VectorStore] = None
    intermediate_values: Dict[str, Any] = field(default_factory=dict)
    logger: Optional[Logger] = None
    last_code_generated: Optional[str] = None
    last_code_executed: Optional[str] = None
    last_prompt_id: str = None
    last_prompt_used: str = None
    output_type: Optional[str] = None

#若傳入 config 是字典格式，轉成 Config 實體

    def __post_init__(self):
        if isinstance(self.config, dict):
            self.config = Config(**self.config)

#初始設定 Agent 狀態：資料表、記憶體、Logger、向量資料庫等。

    def initialize(
        self,
        dfs: Union[
            Union[DataFrame, VirtualDataFrame], List[Union[DataFrame, VirtualDataFrame]]
        ],
        config: Optional[Union[Config, dict]] = None,
        memory_size: Optional[int] = 10,
        vectorstore: Optional[VectorStore] = None,
        description: str = None,
    ):
        """Initialize the state with the given parameters."""
        self.dfs = dfs if isinstance(dfs, list) else [dfs]
        self.config = self._get_config(config)
        if config:
            self.config.llm = self._get_llm(self.config.llm)
        self.memory = Memory(memory_size, agent_description=description)
        self.logger = Logger(
            save_logs=self.config.save_logs, verbose=self.config.verbose
        )
        self.vectorstore = vectorstore
        self._configure()

#	確保預設圖表儲存資料夾存在。

    def _configure(self):
        """Configure paths for charts."""
        # Add project root path if save_charts_path is default
        Folder.create(DEFAULT_CHART_DIRECTORY)

#若無提供 Config，從環境變數或預設值產生 Config 實體。

    def _get_config(self, config: Union[Config, dict, None]) -> Config:
        """Load a config to be used for queries."""
        if config is None:
            return ConfigManager.get()

        if isinstance(config, dict):
            if not config.get("llm") and os.environ.get("PANDABI_API_KEY"):
                config["llm"] = BambooLLM()
            return Config(**config)

        return config

#初始化使用的 LLM（預設為 BambooLLM）。

    def _get_llm(self, llm: Optional[LLM] = None) -> LLM:
        """Load and configure the LLM."""
        return llm or BambooLLM()
    
#使用 UUID 為新 prompt 產生唯一 ID 並記錄
    def assign_prompt_id(self):
        """Assign a new prompt ID."""
        self.last_prompt_id = uuid.uuid4()

        if self.logger:
            self.logger.log(f"Prompt ID: {self.last_prompt_id}")

#2.  記憶與中間資料儲存（intermediate_values）

#清除所有中間資料。
    def reset_intermediate_values(self):
        """Resets the intermediate values dictionary."""
        self.intermediate_values.clear()
#加入一筆中間資料。
    def add(self, key: str, value: Any):
        """Adds a single key-value pair to intermediate values."""
        self.intermediate_values[key] = value
#批量加入多筆中間資料。
    def add_many(self, values: Dict[str, Any]):
        """Adds multiple key-value pairs to intermediate values."""
        self.intermediate_values.update(values)
#取出中間資料，若無則回傳預設值。
    def get(self, key: str, default: Any = "") -> Any:
        """Fetches a value from intermediate values or returns a default."""
        return self.intermediate_values.get(key, default)
    
#讀取時：若無 _config 則使用全域設定；
#寫入時：允許傳入 dict 自動轉換成 Config 實體。

    @property
    def config(self):
        """
        Returns the local config if set, otherwise fetches the global config.
        """
        if self._config is not None:
            return self._config

        import pandasai as pai

        return pai.config.get()

    @config.setter
    def config(self, value: Union[Config, dict, None]):
        """
        Allows setting a new config value.
        """
        self._config = Config(**value) if isinstance(value, dict) else value
