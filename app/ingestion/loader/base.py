from abc import ABC, abstractmethod
from typing import Union


class BaseLoader(ABC):

    @abstractmethod
    def load(self, source: Union[str, bytes]):
        """
        从数据源加载 Document。
        """
        pass
