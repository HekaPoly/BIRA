from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

from bira_orchestration.enums import (
    StateCode,
)

if TYPE_CHECKING:
    from bira_orchestration.manager import BiraManager

class State(ABC):
    code: StateCode
    
    def __init__(self, bira_manager: BiraManager):
        self.bira_manager = bira_manager
    
    def handle(self):
        self._prepare()
        self._handle()
        self._decide_next_state()
    
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def _prepare(self):
        pass

    @abstractmethod
    def _handle(self):
        pass

    @abstractmethod
    def _decide_next_state(self):
        pass