# runtime/sources/browser.py

from typing import List
from runtime.sources.base import EventSource
from world.timeline import WorldEvent


class BrowserSource(EventSource):
    def poll(self) -> List[WorldEvent]:
        # Placeholder: browser automation / observer hooks
        return []
