# runtime/sources/base.py

from abc import ABC, abstractmethod
from typing import List
from world.timeline import WorldEvent


class EventSource(ABC):
    """
    Base class for all runtime event sources.

    Contract:
    - bootstrap(): called ONCE before polling starts. Returns snapshot-style
      events representing current OS truth. Default: no-op.
    - poll(): called repeatedly. Returns diff-based events since last poll.
    """

    @abstractmethod
    def poll(self) -> List[WorldEvent]:
        """
        Return new events since last poll.
        Must be non-blocking.
        """
        pass

    def bootstrap(self) -> List[WorldEvent]:
        """Read current OS state and return initial snapshot events.

        Called ONCE before the polling loop starts.
        Default: no-op (sources that don't need bootstrap skip this).
        Override to provide authoritative initial state.

        Bootstrap events are snapshot-style (full current truth),
        not diff-style (transitions). This distinction is critical.
        """
        return []
