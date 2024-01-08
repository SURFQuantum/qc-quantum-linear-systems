import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from qiskit.providers.backend import BackendV2
from qiskit.providers.provider import ProviderV1


class QuTechProvider(ProviderV1):
    """Provider class for QuTech."""

    def __init__(self) -> None:
        self._backends = [QuantumInspireBackend]

    def backends(
        self, name: Optional[str] = None, **kwargs: Dict[Any, Any]
    ) -> List[BackendV2]:
        """Return a list of backends matching the specified filtering.

        Args:
            name (str): name of the backend.
            **kwargs: dict used for filtering.

        Returns:
            list[Backend]: a list of Backends that match the filtering
                criteria.
        """
        if name in self._backends and name is not None:  # type: ignore
            return [backend for backend in self._backends if backend.name == name]
        else:
            raise ValueError(f"Backend {name} not found for provider QuTech")


class QuantumInspireBackend(BackendV2):
    """Backend class for QuantumInspire."""

    def __init__(self) -> None:
        super().__init__(
            provider=QuTechProvider(),
            name="quantum_inspire",
            description="QuantumInspire backend.",
            online_date=datetime.datetime,
            backend_version="1.0",
        )

    def target(self) -> None:
        """A :class:`qiskit.transpiler.Target` object for the backend.

        :rtype: Target
        """
        pass

    def max_circuits(self) -> None:
        """The maximum number of circuits (or Pulse schedules) that can be run in a
        single job.

        If there is no limit this will return None
        """
        pass

    @classmethod
    def _default_options(cls) -> None:
        """Return the default options.

        This method will return a :class:`qiskit.providers.Options`
        subclass object that will be used for the default options. These
        should be the default parameters to use for the options of the
        backend.

        Returns:
            qiskit.providers.Options: A options object with
                default values set
        """
        pass

    @property
    def dtm(self) -> float:
        """Return the system time resolution of output signals.

        Returns:
            The output signal timestep in seconds.

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                output signal timestep
        """
        raise NotImplementedError

    @property
    def meas_map(self) -> List[List[int]]:
        """Return the grouping of measurements which are multiplexed.

        This is required to be implemented if the backend supports Pulse
        scheduling.

        Returns:
            The grouping of measurements which are multiplexed

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                measurement mapping
        """
        raise NotImplementedError
