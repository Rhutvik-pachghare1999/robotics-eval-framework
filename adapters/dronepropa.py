"""DronePropA adapter — STUB. Verify fault taxonomy + per-flight structure first."""
from .base import BaseAdapter


class DronePropAAdapter(BaseAdapter):
    REGISTRY_ID = "dronepropa"
    VERIFIED = False
    LABEL_SEMANTICS = ""  # fill from the Mendeley record: exact fault types + severities

    def load(self):
        self._guard()
        # TODO: parse flight logs -> (accel, gyro, rpm, attitude), fault_type, severity,
        #       flight_id (for leave-one-flight-out grouped splits).
        raise NotImplementedError("Implement after verifying DronePropA fault taxonomy.")
