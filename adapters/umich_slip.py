"""
UMich-CURLY slip-detection adapter — STUB.

DO NOT set VERIFIED=True until you have confirmed, from the UMich paper/repo,
what their slip target actually is and whether/how it maps to NeuroTraction's
slip_ratio. The two are NOT assumed equal.
"""
from .base import BaseAdapter


class UMichSlipAdapter(BaseAdapter):
    REGISTRY_ID = "umich_slip"
    VERIFIED = False
    LABEL_SEMANTICS = ""  # fill after reading the source; cite it.

    def load(self):
        self._guard()
        # TODO after verification:
        #   - parse ROS recordings -> (imu, wheel_velocity, timestamps)
        #   - extract the SOURCE slip target (documented), tag recording/run id
        #   - return standardized samples for grouped (leave-one-run-out) splitting
        raise NotImplementedError("Implement after verifying UMich slip semantics.")
