"""Regression tests for the pure PB-QCOST deadline serializer."""
import datetime
import math

import pytest

from deadline_contract import deadline_arg_utc


def test_deadline_arg_uses_parser_accepted_utc_z_form():
    epoch = 1793509170.0
    value = deadline_arg_utc(epoch)
    assert value == "2026-11-01T04:59:30Z"
    legacy_value = datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).isoformat()
    assert value.endswith("Z") and not legacy_value.endswith("Z")


@pytest.mark.parametrize("epoch", [math.nan, math.inf, -math.inf])
def test_deadline_arg_rejects_nonfinite_epoch(epoch):
    with pytest.raises(ValueError, match="must be finite"):
        deadline_arg_utc(epoch)
