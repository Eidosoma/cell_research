"""Frozen communication-delivery semantics for S01 ``emit_signal`` actions.

Delivery profile ``recipient_activation_lag_lww_sum_u8_v1`` is intentionally
small and deterministic.  The recipient observes messages only at its next
activation.  Messages never modify movement legality and communication costs
remain a separate ledger family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .contracts import SuiteValidationError


DELIVERY_PROFILE = "recipient_activation_lag_lww_sum_u8_v1"
COMMUNICATION_LEDGER_FIELDS = (
    "emittedSignalWrites",
    "transmittedSignalBits",
    "recipientDeliveries",
    "bufferOverwrites",
    "consumedDeliveries",
    "observableAggregateReads",
)


@dataclass(frozen=True, slots=True)
class SignalEmission:
    sender_id: str
    event_index: int
    signals: Mapping[int, int]


@dataclass(frozen=True, slots=True)
class SignalObservation:
    recipient_id: str
    activation_index: int
    channel_sums: Mapping[int, int]
    consumed_sender_channel_pairs: int


class RecipientActivationMessageBus:
    """Deliver last writes to pre-transition neighbors at recipient activation.

    - emission happens after the sender's policy observation and native action;
    - recipients are resolved from the supplied pre-transition neighbor map;
    - one pending value is stored per sender/channel/recipient (last write wins);
    - pending values persist until that recipient's next activation;
    - the recipient sees only per-channel saturated sums, never sender IDs;
    - observation consumes every pending value for that recipient;
    - communication never changes legality, retry, stopping, or native costs.
    """

    __slots__ = ("channels", "bits_per_channel", "maximum", "pending", "ledger")

    def __init__(self, *, channels: int, bits_per_channel: int) -> None:
        if not 1 <= channels <= 4:
            raise SuiteValidationError("communication channels must be in [1, 4]")
        if not 1 <= bits_per_channel <= 8:
            raise SuiteValidationError("signal bits per channel must be in [1, 8]")
        if channels * bits_per_channel > 32:
            raise SuiteValidationError("outbound signal width exceeds DSL bound")
        self.channels = channels
        self.bits_per_channel = bits_per_channel
        self.maximum = 2**bits_per_channel - 1
        self.pending: dict[str, dict[tuple[str, int], tuple[int, int]]] = {}
        self.ledger = {field: 0 for field in COMMUNICATION_LEDGER_FIELDS}

    def emit(
        self,
        emission: SignalEmission,
        pre_transition_neighbors: Mapping[str, Sequence[str]],
    ) -> None:
        if emission.event_index < 0 or not emission.sender_id:
            raise SuiteValidationError("invalid signal emission address")
        if emission.sender_id not in pre_transition_neighbors:
            raise SuiteValidationError("sender missing from pre-transition topology")
        for channel, value in emission.signals.items():
            if isinstance(channel, bool) or not isinstance(channel, int):
                raise SuiteValidationError("signal channel must be an integer")
            if not 0 <= channel < self.channels:
                raise SuiteValidationError("signal channel is outside contract")
            if isinstance(value, bool) or not isinstance(value, int):
                raise SuiteValidationError("signal value must be an integer")
            if not 0 <= value <= self.maximum:
                raise SuiteValidationError("signal value is outside channel width")
        recipients = tuple(sorted(set(pre_transition_neighbors[emission.sender_id])))
        if emission.sender_id in recipients:
            raise SuiteValidationError("self-delivery is forbidden")
        for channel, value in sorted(emission.signals.items()):
            self.ledger["emittedSignalWrites"] += 1
            self.ledger["transmittedSignalBits"] += self.bits_per_channel * len(
                recipients
            )
            for recipient in recipients:
                mailbox = self.pending.setdefault(recipient, {})
                key = (emission.sender_id, channel)
                self.ledger["bufferOverwrites"] += int(key in mailbox)
                mailbox[key] = (emission.event_index, value)
                self.ledger["recipientDeliveries"] += 1

    def observe(self, recipient_id: str, activation_index: int) -> SignalObservation:
        if activation_index < 0 or not recipient_id:
            raise SuiteValidationError("invalid signal observation address")
        mailbox = self.pending.get(recipient_id, {})
        if any(event_index >= activation_index for event_index, _ in mailbox.values()):
            raise SuiteValidationError("same/future-event signal delivery is forbidden")
        mailbox = self.pending.pop(recipient_id, {})
        totals = {channel: 0 for channel in range(self.channels)}
        for (sender, channel), (event_index, value) in sorted(mailbox.items()):
            del sender
            totals[channel] = min(self.maximum, totals[channel] + value)
        self.ledger["consumedDeliveries"] += len(mailbox)
        self.ledger["observableAggregateReads"] += self.channels
        return SignalObservation(
            recipient_id,
            activation_index,
            totals,
            len(mailbox),
        )

    def ledger_snapshot(self) -> dict[str, int]:
        return dict(self.ledger)
