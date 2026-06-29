import random

from typing import Generator
from collections import defaultdict

from ..struct.config import Config
from ..struct.model_config import ModelsConfig
from ..struct.recording import Recording, RecordingQuality
from ..struct.channel import Channel, ChannelQuality
from ..struct.epoch import Epoch, RuleResult
from ..struct.rule import Rule, RuleType

from ..consts import CRITICAL_CHANNELS


def determine_quality(
    recording: Recording, config: Config, models_config: ModelsConfig
) -> None:
    """Tests the channels according to rules assigned to them, and then determines the
    quality of the recording from that."""

    # Go over the channels that have a rule assigned to them
    for channel in recording.channels:
        if not channel.name in config.channels_rules:
            channel.quality = None
            continue

        # Go over the rules and test the epochs
        for rule in config.channels_rules[channel.name]:
            for epoch in channel.epochs:
                _test_epoch(epoch, rule, models_config)

        # Now that epochs have been tested on all the rules,
        # see if a RED threshold has been surpassed
        has_red = False
        for rule in config.channels_rules[channel.name]:
            if _test_channel_red(channel, rule):
                has_red = True
                break

        # If a RED threshold has been surpassed, then channel
        # is RED. Work is done for this channel.
        if has_red:
            continue

        # No RED threshold surpassed. Check YELLOW
        has_yellow = False
        for rule in config.channels_rules[channel.name]:
            if _test_channel_yellow(channel, rule):
                has_yellow = True
                break

        # Same idea as RED
        if has_yellow:
            continue

        # Otherwise, the channel is GREEN
        channel.quality = ChannelQuality.GREEN
        channel.rule_that_gave_quality = None

    # By here we have a quality for all the channels or None for channels that
    # do not have a rule assigned to them. We distinguish between the two scenarios
    # addressed in struct.recording.RecordingQuality and determine the quality
    # based on that.
    critical_has_quality = False
    for channel in _iter_critical_channels(recording):
        if channel.quality is not None:
            critical_has_quality = True
            break

    if critical_has_quality:
        _determine_quality_yes_critical(recording)
    else:
        _determine_quality_no_critical(recording)

    assert all(
        [
            hasattr(recording, "quality") and recording.quality is not None,
            hasattr(recording, "quality_justification")
            and recording.quality_justification,
        ]
    ), f"Recheck your work! {recording.file_path.name}"


# region Rules tests


def _test_epoch(epoch: Epoch, rule: Rule, models_config: ModelsConfig) -> None:
    """Tests a rule on an epoch and stores its result in the epoch."""

    result = None

    if RuleType.RANDOM_RULE is rule.rule_type:
        result = random.random() <= rule.value

    elif RuleType.CLIPPED is rule.rule_type:
        result = epoch.clip_frac >= rule.value

    elif RuleType.FLAT is rule.rule_type:
        result = epoch.flatline_max_s >= rule.value

    elif RuleType.HIGH_RMS is rule.rule_type:
        result = epoch.rms_uv >= rule.value

    elif RuleType.LOW_RMS is rule.rule_type:
        result = epoch.rms_uv <= rule.value

    elif RuleType.HIGH_DIFF is rule.rule_type:
        result = epoch.max_diff_uv >= rule.value

    elif RuleType.HIGH_MAD_DIFF is rule.rule_type:
        result = epoch.mad_diff_uv >= rule.value

    elif RuleType.HIGH_PTP is rule.rule_type:
        result = epoch.ptp_uv >= rule.value

    elif RuleType.HIGH_MAD is rule.rule_type:
        result = epoch.mad_uv >= rule.value

    elif RuleType.HAS_ARTF is rule.rule_type:
        result = _test_has_artf_rule(epoch, rule, models_config)

    else:
        assert False, f"You forgot to implement this rule_type: {rule.rule_type}"

    assert result is not None, f"You forgot to assign result value for {rule.rule_type}"

    rule_result = RuleResult()
    rule_result.rule = rule
    rule_result.result = result
    epoch.rule_results[rule.rule_type] = rule_result


def _test_has_artf_rule(epoch: Epoch, rule: Rule, models_config: ModelsConfig) -> bool:
    if not epoch.channel.has_predictions:
        return False

    # Iter over model name that predicted an artifact
    for model_name, prediction in epoch.predictions.items():
        if not prediction:
            continue

        # Look for that model and see if it's artifact is same as that in rule
        for model in models_config.models:
            if model.model_name == model_name and model.artifact_name == rule.value:
                return True

    return False


def _count_epoch_percentage(channel: Channel, rule: Rule) -> float:
    """Counts the percentage of epochs of the channel that pass the given rule."""

    total = 0
    passed = 0

    for epoch in channel.epochs:
        total += 1
        if epoch.rule_results[rule.rule_type].result:
            passed += 1

    return (passed / float(total)) * 100


def _test_channel_red(channel: Channel, rule: Rule) -> bool:
    """Tests if the given channel has surpassed the RED threshold
    for the given rule. If so, it sets the quality and returns True."""

    percentage = _count_epoch_percentage(channel, rule)

    if percentage >= rule.red_threshold:
        channel.quality = ChannelQuality.RED
        channel.rule_that_gave_quality = rule
        return True

    return False


def _test_channel_yellow(channel: Channel, rule: Rule) -> bool:
    """Tests if the given channel has surpassed the YELLOW threshold
    for the given rule. If so, it sets the quality and returns True."""

    # Yes, this counts the same percentage twice. That's fine.
    percentage = _count_epoch_percentage(channel, rule)

    if percentage >= rule.yellow_threshold:
        channel.quality = ChannelQuality.YELLOW
        channel.rule_that_gave_quality = rule
        return True

    return False


# endregion


# region Recording tests


def _determine_quality_yes_critical(recording: Recording) -> None:
    """Determines the quality of a recording in the scenario that at
    least one critical channel has a rule assigned to it (meaning at least
    one critical channel has a quality)."""

    # If any critical channel is RED => recording is RED
    for channel in _iter_critical_channels(recording):
        if channel.quality is ChannelQuality.RED:
            recording.quality = RecordingQuality.RED
            recording.quality_justification = (
                f"Critical channel ({channel.name}) is RED."
            )
            return

    # Critical channels are either YELLOW or GREEN
    # If any critical channel is YELLOW => recording is YELLOW
    for channel in _iter_critical_channels(recording):
        if channel.quality is ChannelQuality.YELLOW:
            recording.quality = RecordingQuality.YELLOW
            recording.quality_justification = (
                f"Critical channel ({channel.name}) is YELLOW."
            )
            return

    # Critical channels are GREEN
    # If any normal channel is RED => recording is YELLOW
    for channel in _iter_non_critical_channels(recording):
        if channel.quality is ChannelQuality.RED:
            recording.quality = RecordingQuality.YELLOW
            recording.quality_justification = (
                f"Critical channel(s) are GREEN, but the channel {channel.name} is RED."
            )
            return

    # Critical channels are GREEN and no normal channel is RED.
    recording.quality = RecordingQuality.GREEN
    recording.quality_justification = (
        f"Critical channel(s) are GREEN, and no channel is RED."
    )


def _determine_quality_no_critical(recording: Recording) -> None:
    """Opposite of _determine_quality_yes_critical."""

    #  No critical channels has a quality.
    # At least one normal channel has a quality.

    # Determine majorities
    majorities = _determine_non_critical_majorities(recording)

    # If majority contains RED => recording is RED
    if ChannelQuality.RED in majorities:
        recording.quality = RecordingQuality.RED
        recording.quality_justification = f"Most of the channels are RED."
        return

    # Majority is either just YELLOW, YELLOW and GREEN, or just GREEN
    # If majority contains YELLOW => recording is YELLOW
    if ChannelQuality.YELLOW in majorities:
        recording.quality = RecordingQuality.YELLOW
        recording.quality_justification = f"Most of the channels are YELLOW."
        return

    # Majority is GREEN
    assert {ChannelQuality.GREEN} == majorities, f"UNREACHABLE."

    recording.quality = RecordingQuality.GREEN
    recording.quality_justification = f"Most of the channels are GREEN."


# endregion

# region Utils


def _iter_critical_channels(recording: Recording) -> Generator[Channel, None, None]:
    """Generator that iterates over the critical channels."""

    for channel in recording.channels:
        if channel.name in CRITICAL_CHANNELS:
            yield channel


def _iter_non_critical_channels(recording: Recording) -> Generator[Channel, None, None]:
    """Generator that iterates over the non critical channels."""

    for channel in recording.channels:
        if channel.name not in CRITICAL_CHANNELS:
            yield channel


def _determine_non_critical_majorities(recording: Recording) -> set[ChannelQuality]:
    """Determines the majority qualities for non critical channels."""

    count: defaultdict[ChannelQuality, int] = defaultdict(int)

    for channel in _iter_non_critical_channels(recording):
        if channel.quality is not None:
            count[channel.quality] += 1

    assert (
        len(count) != 0
    ), f"Edge case escaped config_reader. No rules were assigned. {recording.file_path.name}"

    max_count = max(count.values())
    majorities = {quality for quality, c in count.items() if c == max_count}
    return majorities


# endregion
