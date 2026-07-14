import yaml
from pathlib import Path
from typing import Callable, Any

from .struct.config import Config
from .consts import NEEDED_CHANNELS
from .struct.rule import Rule, RuleType

# region Utility


class _ConfigError(Exception):
    """Raised internally when a config key is missing or fails validation."""


def _read_simple_key(
    data: dict,
    key: str,
    validator: Callable[[Any], bool],
    error_msg: str,
    missing_msg: str | None = None,
) -> Any:
    """Read and validate a single key from parsed config data.

    Args:
        data: The parsed config dict to read from.
        key: The key to look up.
        validator: Callable that returns True if the value is acceptable.
        error_msg: Message to raise if the key is present but invalid.
        missing_msg: Message to raise if the key is missing entirely.
            Defaults to a generic "missing key: {key}" message.

    Returns:
        The value associated with key, if present and valid.

    Raises:
        _ConfigError: If the key is missing or fails validation.
    """
    if key not in data:
        raise _ConfigError(missing_msg or f"Missing required value: '{key}'")

    if not validator(data[key]):
        raise _ConfigError(error_msg)

    return data[key]


def _read_default_rules(data: dict) -> dict[RuleType, Rule]:
    """Read and validate the default_rules_values section.

    Args:
        data: The parsed config dict to read from.

    Returns:
        Mapping of RuleType to its default Rule.

    Raises:
        _ConfigError: If the section, a rule, or one of its required
            fields (value, red, yellow) is missing or invalid.
    """

    raw_rules = _read_simple_key(
        data,
        "default_rules_values",
        lambda v: isinstance(v, dict),
        "default_rules_values must be a mapping of rule name to its fields",
    )

    rules: dict[RuleType, Rule] = {}

    for rule_name, rule_data in raw_rules.items():
        try:
            rule_type = RuleType[rule_name]
        except KeyError:
            raise _ConfigError(f"Unknown rule type: '{rule_name}'")

        if not isinstance(rule_data, dict):
            raise _ConfigError(f"Rule '{rule_name}' must be a mapping")

        value = _read_simple_key(
            rule_data,
            "value",
            lambda v: v is not None,
            f"'{rule_name}.value' must not be null",
            f"Rule '{rule_name}' is missing required field: 'value'",
        )
        red = _read_simple_key(
            rule_data,
            "red",
            lambda v: isinstance(v, (int, float)) and 0 <= v <= 100,
            f"'{rule_name}.red' must be a number between 0 and 100",
            f"Rule '{rule_name}' is missing required field: 'red'",
        )
        yellow = _read_simple_key(
            rule_data,
            "yellow",
            lambda v: isinstance(v, (int, float)) and 0 <= v <= 100 and v < red,
            f"'{rule_name}.yellow' must be a number between 0 and 100, less than red ({red})",
            f"Rule '{rule_name}' is missing required field: 'yellow'",
        )

        rule = Rule()
        rule.rule_type = rule_type
        rule.value = value
        rule.red_threshold = red
        rule.yellow_threshold = yellow
        rules[rule_type] = rule

    return rules


def _read_inherited_key(
    rule_data: dict,
    default_rule: Rule | None,
    key: str,
    default_attr: str,
    validator: Callable[[Any], bool],
    error_msg: str,
    channel_name: str,
    rule_name: str,
) -> Any:
    """Read a single field from a channel rule override, falling back to
    the default rule's value if the field is absent.

    Args:
        rule_data: The channel-level rule mapping (e.g. {"value": 0.9}).
        default_rule: The matching default Rule, or None if no default exists.
        key: The field name to look up (e.g. "value", "red", "yellow").
        default_attr: The attribute name on Rule to fall back to (e.g. "red_threshold").
        validator: Callable that returns True if a present value is acceptable.
        error_msg: Message to raise if the key is present but invalid.
        channel_name: Channel name, for error messages.
        rule_name: Rule name, for error messages.

    Returns:
        The resolved value for this field.

    Raises:
        _ConfigError: If the field is missing with no default to fall back
            on, or present but invalid.
    """

    if key not in rule_data:
        if default_rule is None:
            raise _ConfigError(
                f"Channel '{channel_name}', rule '{rule_name}' has no default "
                f"and is missing required field: '{key}'"
            )

        return getattr(default_rule, default_attr)

    if not validator(rule_data[key]):
        raise _ConfigError(error_msg)

    return rule_data[key]


def _read_channel_rule(
    rule_name: str,
    rule_data: dict | None,
    default_rules: dict[RuleType, Rule],
    channel_name: str,
) -> Rule:
    """Read and validate a single rule entry for a channel.

    Args:
        rule_name: The rule's name as written in YAML (e.g. "RANDOM_RULE").
        rule_data: Either None (use default rule unmodified) or a mapping
            overriding value/red/yellow on top of the matching default.
        default_rules: Already-parsed default rules, used for inheritance.
        channel_name: Channel name, for error messages.

    Returns:
        The resolved Rule for this channel.

    Raises:
        _ConfigError: If the rule type is unknown, the rule has no default
            and is missing required fields, or a present field is invalid.
    """

    try:
        rule_type = RuleType[rule_name]
    except KeyError:
        raise _ConfigError(
            f"Channel '{channel_name}' uses unknown rule type: '{rule_name}'"
        )

    default_rule = default_rules.get(rule_type)

    if rule_data is None:
        if default_rule is None:
            raise _ConfigError(
                f"Channel '{channel_name}', rule '{rule_name}': "
                f"no default exists for this rule, so it cannot be null"
            )
        return default_rule

    if not isinstance(rule_data, dict):
        raise _ConfigError(
            f"Channel '{channel_name}', rule '{rule_name}' must be null or a mapping"
        )

    value = _read_inherited_key(
        rule_data,
        default_rule,
        "value",
        "value",
        lambda v: v is not None,
        f"Channel '{channel_name}', rule '{rule_name}': 'value' must not be null",
        channel_name,
        rule_name,
    )
    red = _read_inherited_key(
        rule_data,
        default_rule,
        "red",
        "red_threshold",
        lambda v: isinstance(v, (int, float)) and 0 <= v <= 100,
        f"Channel '{channel_name}', rule '{rule_name}': 'red' must be a number between 0 and 100",
        channel_name,
        rule_name,
    )
    yellow = _read_inherited_key(
        rule_data,
        default_rule,
        "yellow",
        "yellow_threshold",
        lambda v: isinstance(v, (int, float)) and 0 <= v <= 100 and v < red,
        f"Channel '{channel_name}', rule '{rule_name}': 'yellow' must be a number between 0 and 100, less than red ({red})",
        channel_name,
        rule_name,
    )

    rule = Rule()
    rule.rule_type = rule_type
    rule.value = value
    rule.red_threshold = red
    rule.yellow_threshold = yellow
    return rule


def _read_channel_rules_list(
    channel_name: str,
    channel_data: Any,
    default_rules: dict[RuleType, Rule],
) -> list[Rule]:
    """Read and validate all rules for a single channel.

    Args:
        channel_name: The channel's name as written in YAML (e.g. "EEG").
        channel_data: The mapping of rule name to rule data for this channel.
        default_rules: Already-parsed default rules, used for inheritance.

    Returns:
        List of resolved Rules for this channel.

    Raises:
        _ConfigError: If channel_data isn't a mapping, or any rule within
            it fails validation.
    """

    if not isinstance(channel_data, dict):
        raise _ConfigError(f"Channel '{channel_name}' must be a mapping")

    return [
        _read_channel_rule(rule_name, rule_data, default_rules, channel_name)
        for rule_name, rule_data in channel_data.items()
    ]


def _read_channel_rules(
    data: dict, default_rules: dict[RuleType, Rule]
) -> dict[str, list[Rule]]:
    """Read and validate the channels section.

    Args:
        data: The parsed config dict to read from.
        default_rules: Already-parsed default rules, used as a fallback
            for fields not overridden at the channel level.

    Returns:
        Mapping of channel name to its list of rules.

    Raises:
        _ConfigError: If the section, a channel, or one of its rules is
            missing or invalid, or if no rules are defined anywhere.
    """

    raw_channels = _read_simple_key(
        data,
        "channels",
        lambda v: isinstance(v, dict),
        "channels must be a mapping of channel name to its rules",
    )

    channels_rules = {
        channel_name: _read_channel_rules_list(
            channel_name, channel_data, default_rules
        )
        for channel_name, channel_data in raw_channels.items()
    }

    if not any(channels_rules.values()):
        raise _ConfigError("At least one rule must be defined across all channels")

    for channel_name in channels_rules.keys():
        if channel_name not in NEEDED_CHANNELS:
            raise _ConfigError(f"Unknown channel name: {channel_name}")

    return channels_rules


# endregion


def read_config_file(config_file: Path) -> tuple[Config, None] | tuple[None, str]:
    """Read and parse the config file.

    As it is right now it is not 100% erroneous input proof.

    Args:
        config_file: Path to the config file to read.

    Returns:
        A tuple of (Config, None) if the file was read and parsed
        successfully, or (None, str) where the string explains what
        went wrong.
    """

    # Parse if not Path object already
    config_file = Path(config_file)

    # Make sure file exists
    if not config_file.is_file():
        return None, f"Config file not found: {config_file.resolve()}"

    # Try and read the YAML config file
    try:
        with config_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return None, f"Invalid YAML in config file: {e}"
    except OSError as e:
        return None, f"Could not read config file: {e}"

    # Extract the config values
    config = Config()
    config.config_file = config_file

    try:
        config.min_duration_h = _read_simple_key(
            data,
            "min_duration_h",
            lambda v: isinstance(v, (int, float)) and v >= 0,
            "min_duration_h must be a number >= 0",
        )

        config.max_duration_h = _read_simple_key(
            data,
            "max_duration_h",
            lambda v: isinstance(v, (int, float)) and v >= config.min_duration_h,
            f"max_duration_h must be a number >= min_duration_h ({config.min_duration_h})",
        )

        # Load default rules values
        default_rules = _read_default_rules(data)

        # Load channel rules
        config.channels_rules = _read_channel_rules(data, default_rules)

    except _ConfigError as e:
        return None, str(e)

    # All good
    return config, None
