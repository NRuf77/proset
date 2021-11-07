"""Helper functions for use in benchmark scripts.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import logging


# pylint: disable=too-few-public-methods
class _LogTracker:  # pragma: no cover
    """Singleton class for tracking the state of the console log.

    :cvar log_enabled: boolean; whether the console log has already been enabled
    """
    log_enabled = False


def start_console_log(log_level=logging.INFO):  # pragma: no cover
    """Start root logger output to console if not already enabled.

    :param log_level: log level specifier recognized by logging module
    :return: no return value; logging to console is started if not already enabled; calling this function multiple times
        only affects the log level but does not create additional log handlers
    """
    logger = logging.getLogger()
    if not _LogTracker.log_enabled:  # avoid adding multiple handlers in the same Python session
        log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler = logging.StreamHandler()
        handler.setFormatter(log_format)
        logger.addHandler(handler)
        _LogTracker.log_enabled = True
    logger.setLevel(log_level)
