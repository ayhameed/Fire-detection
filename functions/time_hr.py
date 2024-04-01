import time

def seconds_to_hms(seconds):
    """
    Converts the given number of seconds to hours, minutes, and seconds.

    Args:
        seconds (int): The number of seconds to convert.

    Returns:
        str: A string representation of the converted time in the format "hours.minutes.seconds".
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours}.{minutes}.{seconds}"