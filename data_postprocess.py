"""Processes data to a format that the robot can interpret to drive itself."""


def denormalize_vector(vector: float) -> float:
    """
    Denormalize the normalized vector angle so the robot can use it again.
    :param vector: normalized vector value between 0.0 and 1.0
    :return: vector value between 0.25 and 0.75
    """
    return (vector * 0.5) + 0.25


if __name__ == '__main__':
    pass
