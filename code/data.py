class Location:
    """
    Represents a location with an ID, x-coordinate, and y-coordinate.
    """

    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

class Box:
    """
    Represents a box with its attributes.

    Attributes:
        id (int): The ID of the box.
        loc (str): The current location of the box.
        target (str): The target location for the box.
        prefer (str): The weight of the box.
    """

    def __init__(self, id, loc, target, prefer):
        self.id = id
        self.loc = loc
        self.target = target
        self.prefer = prefer
