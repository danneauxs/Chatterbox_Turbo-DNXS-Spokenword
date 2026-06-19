class AttrDict(dict):
    """Represents a dictionary that allows attribute-style access to its keys.
    Attributes can be accessed using dot notation instead of bracket notation.
    This makes it easier and more intuitive to work with nested dictionaries.
    Internally, this class maintains both dictionary and object behaviors seamlessly.
    """
    def __init__(self, *args, **kwargs):
        """Converts keyword arguments into dictionary attributes.
        Args:
        *args: Positional arguments passed to superclass constructor.
        **kwargs: Keyword arguments to be converted to instance attributes.
        Returns:
        None
        """
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
