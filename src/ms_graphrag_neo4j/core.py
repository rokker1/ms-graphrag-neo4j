class MyObject:
    """
    A placeholder object that will be exported from the package.
    
    This is just a simple placeholder class that can be extended with
    actual functionality for your package.
    """
    
    def __init__(self, name="default"):
        self.name = name
        
    def get_name(self):
        """Return the name of the object."""
        return self.name
    
    def __repr__(self):
        return f"MyObject(name='{self.name}')"