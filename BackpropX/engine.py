class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data : int | float, _parents=(), _op='', label =''):
        self.data = data
        # internal variables used for autograd graph construction
        self._prev = set(_parents)
        self.label = label
        self._op = _op # the operation that produced this node, for graphviz / debugging / etc

    def __add__(self, other):#operator overloading
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __radd__(self, other): # other + self
        return self + other

    def __repr__(self):
        return f"Value(data={self.data})"