class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data : int | float, _parents=(), _op='', label =''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._prev = set(_parents)
        self._backward = lambda: None
        self.label = label
        self._op = _op # the operation that produced this node, for graphviz / debugging / etc

    def __add__(self, other):#operator overloading
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad = 1 * out.grad
            other.grad = 1 * out.grad
        out._backward = _backward
        
        return out
    
    def backward(self):
        topo: list[Value] = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    # print(child, "\n")
                    build_topo(child)
                topo.append(v)
                
        build_topo(self)
        self.grad = 1
        print(topo)
        print("reverse")
        # for i in range(len(topo)-1,-1, -1):
        #     print(topo[i])
        for node in reversed(topo):
            node._backward()

    def __radd__(self, other): # other + self
        return self + other

    def __repr__(self):
        return f"Value(data={self.data}, label={self.label})"