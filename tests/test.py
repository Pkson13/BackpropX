import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from BackpropX.engine import Value
from BackpropX.draw import draw_dot


# sys.path.append(os.getcwd())
# print(sys.path)

# a = Value(2);a.label = "a"
# b = Value(3);b.label = 'b'
# c = a + b; c.label = 'c'
# c.backward()
# draw_dot(c)

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'
L.backward()
draw_dot(L)
