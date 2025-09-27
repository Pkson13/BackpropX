import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from BackpropX.nn import MLP
from BackpropX.draw import draw_dot
# from BackpropX.engine import Value

nn = MLP(3, [4,4,1])
x = [2.0, 3.0, -1.0]

res = nn(x)
print(res)
res.backward()
draw_dot(res)