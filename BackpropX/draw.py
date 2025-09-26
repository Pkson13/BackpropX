from graphviz import Digraph
from BackpropX.engine import Value

from graphviz import Digraph

def trace(root : Value):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root : Value):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}, filename=f'output/{root.label}graph') # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f  }" % (n.label, n.data,n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  dot.render(view=True)

  return dot

# dot = graphviz.Digraph("Neural net graphs", filename='doctest-output/round-table.gv')
# dot.node('A', 'King Arthur')  
# dot.node('B', 'Sir Bedevere the Wise')
# dot.node('L', 'Sir Lancelot the Brave')

# dot.edges(['AB', 'AL'])
# dot.edge('B', 'L', constraint='false')
# print(dot)
# # dot.render('doctest-output/round-table.gv', view=True).replace('\\', '/')
# dot.render(view=True)