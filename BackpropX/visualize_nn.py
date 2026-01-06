
# from .nn import MLP
from manim import *
from BackpropX.nn import MLP

import random

class MLPscene(Scene):
  
  def construct(self) -> None:
    nn = MLP(3, [3,4,4,3])
    group = VGroup()
    layer_groups = []
    
    for layer in nn.layers:
      subgroup = VGroup()
      for neuron in layer.neurons:
        circle = Circle(radius=0.3, color=BLUE, stroke_width=2)
        circle.set_fill(BLUE, opacity=0.3)
        subgroup.add(circle)
      # Arrange neurons vertically within each layer
      subgroup.arrange(DOWN, buff=0.4)
      layer_groups.append(subgroup)
      group.add(subgroup)
    
    # Arrange all layers horizontally
    group.arrange(RIGHT, buff=1.5)
    
    # Add connections between layers
    lines = VGroup()
    for i in range(len(layer_groups) - 1):
      for neuron1 in layer_groups[i]:
        for neuron2 in layer_groups[i + 1]:
          line = Line(
            neuron1.get_right(), 
            neuron2.get_left(), 
            stroke_width=1, 
            color=GRAY,
            stroke_opacity=0.8
          )
          lines.add(line)
    
    # Combine everything
    full_network = VGroup(group, lines)
    
    # Center and scale to fit the screen
    full_network.move_to(ORIGIN)
    full_network.scale_to_fit_height(config.frame_height - 1)
    self.play(FadeIn(group))
    self.play(Create(lines, lag_ratio=0.1, run_time=1.5))
    
  
