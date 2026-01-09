
# from .nn import MLP
from manim import *
from manim.camera.camera import Camera
from manim.renderer.cairo_renderer import CairoRenderer
from manim.renderer.opengl_renderer import OpenGLRenderer
from BackpropX.nn import MLP
from manim.camera.three_d_camera import ThreeDCamera

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize MLP neural network')
parser.add_argument('--nin', type=int, required=True, help='Number of inputs')
parser.add_argument('--nouts', type=int, nargs='+', required=True, 
                    help='List of layer sizes (e.g., --nouts 3 4 4 3)')
parser.add_argument("--quality", type=str, required=True, help="Rendering quality", choices=["low_quality", "medium_quality", "high_quality", "fourk_quality"])
args, _ = parser.parse_known_args()

class MLPscene(Scene):
  def __init__(self,nin: int, nouts: list[int], renderer: CairoRenderer | OpenGLRenderer | None = None, camera_class: type[Camera] = ThreeDCamera, always_update_mobjects: bool = False, random_seed: int | None = None, skip_animations: bool = False) -> None:
    super().__init__(renderer, camera_class, always_update_mobjects, random_seed, skip_animations)
    self.nin = nin
    self.nouts = nouts
  
  def construct(self) -> None:
    nn = MLP(self.nin, self.nouts)
    group = VGroup()
    layer_groups = []
    
    for layer in nn.layers:
      subgroup = VGroup()
      for neuron in layer.neurons:
        circle = Circle(radius=0.3, color=BLUE, stroke_width=2)
        circle.set_fill(BLUE, opacity=0.3)
        weightvalue = sum(x.data for x in neuron.w)
        biasvalue = neuron.b.data
        textgrp = VGroup()

        weight_text = Text(f"W = {weightvalue: .2f}", font_size=6)
        bias_text = Text(f"B = {biasvalue: .2f}", font_size=6)
        textgrp.add(weight_text, bias_text)
        textgrp.arrange(DOWN, buff=0.1)
        circle.add(textgrp)
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

if __name__ == "__main__":
  print(f"Visualizing MLP with {args.nin} inputs and layers {args.nouts}")
  with tempconfig({"quality": args.quality, "preview": True}):
    nnscene = MLPscene(nin=args.nin, nouts=args.nouts)
    nnscene.render()
    
  
