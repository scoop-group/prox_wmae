""" 
This scripts renders all example plots for the paper 

  Baumgärtner, Herzog, Schmidt, Weiß
  The Proximal Map of the Weighted Mean Absolute Error
  arxiv: [2209.13545](https://arxiv.org/abs/2209.13545)
 
"""
import os
import demo_cameraman
import demo_deflection

# Run the cameraman code examples that generate the cameraman image denoising.
print("Running the checkerboard denoising Algorithm 2")
demo_cameraman.main()

# Run the deflection energy minimization example computations.
print("Solving the deflection energy minimization problem, Section 5")
demo_deflection.main()
# Call paraview via pvpython to render the deflection results.
print("Rendering results with paraview")
os.system("pvpython paraview_render_results.py")
