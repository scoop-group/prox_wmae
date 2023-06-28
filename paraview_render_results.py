""" This module renders the resulting deflection results using pvpython (paraview) and exports them into .png-files. 

This imports the solutions generated in "demo_deflection.py" and renders the images in Figure (5.1) in the corresponding paper
  Baumgärtner, Herzog, Schmidt, Weiß
  The Proximal Map of the Weighted Mean Absolute Error
  arxiv: [2209.13545](https://arxiv.org/abs/2209.13545)

The rendered images are stored into the files "deflection_disk.png" and "deflection_L.png". 

Call this module with pvpython:

.. code:

    pvpython paraview_render_results.py

```

This requires the files "Omega_1.xmdf" and "Omega_2.xdmf" to contain the resulting deflection in the field "y", 
as is done in "demo_deflection.py". 

"""

import os
import paraview.simple as pv

def render_result(function , data_file, export_file):
    """ renders the :code:`function` from xdmf-file :code:`data_file` and stores it into :code:`export_file`

    Parameters
    ----------
    function : string 
        function name in the .xdmf-file
    data_file : string
        filepath of the .xdmf file storign the function values of function :code:`function` in timestamp 0
    export_file : string 
        filepath of the exported rendered image

    """
    # setup paraview views
    layout = pv.CreateLayout(name = 'Layout' )
    pv.SetActiveView(None)
    renderView = pv.CreateView('RenderView')
    pv.AssignViewToLayout(renderView,layout,hint=0)
    renderView.InteractionMode = '3D'
    renderView.OrientationAxesVisibility = 0
    renderView.ResetCamera(False)

    # open file, extract deflection function values
    xdmf_handle = pv.Xdmf3ReaderS(registrationName = function, FileName=[os.getcwd() + '/' + data_file ])
    pv_zxdmfDisplay = pv.Show(xdmf_handle, renderView, 'UnstructuredGridRepresentation')
    pv_zxdmfDisplay.Representation = 'Surface'
    # hide 2D representation
    pv.Hide(xdmf_handle,renderView)
    # create warped representation of function values
    warp_source = xdmf_handle 
    warp = pv.WarpByScalar(registrationName = function +'_warp', Input=warp_source)
    warp.Scalars = ['POINTS', function]
    warpDisplay = pv.Show(warp, renderView, 'UnstructuredGridRepresentation')
    warpDisplay.Representation = 'Surface With Edges'
    warpDisplay.ColorArrayName = ['POINTS', function]
    warp.ScaleFactor = 10
    warpDisplay.SetScalarBarVisibility(renderView,False)
    # set colorspace to represent thresholds defined in demo_deflection.py
    # define step colors
    colors = [[0,119,187],[51,187,238],[0,153,136],[238,119,51],[204,51,17]]
    # define step-heights d^i 
    steps = [0.0,0.01,0.02,0.03,0.04]
    rgb_points = []
    # create step-colorspace representation, which is a flat tuple of the step-value and its rgb-color-code
    for i in range(len(steps)):
        rgb_points.append(steps[i])
        rgb_points.append(colors[i][0])
        rgb_points.append(colors[i][1])
        rgb_points.append(colors[i][2])
    # edit paraviews color-function
    LUT = pv.GetColorTransferFunction(function)
    LUT.ColorSpace = 'Step'
    LUT.RGBPoints = rgb_points
    LUT.AboveRangeColor = colors[-1]
    # add annotations for color legend in flat tuple of corresponding values followed by their annotation in the legend
    ScalarBar = pv.GetScalarBar(LUT, renderView)
    LUT.Annotations = ['0', '$0$', '0.01', '$0.01 = d_1$', '0.02', '$0.02 = d_2$', '0.03', '$0.03 = d_3$','0.04', '$0.04 = d_4$',]
    ScalarBar.Title = '$z$'
    ScalarBar.ComponentTitle = ''
    ScalarBar.WindowLocation = 'Upper Right Corner'
    ScalarBar.TitleColor = [0.0, 0.0, 0.0]
    ScalarBar.TitleFontSize = 32
    ScalarBar.LabelColor = [0.0, 0.0, 0.0]
    ScalarBar.LabelFontSize = 32
    ScalarBar.AutomaticLabelFormat = 0
    ScalarBar.DrawTickMarks = 0
    ScalarBar.DrawTickLabels = 0
    ScalarBar.AddRangeLabels = 0
    ScalarBar.TextPosition = 'Ticks left/bottom, annotations right/top'
    renderView.Update()
    # adapt camera position
    layout.SetSize(1826, 1088) # (3652,2176)
    renderView.CameraPosition = [1.55, 2.06, 1.22]
    renderView.CameraFocalPoint = [0.52, 0.59, 0.15]
    renderView.CameraViewUp = [-0.26, -0.44, 0.86]
    renderView.CameraParallelScale = 0.79
    renderView.Update()
    # export screenshot as png
    pv.SaveScreenshot(export_file, renderView, ImageResolution=[3652, 2176], TransparentBackground=1)
    return

if __name__ == '__main__' or True: 
    render_result("z","Omega_1.xdmf",os.getcwd()+'/deflection_disk.png')
    render_result("z","Omega_2.xdmf",os.getcwd()+'/deflection_L.png')
