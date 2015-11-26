#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def rectangular_quads(x_count, y_count, x_origin, y_origin, width, height):
    """
    Regular quadrilateral planar grid of rectangular region
    :type x_count: int
    :type y_count: int
    :type x_origin: float
    :type y_origin: float
    :type width: float
    :type height: float
    :param x_count: Nodes count in x-direction
    :param y_count: Nodes count in y-direction
    :param x_origin: X of region's origin (left bottom)
    :param y_origin: Y of region's origin (left bottom)
    :param width: Width of region
    :param height: Height of region
    :return: Tuple of numpy arrays: nodes [x_count*y_count; 2], elements[(x_count-1)*(y_count-1); 4]
    """
    nodes = np.zeros((x_count * y_count, 2))
    elements = np.zeros(((x_count - 1) * (y_count - 1), 4))
    hx = width / (x_count - 1)
    hy = height / (y_count - 1)
    for i in range(x_count):
        x = x_origin + i * hx
        for j in range(y_count):
            y = y_origin + j * hy
            nodes[i * y_count + j, 0] = x
            nodes[i * y_count + j, 1] = y
    for i in range(x_count - 1):
        for j in range(y_count - 1):
            elements[i * (y_count - 1) + j, 0] = i * y_count + j
            elements[i * (y_count - 1) + j, 1] = (i + 1) * y_count + j
            elements[i * (y_count - 1) + j, 2] = (i + 1) * y_count + (j + 1)
            elements[i * (y_count - 1) + j, 3] = i * y_count + (j + 1)
    return nodes, elements


def draw_vtk(nodes, elements, values=None, colors_count=8, use_gray=False, title=None, background=(0.9, 0.9, 0.9)):
    """
    Function draws planar unstructured mesh using vtk
    :param background: Background RGB-color value
    :param use_gray: if true than gray-scale colormap is used
    :param colors_count: Colors count for values visualization
    :param nodes: nodes array [nodes_count; 2]
    :param elements: elements array [elements_count; element_nodes]
    :param values: values array (coloring rule)
    :return: nothing
    """
    import vtk
    points = vtk.vtkPoints()
    for n in nodes:
        points.InsertNextPoint([n[0], n[1], 0.0])
    cells_array = vtk.vtkCellArray()
    for el in elements:
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(el))
        for i in range(len(el)):
            polygon.GetPointIds().SetId(i, el[i])
        cells_array.InsertNextCell(polygon)
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfColors(colors_count)
    if use_gray:
        lut.SetValueRange(0.0, 1.0)
        lut.SetSaturationRange(0.0, 0.0) # no color saturation
        lut.SetRampToLinear()
    lut.Build()
    actor = vtk.vtkActor()
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(background)
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    renderer.AddActor(actor)
    if values is None:
        values = nodes[:, 0]
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells_array)
    scalars = vtk.vtkFloatArray()
    for v in values:
        scalars.InsertNextValue(v)
    polydata.GetPointData().SetScalars(scalars)
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(polydata)
    else:
        mapper.SetInputData(polydata)
    mapper.SetScalarRange(values.min(), values.max())
    mapper.SetScalarVisibility(1)
    mapper.SetLookupTable(lut)
    actor.SetMapper(mapper)
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetOrientationToHorizontal()
    scalar_bar.SetLookupTable(lut)
    if title is not None:
        scalar_bar.SetTitle(title)
    scalar_bar_widget = vtk.vtkScalarBarWidget()
    scalar_bar_widget.SetInteractor(render_window_interactor)
    scalar_bar_widget.SetScalarBarActor(scalar_bar)
    scalar_bar_widget.On()
    render_window.Render()
    render_window_interactor.Start()

if __name__ == "__main__":
    (nodes, quads) = rectangular_quads(21, 11, 0, 0, 20, 10)
    # print(quads)
    draw_vtk(nodes=nodes, elements=quads, use_gray=False, title='Test')
    print('Привет мир!')