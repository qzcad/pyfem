#!/usr/bin/env python
# -*- coding: utf-8 -*-


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
    from numpy import zeros
    from numpy import float_
    from numpy import int_
    nodes = zeros((x_count * y_count, 2), dtype=float_)
    elements = zeros(((x_count - 1) * (y_count - 1), 4), dtype=int_)
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


def draw_vtk(nodes,
             elements,
             values=None,
             colors_count=8,
             contours_count=9,
             use_gray=False,
             title=None,
             background=(0.95, 0.95, 0.95),
             show_mesh=False,
             mesh_color=(0.8, 0.8, 0.8)):
    """
    Function draws planar unstructured mesh using vtk
    :param show_mesh: if true than mesh lines are shown
    :param mesh_color: color of mesh lines (polygons edges)
    :param contours_count: Contour lines count
    :param title: Title of the scalar bar
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
    lut.SetNumberOfTableValues(colors_count)
    lut.SetHueRange(0.66667, 0.0)
    if use_gray:
        lut.SetValueRange(1.0, 0.0)
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

    bcf = vtk.vtkContourFilter()
    bcf.SetInput(polydata)
    # bcf.SetNumberOfContours(contours_count)
    bcf.GenerateValues(contours_count, [values.min(), values.max()])
    bcf.Update()
    cfMapper = vtk.vtkPolyDataMapper()
    cfMapper.ImmediateModeRenderingOn()
    cfMapper.SetInput(bcf.GetOutput())
    cfMapper.SetScalarRange(values.min(), values.max())
    cfMapper.SetLookupTable(lut)
    cfMapper.ScalarVisibilityOff()
    cfActor = vtk.vtkActor()
    cfActor.SetMapper(cfMapper)
    cfActor.GetProperty().SetColor(.0, .0, .0)
    renderer.AddActor(cfActor)

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(polydata)
    else:
        mapper.SetInputData(polydata)
    mapper.SetScalarRange(values.min(), values.max())
    mapper.SetScalarVisibility(1)
    mapper.SetLookupTable(lut)
    actor.SetMapper(mapper)
    if show_mesh:
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(mesh_color)
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