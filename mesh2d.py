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
    hx = width / float(x_count - 1)
    hy = height / float(y_count - 1)
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


def rectangular_triangles(x_count, y_count, x_origin, y_origin, width, height):
    """
    Regular triangular planar grid of rectangular region
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
    :return: Tuple of numpy arrays: nodes [x_count*y_count; 2], elements[2 * (x_count-1)*(y_count-1); 3]
    """
    from numpy import zeros
    from numpy import float_
    from numpy import int_
    nodes = zeros((x_count * y_count, 2), dtype=float_)
    elements = zeros((2 * (x_count - 1) * (y_count - 1), 3), dtype=int_)
    hx = width / (x_count - 1)
    hy = height / (y_count - 1)
    cx = x_origin + width / 2.0
    cy = y_origin + height / 2.0
    for i in range(x_count):
        x = x_origin + i * hx
        for j in range(y_count):
            y = y_origin + j * hy
            nodes[i * y_count + j, 0] = x
            nodes[i * y_count + j, 1] = y
    for i in range(x_count - 1):
        for j in range(y_count - 1):
            if nodes[i * y_count + j, 0] >= cx and nodes[(i + 1) * y_count + j, 0] >= cx and nodes[(i + 1) * y_count + (j + 1), 0] >= cx and nodes[i * y_count + (j + 1), 0] >= cx and\
                    nodes[i * y_count + j, 1] >= cy and nodes[(i + 1) * y_count + j, 1] >= cy and nodes[(i + 1) * y_count + (j + 1), 1] >= cy and nodes[i * y_count + (j + 1), 1] >= cy or\
                    nodes[i * y_count + j, 0] <= cx and nodes[(i + 1) * y_count + j, 0] <= cx and nodes[(i + 1) * y_count + (j + 1), 0] <= cx and nodes[i * y_count + (j + 1), 0] <= cx and\
                    nodes[i * y_count + j, 1] <= cy and nodes[(i + 1) * y_count + j, 1] <= cy and nodes[(i + 1) * y_count + (j + 1), 1] <= cy and nodes[i * y_count + (j + 1), 1] <= cy:
                elements[2 * (i * (y_count - 1) + j), 0] = i * y_count + j
                elements[2 * (i * (y_count - 1) + j), 1] = (i + 1) * y_count + j
                elements[2 * (i * (y_count - 1) + j), 2] = i * y_count + (j + 1)
                elements[2 * (i * (y_count - 1) + j) + 1, 0] = (i + 1) * y_count + j
                elements[2 * (i * (y_count - 1) + j) + 1, 1] = (i + 1) * y_count + (j + 1)
                elements[2 * (i * (y_count - 1) + j) + 1, 2] = i * y_count + (j + 1)
            else:
                elements[2 * (i * (y_count - 1) + j), 0] = i * y_count + j
                elements[2 * (i * (y_count - 1) + j), 1] = (i + 1) * y_count + j
                elements[2 * (i * (y_count - 1) + j), 2] = (i + 1) * y_count + (j + 1)
                elements[2 * (i * (y_count - 1) + j) + 1, 0] = i * y_count + j
                elements[2 * (i * (y_count - 1) + j) + 1, 1] = (i + 1) * y_count + (j + 1)
                elements[2 * (i * (y_count - 1) + j) + 1, 2] = i * y_count + (j + 1)
    return nodes, elements


def annular_sector(xi_count, eta_count, alpha, min_radius, max_radius):
    """
    Routine generates structured quadrilateral grid for an annular sector
    :param xi_count: Nodes count in the angular direction
    :param eta_count: Nodes count in the radial direction
    :param alpha: The central angle in radians
    :param min_radius: The outer radius of the annulus
    :param max_radius: The inner radius of the annulus
    :return: Tuple of numpy arrays: nodes [x_count*y_count; 2], elements[(x_count-1)*(y_count-1); 4]
    """
    from numpy import zeros
    from numpy import float_
    from numpy import int_
    from math import cos, sin, sqrt, tanh
    nodes = zeros((xi_count * eta_count, 2), dtype=float_)
    elements = zeros(((xi_count - 1) * (eta_count - 1), 4), dtype=int_)
    hxi = 1.0 / float(xi_count - 1)
    heta = 1.0 / float(eta_count - 1)
    p = sqrt((min_radius - min_radius * cos(alpha * hxi))**2.0 + (min_radius * sin(alpha * hxi))**2.0)
    q = 2.0 - p
    for i in range(xi_count):
        for j in range(eta_count):
            eta = float(j) * heta
            s = p * eta + (1.0 - p) * (1.0 - tanh(q * (1.0 - eta)) / tanh(q))
            x = (min_radius + (max_radius - min_radius) * s) * cos(alpha * float(i) * hxi)
            y = (min_radius + (max_radius - min_radius) * s) * sin(alpha * float(i) * hxi)
            nodes[i * eta_count + j, 0] = x
            nodes[i * eta_count + j, 1] = y
    for i in range(xi_count - 1):
        for j in range(eta_count - 1):
            elements[i * (eta_count - 1) + j, 0] = i * eta_count + j
            elements[i * (eta_count - 1) + j, 1] = i * eta_count + (j + 1)
            elements[i * (eta_count - 1) + j, 2] = (i + 1) * eta_count + (j + 1)
            elements[i * (eta_count - 1) + j, 3] = (i + 1) * eta_count + j

    return nodes, elements


def draw_vtk(nodes,
             elements,
             values=None,
             colors_count=256,
             contours_count=10,
             use_gray=False,
             title=None,
             background=(0.95, 0.95, 0.95),
             show_mesh=False,
             mesh_color=(0.25, 0.25, 0.25),
             use_cell_data=False,
             show_labels=False,
             show_axes=False):
    """
    Function draws planar unstructured mesh using vtk
    :param show_axes: if it equals true than axes is drawn
    :param use_cell_data: if it equals true than cell data is used to colorize zones
    :param show_labels: if it equals true than labels are shown
    :param show_mesh: if it equals true than mesh lines are shown
    :param mesh_color: color of mesh lines (polygons edges)
    :param contours_count: Contour lines count
    :param title: Title of the scalar bar
    :param background: Background RGB-color value
    :param use_gray: if it equals true than gray-scale colormap is used
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
        lut.SetSaturationRange(0.0, 0.0)  # no color saturation
        lut.SetRampToLinear()
    lut.Build()
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(background)
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    bcf_actor = vtk.vtkActor()
    bcf_mapper = vtk.vtkPolyDataMapper()
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetPolys(cells_array)

    if values is not None:
        scalars = vtk.vtkFloatArray()
        for v in values:
            scalars.InsertNextValue(v)
        poly_data.GetPointData().SetScalars(scalars)
        bcf = vtk.vtkBandedPolyDataContourFilter()
        if vtk.VTK_MAJOR_VERSION <= 5:
            bcf.SetInput(poly_data)
        else:
            bcf.SetInputData(poly_data)
        bcf.SetNumberOfContours(contours_count)
        bcf.GenerateValues(contours_count, [values.min(), values.max()])
        bcf.SetNumberOfContours(contours_count + 1)
        bcf.SetScalarModeToValue()
        bcf.GenerateContourEdgesOn()
        bcf.Update()
        bcf_mapper.ImmediateModeRenderingOn()
        if vtk.VTK_MAJOR_VERSION <= 5:
            bcf_mapper.SetInput(bcf.GetOutput())
        else:
            bcf_mapper.SetInputData(bcf.GetOutput())
        bcf_mapper.SetScalarRange(values.min(), values.max())
        bcf_mapper.SetLookupTable(lut)
        bcf_mapper.ScalarVisibilityOn()
        if use_cell_data:
            bcf_mapper.SetScalarModeToUseCellData()

        bcf_actor.SetMapper(bcf_mapper)
        renderer.AddActor(bcf_actor)
        edge_mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            edge_mapper.SetInput(bcf.GetContourEdgesOutput())
        else:
            edge_mapper.SetInputData(bcf.GetContourEdgesOutput())
        edge_mapper.SetResolveCoincidentTopologyToPolygonOffset()
        edge_actor = vtk.vtkActor()
        edge_actor.SetMapper(edge_mapper)
        if use_gray:
            edge_actor.GetProperty().SetColor(0.0, 1.0, 0.0)
        else:
            edge_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
        renderer.AddActor(edge_actor)

        if show_labels:
            mask = vtk.vtkMaskPoints()
            if vtk.VTK_MAJOR_VERSION <= 5:
                mask.SetInput(bcf.GetOutput())
            else:
                mask.SetInputData(bcf.GetOutput())
            mask.SetOnRatio(bcf.GetOutput().GetNumberOfPoints() / 20)
            mask.SetMaximumNumberOfPoints(20)
            # Create labels for points - only show visible points
            visible_points = vtk.vtkSelectVisiblePoints()
            visible_points.SetInputConnection(mask.GetOutputPort())
            visible_points.SetRenderer(renderer)
            ldm = vtk.vtkLabeledDataMapper()
            ldm.SetInputConnection(mask.GetOutputPort())
            ldm.SetLabelFormat("%.2E")
            ldm.SetLabelModeToLabelScalars()
            text_property = ldm.GetLabelTextProperty()
            text_property.SetFontFamilyToArial()
            text_property.SetFontSize(10)
            if use_gray:
                text_property.SetColor(0.0, 1.0, 0.0)
            else:
                text_property.SetColor(0.0, 0.0, 0.0)
            text_property.ShadowOff()
            text_property.BoldOff()
            contour_labels = vtk.vtkActor2D()
            contour_labels.SetMapper(ldm)
            renderer.AddActor(contour_labels)

        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetOrientationToHorizontal()
        scalar_bar.SetLookupTable(lut)
        if title is not None:
            scalar_bar.SetTitle(title)
        scalar_bar_widget = vtk.vtkScalarBarWidget()
        scalar_bar_widget.SetInteractor(render_window_interactor)
        scalar_bar_widget.SetScalarBarActor(scalar_bar)
        scalar_bar_widget.On()
    else:
        if vtk.VTK_MAJOR_VERSION <= 5:
            bcf_mapper.SetInput(poly_data)
        else:
            bcf_mapper.SetInputData(poly_data)
        bcf_actor.GetProperty().SetColor(0.0, 1.0, 0.0)

    if show_mesh:
        bcf_actor.GetProperty().EdgeVisibilityOn()
        bcf_actor.GetProperty().SetEdgeColor(mesh_color)

    bcf_actor.SetMapper(bcf_mapper)
    renderer.AddActor(bcf_actor)

    if show_axes:
        axes = vtk.vtkAxesActor()
        renderer.AddActor(axes)

    render_window.Render()
    render_window_interactor.Start()

if __name__ == "__main__":
    (nodes, quads) = rectangular_quads(11, 11, 0, 0, 1, 1)
    draw_vtk(nodes=nodes, elements=quads, title='Quadrilateral Grid', show_mesh=True, show_axes=True)
    (nodes, triangles) = rectangular_triangles(21, 11, 0, 0, 20, 10)
    draw_vtk(nodes=nodes, elements=triangles, title='Triangular Grid', show_mesh=True)
    (nodes, quads) = annular_sector(11, 11, 0.8, 1.0, 4.0)
    draw_vtk(nodes=nodes, elements=quads, show_mesh=True)
