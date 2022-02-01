"""Export to disk"""


def export_mesh_to_obj(path, vertices, faces):
    """
    exports mesh as OBJ
    :param path: output path for the OBJ file
    :param vertices: Nx3 vertices
    :param faces: Mx3 faces
    :return: None
    """

    # write vertices starting with "v "
    # write faces starting with "f "

    # ###############
    # TODO: Implement
    with open(path, 'w') as f:
        for i in range(vertices.shape[0]):
            # f.write('v {:.3f} {:.3f} {:.3f}\n'.format(vertices[i][0], vertices[i][1], vertices[i][1]))
            f.write('v {} {} {}\n'.format(vertices[i][0], vertices[i][1], vertices[i][2]))
        for i in range(faces.shape[0]):
            f.write('f {} {} {}\n'.format(faces[i][0] + 1, faces[i][1] + 1, faces[i][2] + 1))
    # ###############


def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """

    # ###############
    # TODO: Implement
    with open(path, 'w') as f:
        for i in range(pointcloud.shape[0]):
            f.write('v {} {} {}\n'.format(pointcloud[i][0], pointcloud[i][1], pointcloud[i][2]))
    # ###############
