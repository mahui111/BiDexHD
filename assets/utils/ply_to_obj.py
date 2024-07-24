import os
import trimesh


def ply_to_obj(ply_file, input_dir,  output_dir='./obj_meshes'):
    mesh = trimesh.load_mesh(os.path.join(input_dir, ply_file))
    offset = mesh.vertices.mean(axis=0, keepdims=True)
    obj_verts = mesh.vertices - offset

    # sample object points
    mesh = trimesh.Trimesh(vertices=obj_verts, faces=mesh.faces)
    mesh.export(os.path.join(output_dir, ply_file.replace('.ply', '.obj')))

if __name__ == '__main__':
    input_dir = ''
    output_dir = './obj/meshes'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in os.listdir(input_dir):
        if file.endswith('.ply'):
            ply_to_obj(file, input_dir, output_dir)
            print(f'{file} converted to obj')