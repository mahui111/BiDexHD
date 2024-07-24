import os


def single_mesh_to_urdf(template, mesh_file, output_dir):
    with open(template, "r") as f:
        urdf = f.read()
    urdf = urdf.replace("template.obj", mesh_file)
    with open(os.path.join(output_dir, mesh_file.replace(".obj", ".urdf")), "w") as f:
        f.write(urdf)


if __name__ == "__main__":
    template_urdf = "template.urdf"
    input_dir = "./obj/meshes"
    output_dir = "./obj/urdf"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in os.listdir(input_dir):
        if file.endswith(".obj"):
            single_mesh_to_urdf(template_urdf, file, output_dir)
            print(f"{file} converted to urdf")
