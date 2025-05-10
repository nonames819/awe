import h5py,os

def read_hdf5(task):
    file_path = "/home/caohaidong/code/awe/robomimic/datasets/"+task+"/ph/low_dim.hdf5"
    output = "test/output_"+task+"_new.txt"
    file_info = "file:"+task+file_path

    with open(output, 'w') as f:
        def print_hdf5_structure(root, child, print_value = False):
            # print(root)
            if root == '/data/demo_1' or root == '/mask': 
                print_value = True
            if isinstance(child, h5py.Group):
                f.write(f"{root} - Group\n")
                for key in child.keys():
                    print_hdf5_structure(f"{root}/{key}", child[key],print_value)
            elif isinstance(child, h5py.Dataset):
                f.write(f"{root} - Dataset, Size: {child.shape}, Data type: {child.dtype}\n")
                if print_value:
                    f.write(f"{root} - value: {child[:]}\n")

        with h5py.File(file_path, 'r') as file:
            f.write(file_info+"\n\n")
            f.write("HDF5 structure:\n")
            print_hdf5_structure('', file, print_value=False)

    dir, _ = os.path.split(file_path)
    command = "du -sh " + dir
    os.system(command)

if __name__ == "__main__":
    # task: square, can, lift
    task = ["square", "can", "lift"]
    for t in task:
        read_hdf5(t)

# content = h5py.File(file_path, 'r')
# print(content.keys())
# print(content['data'].shape)
# print(content['mask'].shape)
