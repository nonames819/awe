import h5py,os

file_path = "/home/caohaidong/code/awe/robomimic/datasets/square/ph/low_dim.hdf5"
output = "test/output_square.txt"

with open(output, 'w') as f:

    def print_hdf5_structure(root, son, print_value = False):
        # print(root)
        if root == '/data/demo_0' or root == '/mask': 
            print_value = True
        if isinstance(son, h5py.Group):
            f.write(f"{root} - Group\n")
            for key in son.keys():
                print_hdf5_structure(f"{root}/{key}", son[key],print_value)
        elif isinstance(son, h5py.Dataset):
            f.write(f"{root} - Dataset, Size: {son.shape}, Data type: {son.dtype}\n")
            if print_value:
                f.write(f"{root} - value: {son[:]}\n")

    with h5py.File(file_path, 'r') as file:
        f.write("HDF5 structure:\n")
        print_hdf5_structure('', file, print_value=False)


dir, _ = os.path.split(file_path)
command = "du -sh " + dir
os.system(command)

# content = h5py.File(file_path, 'r')
# print(content.keys())
# print(content['data'].shape)
# print(content['mask'].shape)
