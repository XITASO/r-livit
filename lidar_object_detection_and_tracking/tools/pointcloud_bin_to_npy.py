import numpy as np
import sys

def bin_to_npy(bin_file_path, npy_file_path):
    # Specify the data structure for each point: x, y, z, intensity
    # Read the .bin file
    points = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
    # Save the point cloud data as a .npy file
    np.save(npy_file_path, points)

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <bin_file_path> <npy_file_path>")
        sys.exit(1)
    
    bin_file_path = sys.argv[1]
    npy_file_path = sys.argv[2]
    
    bin_to_npy(bin_file_path, npy_file_path)
    print(f"Converted {bin_file_path} to {npy_file_path} successfully.")

if __name__ == "__main__":
    main()
