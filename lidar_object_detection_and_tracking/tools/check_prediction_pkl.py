import pickle
import pprint

def load_and_print_pkl(pkl_file, txt_log_file):
    try:
        # Load data from the .pkl file
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        # Print the data to the console
        print("Data in the .pkl file:")
        pprint.pprint(data)

        # Save the data to a .txt log file
        with open(txt_log_file, 'w') as log_file:
            # Use pprint to format the data nicely in the text file
            log_file.write(pprint.pformat(data))
        
        print(f"Data has been successfully saved to {txt_log_file}")

    except FileNotFoundError:
        print(f"Error: The file '{pkl_file}' was not found.")
    except pickle.UnpicklingError:
        print(f"Error: Failed to unpickle the file '{pkl_file}'. It may not be a valid .pkl file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Specify the paths for the .pkl file and the output .txt log file
    pkl_file_path = '/mnt/data/kitti/r_livit_detection/r_livit_dbinfos_train.pkl'  # Replace with your .pkl file path
    txt_log_file_path = '/home/user/workspace/output/home/user/workspace/tools/cfgs/r_livit_models/pointpillar/default/eval/eval_with_train/epoch_80/val/groundtruth_log.txt'  # Replace with your desired output file path
    
    # Call the function
    load_and_print_pkl(pkl_file_path, txt_log_file_path)