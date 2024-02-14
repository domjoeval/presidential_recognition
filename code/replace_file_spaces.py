import os

def replace_spaces_with_underscores(directory_path):
    # Get the list of files in the directory
    files = os.listdir(directory_path)

    # Iterate through each file
    for file_name in files:
        # Check if it's a file (not a directory)
        if os.path.isfile(os.path.join(directory_path, file_name)):
            # Replace spaces with underscores in the file name
            new_file_name = file_name.replace(' ', '_')

            # Construct the full paths for the old and new file names
            old_path = os.path.join(directory_path, file_name)
            new_path = os.path.join(directory_path, new_file_name)

            # Rename the file
            os.rename(old_path, new_path)
            print(f'Renamed: {file_name} -> {new_file_name}')

# Specify the directory path you want to process
directory_path = "./data/speech"

# Call the function to replace spaces with underscores
replace_spaces_with_underscores(directory_path)
