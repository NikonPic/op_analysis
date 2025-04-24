# %%
import os

def make_folder(folder_path):
    """
    Create a folder if it does not exist.

    Args:
        folder_path (str): The path of the folder to create.
    """
    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"Folder '{folder_path}' created successfully.")
    except Exception as e:
        print(f"Error creating folder '{folder_path}': {e}")


new_folder_path = './internal'  # Specify the folder path you want to create
make_folder(new_folder_path)

with open('nums.txt', 'r') as f:
    nums = f.readlines()
    nums = [int(num.strip()) for num in nums]  # Convert to integers

# copy over all images from the folder to the new folder if num in name of image
all_img_names = os.listdir('./images')
for img_name in all_img_names:
    for num in nums:
        if str(num) in img_name:
            src = os.path.join('./images', img_name)
            dst = os.path.join(new_folder_path, img_name)
            try:
                os.rename(src, dst)  # Move the file
                print(f"Moved '{img_name}' to '{new_folder_path}'")
            except Exception as e:
                print(f"Error moving '{img_name}': {e}")
                
# %%

# go trough new folder and remove "Sn" from all names in that folder
for img_name in os.listdir(new_folder_path):
    new_name = img_name.replace('Sn', '')
    src = os.path.join(new_folder_path, img_name)
    dst = os.path.join(new_folder_path, new_name)
    try:
        os.rename(src, dst)  # Rename the file
        print(f"Renamed '{img_name}' to '{new_name}'")
    except Exception as e:
        print(f"Error renaming '{img_name}': {e}")
# %%
