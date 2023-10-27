import os , sys
import json
from glob import glob
import argparse


def format_ref_images(file_paths):
    file_names = [os.path.basename(path).split('.')[0] for path in file_paths]
    result_dict = {k:[v] for k, v in zip(file_names, file_paths)}

    return result_dict

def remove_last_underscore(input_string):
    parts = input_string.split('_')

    if len(parts) > 1:
        parts = parts[:-1]

    result = '_'.join(parts)
    
    return result
    
def match_prefix(file_name,image_formats):
    for img_prefix in image_formats:
        if os.path.exists(file_name):
            return file_name
        else:
            current_prefix = file_name.split('/')[-1].split('.')[-1]
            file_name = file_name.replace(current_prefix,img_prefix)

    print('Warning: No match file in compared version!')
    return file_name

def find_value_for_key(file_name, dictionary):
    parts = file_name.split('/')
    last_part = parts[-1].split('.')[0]

    user_id = remove_last_underscore(last_part)

    if user_id in dictionary.keys():
        return dictionary[user_id]
    else:
        return None


if __name__=="__main__":


    parser = argparse.ArgumentParser(description='Description of your script')

    parser.add_argument('--ref_images', type=str, default='', help='Path to the user_id reference directory')
    parser.add_argument('--version1_dir', type=str,help='Path to version1 output result')
    parser.add_argument('--version2_dir', type=str,help='Path to version2 output result')
    parser.add_argument('--output_json', type=str,help='Path to output_datajson')

    args = parser.parse_args()

    image_formats = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    ref_images = []
    for image_format in image_formats:
        ref_images.extend(glob(os.path.join(args.ref_images, image_format)))
        
    if len(ref_images) == 0:
        print(f'Your reference dir contains no reference images. Set --ref_images to your user_id reference directory')
    else:
        print(f'reference images contains : {ref_images}')

    ref_dicts = format_ref_images(ref_images)
    # print(ref_dicts)

    result_data = []
    abs_path=True

    version1_dir = args.version1_dir
    version2_dir = args.version2_dir
    method_a = version1_dir.strip().split('/')[-1]
    method_b = version2_dir.strip().split('/')[-1]

    image_formats = ['jpg', 'jpeg', 'png', 'webp']

    for root, dirs, files in os.walk(version1_dir):
        for filename in files:
            if filename.split('.')[-1] in image_formats:
                file_path = os.path.join(root, filename)
                file_path2 = os.path.join(version2_dir, filename)     
                file_path2 = match_prefix(file_path2,image_formats)

                reference = find_value_for_key(file_path, ref_dicts)
                
                if reference:
                    if abs_path:
                        file_path = os.path.abspath(file_path)
                        file_path2 = os.path.abspath(file_path2)
                        reference = [os.path.abspath(t) for t in reference]
                    print(os.path.exists(file_path2))

                    if  os.path.exists(file_path2) and reference is not None:
                        data_item = {
                            "id": len(result_data), 
                            "method1": method_a,  
                            "img1": file_path,  
                            "method2": method_b,  
                            "img2": file_path2, 
                            "reference_imgs": reference 
                        }
        
                        result_data.append(data_item)
                    else:
                        pass
                else:
                    user_id = file_path.split('/')[-1].split('_')[0]
                    print(f'No matching user_id for {file_path}. Aborting! \
                            Please rename the ref image as {user_id}.jpg')

    output_json = args.output_json
    with open(output_json, "w") as json_file:
        json.dump(result_data, json_file, indent=4)

    print(f"Generated JSON file: {output_json}")