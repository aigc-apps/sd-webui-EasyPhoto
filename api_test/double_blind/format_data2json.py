import os , sys
import json
from glob import glob
import argparse


def format_ref_images(file_paths):
    file_names = [os.path.basename(path).split('.')[0] for path in file_paths]
    result_dict = {k:[v] for k, v in zip(file_names, file_paths)}

    return result_dict

def find_value_for_key(file_name, dictionary):
    parts = file_name.split('/')
    last_part = parts[-1]
    words = last_part.split('_')
    # print(words, dictionary.keys())
    for word in words:
        if word in dictionary:
            return dictionary[word]
    
    return None


if __name__=="__main__":


    parser = argparse.ArgumentParser(description='Description of your script')

    parser.add_argument('--ref_images', type=str, default='', help='Path to the user_id reference directory')
    parser.add_argument('--version1_dir', type=str,help='Path to version1 output result')
    parser.add_argument('--version2_dir', type=str,help='Path to version2 output result')
    parser.add_argument('--output_json', type=str,help='Path to output_datajson')

    args = parser.parse_args()


    ref_images = glob(os.path.join(args.ref_images, '*.jpg')) + glob(os.path.join(args.ref_images, '*.jpg'))
    if len(ref_images) == 0:
        print(f'Your test_dirs/{args.ref_images} contains no reference images')
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

    for root, dirs, files in os.walk(version1_dir):
        for filename in files:
            if filename.endswith(".jpg"):
                file_path = os.path.join(root, filename)
                file_path2 = os.path.join(version2_dir, filename)
                reference = find_value_for_key(file_path, ref_dicts)
                if abs_path:
                    file_path = os.path.abspath(file_path)
                    file_path2 = os.path.abspath(file_path2)
                    reference = [os.path.abspath(t) for t in reference]
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

    output_json = args.output_json
    with open(output_json, "w") as json_file:
        json.dump(result_data, json_file, indent=4)

    print(f"Generated JSON file: {output_json}")