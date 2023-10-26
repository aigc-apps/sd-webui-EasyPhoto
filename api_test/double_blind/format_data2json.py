import os , sys
import json
from glob import glob


def format_ref_images(file_paths):
    # 提取每个文件名的最后一部分
    file_names = [path.split('/')[-1] for path in file_paths]

    # 将文件名拆分为单词
    file_words = [name.split('_') for name in file_names]

    # 初始化共同部分和独立部分
    common_parts = file_words[0]
    independent_parts = []

    # 遍历文件名中的单词列表
    for words in file_words[1:]:
        temp_common = []
        temp_independent = []

        for word in words:
            if word in common_parts:
                temp_common.append(word)
            else:
                temp_independent.append(word)

        common_parts = temp_common
        independent_parts.extend(temp_independent)

    # 过滤所有单词以找到每个单词的独立部分
    all_words = [word for words in file_words for word in words]
    independent_parts = [word for word in all_words if word not in common_parts]

    # 将独立部分和源文件路径创建字典
    result_dict = {part: [] for part in independent_parts}

    for path, words in zip(file_paths, file_words):
        for part in words:
            if part in independent_parts:
                result_dict[part].append(path)

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


REF_DIR = 'ref_image'
TEM_DIR = 'templates'

if __name__=="__main__":

    
    # input_folder = "your_input_folder_path"
    input_folder = sys.argv[1]

    subfolders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
    original_subdirs = subfolders

    subfolders.remove(REF_DIR)
    subfolders.remove(TEM_DIR)
    if len(subfolders) == 2:
        method_a = subfolders[0]
        method_b = subfolders[1]
        print(f'Your test method is {method_a}  and {method_b}')
    else:
        print(f'Your test dirs contains {original_subdirs}, which should be [ref_image,templates,\'method_a_name\', \'method_b_name\' ]')
        exit()
    

    # ref image get
    ref_images = glob(os.path.join(input_folder, REF_DIR, '*.jpg')) + glob(os.path.join(input_folder, REF_DIR, '*.png'))
    if len(ref_images) == 0:
        print(f'Your test_dirs/{REF_DIR} contains no reference images')
    else:
        print(f'reference images contains : {ref_images}')

    ref_dicts = format_ref_images(ref_images)

    # print(ref_dicts)
    # 初始化一个空的列表，用于存储结果
    result_data = []

    target_dir = os.path.join(input_folder, method_a)
    # 遍历文件夹
    for root, dirs, files in os.walk(target_dir):
        for filename in files:
            # 检查文件扩展名是否是.jpg
            if filename.endswith(".jpg"):
                # 构建文件的完整路径
                file_path = os.path.join(root, filename)
                file_path2 = file_path.replace(method_a, method_b)
                reference = find_value_for_key(file_path, ref_dicts)
                
                print(file_path, file_path2, reference)
                if 1:
                    file_path = os.path.abspath(file_path)
                    file_path2 = os.path.abspath(file_path2)
                    reference = [os.path.abspath(t) for t in reference]
                if  os.path.exists(file_path2) and reference is not None:
                    data_item = {
                        "id": len(result_data),  # 自定义ID，可以根据需要修改
                        "method1": method_a,  # 方法1的名称，从文件名中提取
                        "img1": file_path,  # 图像1的路径
                        "method2": method_b,  # 方法2的名称，自定义
                        "img2": file_path2,  # 图像2的路径，自定义
                        "reference_imgs": reference # 参考图像的路径，自定义
                    }
                    # 将数据项添加到结果列表
                    result_data.append(data_item)
                else:
                    pass

    # 将结果保存为JSON文件
    output_json = os.path.join(input_folder, "output.json")
    with open(output_json, "w") as json_file:
        json.dump(result_data, json_file, indent=4)

    print(f"Generated JSON file: {output_json}")