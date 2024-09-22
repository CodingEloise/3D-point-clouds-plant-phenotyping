import os

def process_xyz_files(directory):
    # 遍历指定路径下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.xyz'):
            # 判断文件名是否包含字母 'a'
            if 'a' not in filename:
                xyz_filepath = os.path.join(directory, filename)
                txt_filename = filename.replace('.xyz', '.txt')
                txt_filepath = os.path.join(directory, txt_filename)
                
                with open(xyz_filepath, 'r') as xyz_file, open(txt_filepath, 'w') as txt_file:
                    for line_number, line in enumerate(xyz_file):
                        # 跳过注释行
                        if line.startswith('//'):
                            continue
                        
                        columns = line.strip().split()
                        
                        # 如果行有8列，则保存前7列到对应的.txt文件

                        txt_file.write(' '.join(columns[:6]) + '\n')

                print(f"Processed {filename} and saved to {txt_filename}")
            else:
                print(f"Skipped {filename} (does not contain 'a')")

# 设置路径
directory_path = 'E:/3D-point-clouds-plant-phenotyping/data'  # 替换为你的文件路径

# 执行脚本
process_xyz_files(directory_path)
