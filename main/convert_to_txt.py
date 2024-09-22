import os

def convert_xyz_to_txt(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith('.xyz'):
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename.replace('.xyz', '.txt'))
            with open(input_file_path, 'r') as fin, open(output_file_path, 'w') as fout:
                for line in fin:
                    line = line.strip()
                    if not line or line.startswith('//'):
                        continue
                    parts = line.split()
                    try:
                        if len(parts) == 6:
                            x, y, z = map(float, parts[:3])
                            r, g, b = map(float, parts[3:6])
                            label = -1
                            fout.write(f"{x} {y} {z} {r} {g} {b} {label}\n")
                        elif len(parts) == 3:
                            x, y, z = map(float, parts[:3])
                            r, g, b = 0, 0, 0
                            label = -1
                            fout.write(f"{x} {y} {z} {r} {g} {b} {label}\n")
                        elif len(parts) == 8:
                            x, y, z, r, g, b, label = map(float, parts[:7])
                            fout.write(f"{x} {y} {z} {r} {g} {b} {label}\n")
                        else:
                            print(f"Unexpected format in file {filename}: {line.strip()}")
                    except ValueError as e:
                        print(f"Error processing line in file {filename}: {line.strip()}. Error: {e}")
                        continue

            print(f"File {input_file_path} saved as {output_file_path}")
if __name__ == '__main__':
    input_directory = 'E:/3D-point-clouds-plant-phenotyping/data/' 
    output_directory = 'E:/3D-point-clouds-plant-phenotyping/main/data/'  
    convert_xyz_to_txt(input_directory, output_directory)
