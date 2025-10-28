import os
import json

def collect_genotypes(root_folder, output_file):
    genotypes = []

    for folder in os.listdir(root_folder):
        if "run" in folder:
            # open the folder and check if it contains JSON files
            folder_path = os.path.join(root_folder, folder)
            if os.path.isdir(folder_path):
                # open 'iteration_100.json' file
                json_file_path = os.path.join(folder_path, 'iteration_100.json')
                data = json.load(open(json_file_path, 'r'))
                for ind in data:
                    if "genotype" in ind:
                        genotypes.append(ind["genotype"])
   
    # Save collected genotypes to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(genotypes, f)

# Example usage
root_folder = "/media/cdv/nvme980pro/jessica/vae_experiments/autopsge/5parity/independent/standard_train/1.0/"
output_file = "genotypes_5parity.json"  # Replace with your desired output file name
collect_genotypes(root_folder, output_file)