import pandas as pd
import json
import os.path


# Step 1: We convert the dictionary of all questions in string format into a JSON format, written into a .txt file
# qn_dictionary must be a variable in string format, saved in MainV5 python module.
def convert_to_JSON(qn_dictionary, pdf_path):
    # reference the directory for each converted pdf to json to be saved into
    sub_dir = str("ConvertedPDFs/" + pdf_path.split('/')[-1].replace('.pdf','') + "/")
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    # saving the json file into the created/existing directory folder
    save_path = sub_dir
    with open('JSON_dictionary.txt', 'w+') as outfile:
        json.dump(qn_dictionary, outfile)
        os.path.join(str(save_path), str(outfile))


# Step 2: We transfer the JSON format .txt file's contents into a .csv file
def convert_JSON_to_CSV(json_file_path, json_file_directory):
    df = pd.read_json(json_file_path)
    df.to_csv (json_file_directory, index=None)



# not done: importing this module into MainV5, and calling it correctly

# other things to do: index qns in string form, then converting all of them for one exam paper into a python dictionary
# then converting this python dictionary into the csv file as above