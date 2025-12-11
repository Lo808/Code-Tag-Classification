import pandas as pd
import json
import zipfile


def load_json_data(file_path):
    '''
    Docstring for load_json_data
    
    :param file_path: path of the zip file containing the json data set
    '''
    data=[]
    with zipfile.ZipFile(file_path,'r') as z:
        for fname in z.namelist():
            if fname.endswith(".json"):
                try:     
                    with z.open(fname) as f:
                        file_data=json.load(f)

                        data.append(file_data)
                except Exception as e:

                    print(f"Failed reading {fname} from ZIP: {e}")

    #print(f"Loaded {len(data)} files from ZIP folder")

    data_frame=pd.DataFrame(data)
    usefuls_column=['prob_desc_description','source_code','tags']

    return data_frame[usefuls_column]

