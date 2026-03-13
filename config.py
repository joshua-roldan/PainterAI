import configparser
import os

def create_config(ii, folder_path):

    config = configparser.ConfigParser()
 
    # Add sections and key-value pairs
    config['General'] = {'last game': 200}
    config['Agent']   = {'learning rate': 0.01,
                         'gamma': 0.1,
                         'state': 'painting and inner frame' # options: 'entire graph', 'painting and inner frame', 'inner frame'
                         }
    config['Model']   = {'hidden size': 64
                         }
    config['Game' ]   = {'number of colors': 5,
                         'grid dimension 1': 20, 
                         'grid dimension 2': 20
                         }
     # Write the configuration to a file
    if not os.path.exists(f"{folder_path}/experiments/{ii}"): os.makedirs(f"{folder_path}/experiments/{ii}")
    with open(f"{folder_path}/experiments/{ii}/config{ii}.ini", 'w') as configfile:
         config.write(configfile)

 
if __name__ == "__main__":
    folder_path = f"/Users/joshuaroldan/painterAI"
    ii = 0
    create_config(ii, folder_path)
