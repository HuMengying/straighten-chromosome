import os
import numpy as np
#count files and folder
def count_folders_files(rel_path):
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    abs_file_path = os.path.join(script_dir, rel_path)
    files = folders = 0
    for _, dirnames, filenames in os.walk(abs_file_path):
        files += len(filenames)
        folders += len(dirnames)
    return folders,files

#build rel_path
#rel_path->ejem:/single_clase/9700TEST.6.tiff/clase1/clase1_0_9700TEST.6.tiff
#retorna los path a los archivos de la clase que se la pase en clase_folder
def build_rel_path(content_folder,name_example_folder,clase_folder):
    rel_path=content_folder+"/"+name_example_folder+"/"+clase_folder
    num_files_folder=count_folders_files(rel_path)[1]
    rel_paths_img_clase=np.empty(num_files_folder, dtype='object')
    for i in range(0,len(rel_paths_img_clase)):
        rel_paths_img_clase[i]=rel_path+"/"+clase_folder+"_"+str(i)+"_"+name_example_folder
    return rel_paths_img_clase

#los ALL PATH del caso de estudio (path a las imagenes separadas por clase)
def load_all_PATH(content_folder,name_example_folder):
    path=content_folder+"/"+name_example_folder
    cantidad_de_clases=count_folders_files(path)[0]
    string_paths_img=np.empty(cantidad_de_clases, dtype='object')
    for j in range(0,cantidad_de_clases):
        clase_folder="clase"+str(j+1)
        rel_paths_img_clase=build_rel_path(content_folder,name_example_folder,clase_folder)
        string_paths_img[j]=rel_paths_img_clase
    return string_paths_img
