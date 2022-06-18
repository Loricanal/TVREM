import os
import convertapi
import os.path
import sys

try:    
    input_folder = sys.argv[1]
except:
    raise("You have to specify the input folder")
try:
    output_folder = sys.argv[2]
except:
    raise("You have to specify the output folder")

try:
	os.makedirs(output_folder)
except:
	pass

convertapi.api_secret = 'ohOuXuZve1py8vuY'

output_files = [file for file in os.listdir(output_folder)]

todo_files = [os.path.join(input_folder,file) for file in os.listdir(input_folder) if "._" != file[:2] and file.endswith(".pdf") and file.replace('.pdf','.txt') not in output_files]

for file in todo_files:
    print("Computing file",file)
    result = convertapi.convert('txt', { 'File': file })
    result.file.save(os.path.join(output_folder,file.split("/")[-1].replace('.pdf','.txt')))