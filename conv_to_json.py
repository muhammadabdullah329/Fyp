import os

dataset_path = "dataset"

for sub in os.listdir(dataset_path):
	for file in os.listdir(dataset_path+'/'+sub):
		print(sub + file)
		
		video = dataset_path + '/'+ sub + '/' + file
		
		if not os.path.exists('data_out/'+sub + '/' + file):
			os.makedirs('data_out/'+sub + '/' + file)
		
		command = r'bin\OpenPoseDemo.exe --video "{}" --write_json "data_out/{}/{}"'.format(video,sub,file)
		print("executing...", command)
		
		os.system(command)