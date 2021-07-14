import os
import json
import pandas as pd

json_folder = 'data_out'

x_train = []
y_train = []
file_name = []

for sub in os.listdir(json_folder):
	for file in sorted(os.listdir(json_folder+'/'+sub)):
		for x in sorted(os.listdir(json_folder+'/'+sub+'/'+file)):
			#print (sub + file + x)
			with open(json_folder+'/'+sub+'/'+file+'/'+x) as f:
				data = json.load(f)
				try:
					pose = data['people'][0]['pose_keypoints_2d']
				except IndexError:
					continue
				del pose[2::3]
				
				final = pose
				x_train.append(pose)
				y_train.append(sub)
				file_name.append(sub + '_'+file+'_'+x.split('_')[0]+'.mp4')
				#print (sub + file + x)
				#print(final)
				#break

df1 = pd.DataFrame(x_train)
df1['label'] = y_train
df1['filename'] = file_name

df1.to_csv('csv_out/data_csv.csv')

print(df1.head(10))

file_names = df1['filename'].value_counts()
labels = df1['label'].value_counts().index
print(labels)

final_csv = pd.DataFrame()

for index, value in file_names.iteritems():
	data = df1[(df1['filename']== index)]
	if len(data)>0:
		skip = int ((len(data))/12)
		print("Skip value", skip)
		
		for i, d in enumerate(data.index):
			if i % skip != 0:
				data.drop(d, inplace=True)
		if len(data) >12:
			diff = 12 - len(data)
			data = data.iloc[:diff,:]
		final_csv = final_csv.append(data)
		print("length:", len (data))
print(final_csv)		

final_csv.to_csv('csv_out/finaldata_csv.csv')