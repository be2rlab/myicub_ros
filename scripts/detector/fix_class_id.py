
def fix_class_id(train_files, class_names, added_classes):
	for train_file in train_files:
		with open(train_file, 'r') as f:
			train_lines = f.readlines()

		for line in train_lines:

			txt_file = line.replace('images', 'labels').replace('png', 'txt')
			with open(txt_file[:-1], 'r') as f:
				label = f.readline()

			assert len(label.split()) == 5
			# print(line, train_file.split('/')[-1][:-4])
			class_name = train_file.split('/')[-1][:-4]
			class_idx = class_names.index(class_name) + added_classes

			# print(class_idx)
			# print(label)
			# print(str(class_idx) + label[1:])

			with open(txt_file[:-1], 'w') as f:
				f.write(str(class_idx) + label[1:])

if __name__ == '__main__':
	train_files = ['dataset_files/Banana.txt',
		                    'dataset_files/Cube.txt',
		                    'dataset_files/Box.txt',
		                    'dataset_files/Toy.txt',
		                    ]

	added_classes = 0

	class_names = ['Cube', 'Banana', 'Box', 'Toy']
	fix_class_id(train_files, class_names, added_classes)	