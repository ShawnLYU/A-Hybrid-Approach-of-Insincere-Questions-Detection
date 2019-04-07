import csv
from progress.bar import Bar

input_path = '../data/mytest.csv'
output_path = 'mytest_noquots.csv'

input_file = open(input_path, 'r', newline='')
output_file = open(output_path, 'w', newline='')

fieldnames=['qid', 'question_text', 'target']
output_file.write(','.join(fieldnames) + '\n')

line = input_file.readline()
line = input_file.readline()
bar = Bar("Cleaning: ", max=1044839)
while line != '':
	try:
		if line[21] == '\"':
			line = line[:21] + line[22:]
	except IndexError:
		print('wrong line:' + line)
		if line.strip() == '",0':
			line = line.replace('\"', '')
			print('fixed')
	try:
		if line[-4] == '\"':
			line = line[:-4] + line[-3:]
	except IndexError:
		print('wrong line:' + line)
		if line.strip() == '",0':
			line = line.replace('\"', '')
			print('fixed')
	output_file.write(line)
	line = input_file.readline()
	bar.next()
bar.finish()

input_file.close()
output_file.close()