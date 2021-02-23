f = open('data/trace_00.csv')
newFile = open('data/trace_01.csv','a')
lines = f.readlines()
for line in lines:
    line_array = line.split("\t")
    line_array = list(map(lambda x: x.rstrip('\n'),line_array))
    new_line = ','.join(line_array)
    newFile.write(new_line+'\n')
print(len(lines))
