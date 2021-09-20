import Classification_model
import os

a = 'MLP' if Classification_model.xif == '1' else 'CNN'
print('Classification using',a,'\n')

for d in os.listdir("./test"):
    print('Output of the file ' + d + ' is:')
    Classification_model.print_prediction(d)