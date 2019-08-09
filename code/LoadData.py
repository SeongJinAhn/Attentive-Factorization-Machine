import torch
from torch.utils.data import TensorDataset, DataLoader

class Data(object):
    def __init__(self,config,process_type):
        self.process_type = process_type
        self.path = config.path + '/ml-tag.' + process_type + '.libfm'
        self.batch_size = config.batch_size
        self.ToTensor = torch.LongTensor
        self.num_user = 0
        if config.GPU_available:
            self.ToTensor = torch.cuda.LongTensor
        self.group,self.label,self.num_user = self.readFile(self.path)
    
    def readFile(self, path):
        groups = []
        labels = []
        num_user = 0
        with open(path, "r") as f:
            while True:
                line = f.readline().strip('\n').split(" ")
                if line == None or line == ['']:
                    break
                group = [float(i[:-2]) for i in line[1:]]
                num_user = max(num_user, max(group))
                groups.append(group)
                labels.append(float(line[0]))
        return groups,labels,int(num_user)

    def dataLoader(self):
        data = TensorDataset(self.ToTensor(self.group), torch.FloatTensor(self.label))
        return DataLoader(data, batch_size=self.batch_size, shuffle=True)

    def user_number(self):
        return self.num_user