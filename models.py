import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = nn.Conv2d(1, 4, 3, padding=1)
        self.fc1 = nn.Linear(28*28*4, 27)
        # self.fc1 = nn.Conv2d(4, 27, 5)

    def forward(self, x):
		x = F.relu(self.conv0(x))
		# import pdb;pdb.set_trace()
		x = self.fc1(x.view(x.size(0), 4 * 28 * 28))
		return F.log_softmax(x)
    	# x = F.relu(self.fc0(x))
    	# x = F.relu(self.fc1(x))
       	# return x
    	# x = F.relu(self.fc0(x))
    	# x =F.max_pool2d(F.relu(self.fc0(x)), (2, 2))
        # return x
