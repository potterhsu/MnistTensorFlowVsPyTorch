import torch.utils.data
from torch.autograd import Variable
from dataset import Dataset


class Evaluator(object):
    def __init__(self, path_to_lmdb_dir):
        self._loader = torch.utils.data.DataLoader(Dataset(path_to_lmdb_dir), batch_size=128, shuffle=False)

    def evaluate(self, model):
        model.eval()
        num_correct = 0

        for batch_idx, (images, labels) in enumerate(self._loader):
            images, labels = Variable(images.cuda(), volatile=True), Variable(labels.cuda())
            logits = model(images)
            predictions = logits.data.max(1)[1]
            num_correct += predictions.eq(labels.data).cpu().sum()

        accuracy = num_correct / float(len(self._loader.dataset))
        return accuracy
