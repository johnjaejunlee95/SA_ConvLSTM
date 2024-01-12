from models.conv_lstm import ConvLSTM_Model
from models.sa_conv_lstm import SA_ConvLSTM_Model
import torch 


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

    
def get_model(args):
    if args.model == 'convlstm':
        return ConvLSTM_Model(args)
    elif args.model == 'sa_convlstm':
        return SA_ConvLSTM_Model(args)
    
def load_checkpoint(model, args, path):
    checkpoint = torch.load(path, map_location='cpu')
    parameters = checkpoint['model_state_dict']
    start_epoch = checkpoint['epoch']+1
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    args.lr = checkpoint['lr']
    
    weights = {}
    for name, value in zip(model.state_dict().keys(), parameters.values()):
        weights[name] = value.to('cpu')
    model.load_state_dict(weights)
    
    for state in optimizer_state_dict.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to('cpu')
    
    torch.cuda.empty_cache()
    return start_epoch, args.lr, optimizer_state_dict


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr': optimizer.param_groups[0]['lr'],
    }, path)
    if "best" in path:
        print(f"Saved checkpoint at epoch {epoch}")