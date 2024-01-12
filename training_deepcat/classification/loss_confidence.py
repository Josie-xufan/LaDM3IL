import torch
import torch.nn.functional as F

class partial_loss(torch.nn.Module):
    def __init__(self, confidence, conf_ema_m=0.99):
        super().__init__()
        self.confidence = confidence
        self.init_conf = confidence.detach()
        self.conf_ema_m = conf_ema_m

    def set_conf_ema_m(self, epoch, args):
        start = args.soft_ema_range[0]
        end = args.soft_ema_range[1]
        self.conf_ema_m = 1. * epoch / args.epochs * (end - start) + start

    def forward(self, outputs, patch_index):
        # logsm_outputs = F.log_softmax(outputs, dim=1) #patch_num * num_class
        logsm_outputs = F.log_softmax(torch.concat([1-outputs, outputs], axis=1), dim=1) #patch_num * num_class
        final_outputs = logsm_outputs * self.confidence[patch_index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss
    
    def confidence_update(self, temp_un_conf, patch_index, device):
        with torch.no_grad():
            _, prot_pred = (temp_un_conf ).max(dim=1)
            pseudo_label = F.one_hot(prot_pred, 2).float().detach().to(device)
            self.confidence[patch_index, :] = self.conf_ema_m * self.confidence[patch_index, :] + (1 - self.conf_ema_m) * pseudo_label
        return None