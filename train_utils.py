import torch
import os, time, pickle
import numpy as np
from functools import wraps
from tqdm import tqdm

def get_dirs(workspace, task, config_yaml):

    checkpoints_dir = os.path.join(
        workspace,
        "checkpoints",
        task,
        time.strftime("%Y%m%d", time.localtime()),
        "config={}".format(os.path.split(config_yaml)[1]),
        )
    os.makedirs(checkpoints_dir, exist_ok=True)

    statistics_dir = os.path.join(
        workspace,
        "statistics",
        task,
        time.strftime("%Y%m%d", time.localtime()),
        "config={}".format(os.path.split(config_yaml)[1]),
        )
    os.makedirs(statistics_dir, exist_ok=True)

    return checkpoints_dir, statistics_dir

def get_lr_lambda(step, warm_up_steps, reduce_lr_steps=None):

    if step <= warm_up_steps:
        return step / warm_up_steps
    else:
        if reduce_lr_steps is not None:
            return 0.9 ** (step // reduce_lr_steps)
        else:
            return 1

def train_loop_decorator(train_loop):
    @wraps(train_loop)
    def new_train_loop(dataloader, model, statistics_dir, epoch, *args, **kwargs):
        print(f"Train for epoch {epoch}\n-------------------------------")
        
        if isinstance(model, dict):
            for model_name in model:
                model[model_name] = model[model_name].train()
        else:
            model = model.train()

        loss_stat = []
        steps = len(dataloader)

        for batch, data in enumerate(tqdm(dataloader)):
            # Inner train loop
            try:
                loss = train_loop(data, model, *args, **kwargs)
                if batch % 1000 == 0:
                    print(f"loss:{loss} [{batch+1:>5d}/{steps:>5d}]")
                    loss_stat.append({'step': steps * (epoch - 1) + batch + 1 , 'loss': loss})
            except RuntimeError as e:
                # print(f"Caught RuntimeError\n{e}\nTrying to ignore.")    
                raise e

        statistics_path = os.path.join(statistics_dir, "losses_statistics.pkl")
        print(f"Saving statistics to {statistics_path}.")
        if epoch == 1:
            pickle.dump(loss_stat, open(statistics_path, 'wb'))
        else:
            loss_stat_to_save = pickle.load(open(statistics_path, 'rb'))
            loss_stat_to_save.extend(loss_stat)
            pickle.dump(loss_stat_to_save, open(statistics_path, 'wb'))

    return new_train_loop

def test_loop_decorator(test_loop):
    @wraps(test_loop)
    def new_test_loop(dataloader, model, dataloader_name, error_names, statistics_dir, epoch, val=True, *args, **kwargs):
        if val:
            print(f"Valditating for epoch {epoch}\n-------------------------------")
        else:
            print("Testing\n-------------------------------")

        if isinstance(model, dict):
            for model_name in model:
                model[model_name] = model[model_name].eval()
        else:
            model = model.eval()
        
        errors = []
        
        for data in tqdm(dataloader):
            # Inner test loop
            error = test_loop(data, model, dataloader_name, *args, **kwargs)
            errors.append(error)
        
        errors = np.array(errors)
        errors = np.mean(errors, axis=0)
        
        if val:
            # Save the validation statistics.
            statistics_dict = {
            "Dataset": dataloader_name,
            "Epoch": epoch,
            "error names": error_names,
            "errors": errors,
            }
        
            print(statistics_dict)
    
            statistics_path = os.path.join(statistics_dir, f"{dataloader_name}_statistics.pkl")
            print(f"Saving statistics to {statistics_path}.")
            if epoch == 0:
                pickle.dump([statistics_dict], open(statistics_path, 'wb'))
            else:
                statistics = pickle.load(open(statistics_path, 'rb'))
                statistics.append(statistics_dict)
                pickle.dump(statistics, open(statistics_path, 'wb'))
        else:
            statistics_dict = {
            "Dataset": dataloader_name,
            "error names": error_names,
            "errors": errors,
            }

        return statistics_dict

    return new_test_loop

def save_checkpoints(model, checkpoints_dir, epoch):
    checkpoint_path = os.path.join(checkpoints_dir, f"epoch={epoch}.pth")
    print(f"Saving checkpoints for epoch {epoch} to {checkpoint_path}.")
    if torch.cuda.device_count() > 1:
        # torch.save(model.module.state_dict(), checkpoint_path)
        torch.save(model.state_dict(), checkpoint_path)
    else:
        torch.save(model.state_dict(), checkpoint_path)

def load_checkpoints(model, checkpoints_dir, epoch):
    checkpoint_path = os.path.join(checkpoints_dir, f"epoch={epoch}.pth")
    print(f"Loading checkpoints for epoch {epoch} from {checkpoint_path}.")
    if torch.cuda.device_count() > 1:
        # torch.save(model.module.state_dict(), checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        model.load_state_dict(torch.load(checkpoint_path))
 
class CrossEntropyLossWithProb:
    
    def __init__(self, weight=None, ignore_index=None):
        self.weight = weight
        self.ignore_index = ignore_index
    
    def to(self, *args, **kwargs):
        if self.weight is not None:
            self.weight = self.weight.to(*args, **kwargs)
        return self
    
    def __call__(self, inputs, targets):
        # inputs: (N, C, ...)
        # targets: (N, C, ...)
        inputs = torch.log_softmax(inputs, dim=1)
        loss = inputs * targets
        if self.weight is not None:
            loss = (self.weight * loss.transpose(1, -1)).transpose(1, -1)
        loss = - torch.sum(loss, dim=1)
        return torch.mean(loss)

class EarlyStopping:
    def __init__(self, monitor=None, mode='min', patience=1):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self._value = -np.inf if mode == 'max' else np.inf
        self._times = 0

    def reset(self):
        self._value = -np.inf if self.mode == 'max' else np.inf
        self._times = 0

    def __call__(self, metrics):
        if isinstance(metrics, dict):
            metrics = metrics[self.monitor]
        if (self.mode == 'min' and metrics <= self._value) or (self.mode == 'max' and metrics >= self._value):
            self._value = metrics
            self._times = 0
        else:
            self._times += 1
        if self._times >= self.patience:
            return True
        else:
            return False

def set_weights(loss_functions, configs, device, pitch_classes=129):

    if 'rest_weight' in configs:
        rest_weight = configs['rest_weight']
        weight = torch.tensor([rest_weight] * pitch_classes, dtype=torch.float32, device=device)
        weight[128] = 1.
        loss_functions['pitch'].weight = weight

    if 'onset_weight' in configs:
        onset_weight = configs['onset_weight']
        loss_functions['onset'].pos_weight = torch.tensor(onset_weight, dtype=torch.float32, device=device)

    return loss_functions
