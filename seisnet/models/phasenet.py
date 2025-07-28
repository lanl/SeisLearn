import torch
import torch.nn as nn
import torch.nn.functional as F

class PhaseNetBase(nn.Module):
    def __init__(
            self,
            in_channels=3,
            classes=3,
            filter_factor: int = 1,
            **kwargs,):
        
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.classes = classes
        self.filter_factor = filter_factor
        self.depth = 5
        self.kernel_size = 7
        self.stride = 4
        self.filters_root = 8
        self.activation = torch.relu

        self.inc = nn.Conv1d(self.in_channels,
            self.filters_root * filter_factor,
            self.kernel_size, padding="same",
        )
        self.in_bn = nn.BatchNorm1d(self.filters_root * filter_factor, eps=1e-3)

        self.down_branch = nn.ModuleList()
        self.up_branch = nn.ModuleList()

        last_filters = self.filters_root * filter_factor

        # Compressive Encoder
        for i in range(self.depth):
            filters = int(2**i * self.filters_root) * filter_factor
            conv_same = nn.Conv1d(
                last_filters, filters, self.kernel_size, padding="same", bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            if i == self.depth - 1:
                conv_down = None
                bn2 = None
            else:
                if i in [1, 2, 3]:
                    padding = 0  # Pad manually
                else:
                    padding = self.kernel_size // 2
                conv_down = nn.Conv1d(
                    filters,
                    filters,
                    self.kernel_size,
                    self.stride,
                    padding=padding,
                    bias=False,
                )
                bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.down_branch.append(nn.ModuleList([conv_same, bn1, conv_down, bn2]))

        # Expansive decoder
        for i in range(self.depth - 1):

            filters = int(2 ** (3 - i) * self.filters_root) * filter_factor
            conv_up = nn.ConvTranspose1d(
                last_filters, filters, self.kernel_size, 
                self.stride, bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            conv_same = nn.Conv1d(
                2 * filters, filters, self.kernel_size, padding="same", bias=False
            )
            bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.up_branch.append(nn.ModuleList([conv_up, bn1, conv_same, bn2]))

        
        if self.classes == 3:
            self.out = nn.Conv1d(last_filters, self.classes, 1, padding="same")
            self.output_head = torch.nn.Softmax(dim=1)
        elif self.classes == 1:
            self.out = nn.Conv1d(last_filters, self.classes, kernel_size=1, stride=1, padding="same")
            self.output_head = torch.nn.Sigmoid()


    def forward(self, x, logits=False):
        x = self.activation(self.in_bn(self.inc(x)))

        skips = []
        for i, (conv_same, bn1, conv_down, bn2) in enumerate(self.down_branch):
            x = self.activation(bn1(conv_same(x)))

            if conv_down is not None:
                skips.append(x)
                if i == 1:
                    x = F.pad(x, (2, 3), "constant", 0)
                elif i == 2:
                    x = F.pad(x, (1, 3), "constant", 0)
                elif i == 3:
                    x = F.pad(x, (2, 3), "constant", 0)

                x = self.activation(bn2(conv_down(x)))

        for i, ((conv_up, bn1, conv_same, bn2), skip) in enumerate(
            zip(self.up_branch, skips[::-1])
        ):
            x = self.activation(bn1(conv_up(x)))
            x = x[:, :, 1:-2]

            x = self._merge_skip(skip, x)
            x = self.activation(bn2(conv_same(x)))

        x = self.out(x)
        if logits:
            return x
        else:
            return self.output_head(x)
    
    # @staticmethod
    def _merge_skip(self, skip, x):
        offset = (x.shape[-1] - skip.shape[-1]) // 2
        x_resize = x[:, :, offset : offset + skip.shape[-1]]

        return torch.cat([skip, x_resize], dim=1)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def batch_predict(self,sample:torch.Tensor):
        """
        Perform inference on a batch of waveforms
        NOTE: waveforms should be normalized prior to prediction

        Example:
            model = PhaseNetBase()
            batch = next(iter(dataloader))
            pred = model.batch_predict(batch["X"])
        """
        self.eval()  # close the model for evaluation

        with torch.no_grad():
            pred = self(sample.to(self.device)) 
            pred = pred.detach().cpu().numpy()
        
        return pred
    
    def single_predict(self,sample):
        """
        Perform inference on a single waveform
        NOTE: waveforms should be normalized prior to prediction

        Example:
            from seisnet.dataloaders import Normalize
            norm = Normalize()
            
            model = PhaseNetBase()
            wvfm = np.load(<filename>)
            wvfm = norm(wvfm)
            pred = model.predict(wvfm["X"])
        """
        self.eval()  # close the model for evaluation

        with torch.no_grad():
            # Add a fake batch dimension
            pred = self(torch.tensor(sample, device=self.device).unsqueeze(0)) 
            pred = pred[0].detach().cpu().numpy()
        
        return pred

    def load_checkpoint(self, saved_path:str):
        """
        Example:
            model = PhaseNetBase()
            model.load_checkpoint(</path/to/saved/checkpoint>)
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
            ckpt = torch.load(saved_path, weights_only=True)
            self.load_state_dict(ckpt["model_state_dict"])
        else:
            device = torch.device('cpu')
            ckpt = torch.load(saved_path, map_location=device)#, weights_only=True
            state_dict = ckpt["model_state_dict"]
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[new_key] = v
            self.load_state_dict(new_state_dict)
        self.to(device)
        



# Logic for freezing layers
# def build_model(pretrained=True, fine_tune=True, num_classes=1):
#     """
#     Function to build the neural network model. Returns the final model.
#     Parameters
#     :param pretrained (bool): Whether to load the pre-trained weights or not.
#     :param fine_tune (bool): Whether to train the hidden layers or not.
#     :param num_classes (int): Number of classes in the dataset. 
#     """
#     if pretrained:
#         print('[INFO]: Loading pre-trained weights')
#     elif not pretrained:
#         print('[INFO]: Not loading pre-trained weights')
#     model = models.resnet18(pretrained=pretrained)
#     if fine_tune:
#         print('[INFO]: Fine-tuning all layers...')
#         for params in model.parameters():
#             params.requires_grad = True
#     elif not fine_tune:
#         print('[INFO]: Freezing hidden layers...')
#         for params in model.parameters():
#             params.requires_grad = False
            
#     # change the final classification head, it is trainable
#     model.fc = nn.Linear(512, num_classes)
#     return model