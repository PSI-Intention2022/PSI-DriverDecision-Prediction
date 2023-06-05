import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# import torchvision.transforms as transforms

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class ResLSTMDrivingGlobal(nn.Module):
    def __init__(self, args, model_opts):
        super(ResLSTMDrivingGlobal, self).__init__()
        #
        # self.enc_in_dim = model_opts['enc_in_dim']  # 4, input bbox+convlstm_output context vector
        # self.enc_out_dim = model_opts['enc_out_dim']  # 64
        # self.dec_in_emb_dim = model_opts['dec_in_emb_dim']  # 1 for intent, 1 for speed, ? for rsn
        # self.dec_out_dim = model_opts['dec_out_dim']  # 64 for lstm decoder output
        # self.output_dim = model_opts['output_dim']  # 4 for bbox, 2/3: intention; 62 for reason; 1 for trust score; 4 for trajectory.
        #
        # n_layers = model_opts['n_layers']
        # dropout = model_opts['dropout']
        # predict_length = model_opts['predict_length']

        self.predict_length = args.predict_length
        self.args = args
        # EncoderCNN architecture
        CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
        CNN_embed_dim = 512  # latent dim extracted by 2D CNN
        res_size = 224  # ResNet image size
        dropout_p = 0.0  # dropout probability

        # DecoderRNN architecture
        RNN_hidden_layers = 3
        RNN_hidden_nodes = 512
        RNN_FC_dim = 256

        # Create model
        self.cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p,
                                    CNN_embed_dim=CNN_embed_dim).to(device)
        num_pred_class = 3
        self.rnn_decoder_speed = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                                h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=num_pred_class).to(device)
        self.rnn_decoder_dir = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers,
                                h_RNN=RNN_hidden_nodes, h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=num_pred_class).to(device)


        # if model_opts['output_activation'] == 'tanh':
        #     self.activation = nn.Tanh()
        # elif model_opts['output_activation'] == 'sigmoid':
        #     self.activation = nn.Sigmoid()
        # else:
        #     self.activation = nn.Identity()

        self.module_list = [self.cnn_encoder, self.rnn_decoder_speed, self.rnn_decoder_dir]
        # self.intent_embedding = 'int' in self.args.model_name
        # self.reason_embedding = 'rsn' in self.args.model_name
        # self.speed_embedding = 'speed' in self.args.model_name

    def forward(self, data):
        images = data['image'] # bs x ts x c x h x w


        visual_feats = self.cnn_encoder(images)
        pred_speed = self.rnn_decoder_speed(visual_feats)
        pred_dir = self.rnn_decoder_dir(visual_feats)

        return pred_speed, pred_dir
        #
        # bbox = data['bboxes'][:, :self.args.observe_length, :].type(FloatTensor)
        # # enc_input/dec_input_emb: bs x ts x enc_input_dim/dec_emb_input_dim
        # enc_input = bbox
        #
        # # 1. encoder
        # enc_output, (enc_hc, enc_nc) = self.encoder(enc_input)
        # # because 'batch_first=True'
        # # enc_output: bs x ts x (1*hiden_dim)*enc_hidden_dim --- only take the last output, concatenated with dec_input_emb, as input to decoder
        # # enc_hc:  (n_layer*n_directions) x bs x enc_hidden_dim
        # # enc_nc:  (n_layer*n_directions) x bs x enc_hidden_dim
        # enc_last_output = enc_output[:, -1:, :]  # bs x 1 x hidden_dim
        #
        # # 2. decoder
        # traj_pred_list = []
        # evidence_list = []
        # prev_hidden = enc_hc
        # prev_cell = enc_nc
        #
        # dec_input_emb = None
        # # if self.intent_embedding:
        # #     # shape: (bs,)
        # #     intent_gt_prob = data['intention_prob'][:, self.args.observe_length].type(FloatTensor)
        # #     intent_pred = data['intention_pred'].type(FloatTensor) # bs x 1
        #
        # for t in range(self.predict_length):
        #     if dec_input_emb is None:
        #         dec_input = enc_last_output
        #     else:
        #         dec_input = torch.cat([enc_last_output, dec_input_emb[:, t, :].unsqueeze(1)])
        #
        #     dec_output, (dec_hc, dec_nc) = self.decoder(dec_input, (prev_hidden, prev_cell))
        #     logit = self.fc(dec_output.squeeze(1)) # bs x 4
        #     traj_pred_list.append(logit)
        #     prev_hidden = dec_hc
        #     prev_cell = dec_nc
        #
        # traj_pred = torch.stack(traj_pred_list, dim=0).transpose(1, 0) # ts x bs x 4 --> bs x ts x 4

        # return traj_pred

    def build_optimizer(self, args):
        param_group = []
        learning_rate = args.lr
        # if self.backbone is not None:
        #     for name, param in self.backbone.named_parameters():
        #         if not self.args.freeze_backbone:
        #             param.requres_grad = True
        #             param_group += [{'params': param, 'lr': learning_rate * 0.1}]
        #         else:
        #             param.requres_grad = False


        for module in self.module_list:
            param_group += [{'params': module.parameters(), 'lr': learning_rate}]

        optimizer = torch.optim.Adam(param_group, lr=args.lr, eps=1e-7)

        for param_group in optimizer.param_groups:
            param_group['lr0'] = param_group['lr']

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # self.optimizer = optimizer

        return optimizer, scheduler

    def lr_scheduler(self, optimizer, cur_epoch, args, gamma=10, power=0.75):
        decay = (1 + gamma * cur_epoch / args.epochs) ** (-power)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr0'] * decay
            param_group['weight_decay'] = 1e-3
            param_group['momentum'] = 0.9
            param_group['nesterov'] = True
        return


    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)


# 2D CNN encoder using ResNet-50 pretrained
class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-50 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d): # Input: bs x ts x C x H x W
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)  # flatten output of conv

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=3):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x