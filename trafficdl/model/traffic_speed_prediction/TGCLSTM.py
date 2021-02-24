from trafficdl.model.abstract_model import AbstractModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from trafficdl.model import loss
import math


class FilterLinear(nn.Module):
    def __init__(self, device, input_dim, output_dim, in_features, out_features, filter_square_matrix, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features

        self.num_nodes = filter_square_matrix.shape[0]
        self.filter_square_matrix = Variable(filter_square_matrix.repeat(output_dim, input_dim).to(device),
                                             requires_grad=False)

        self.weight = Parameter(torch.Tensor(out_features, in_features).to(device))  # [out_features, in_features]
        if bias:
            self.bias = Parameter(torch.Tensor(out_features).to(device))  # [out_features]
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')'


class TGCLSTM(AbstractModel):
    def __init__(self, config, data_feature):
        '''
        Args:
            K: K-hop graph
            A: adjacency matrix
            FFR: free-flow reachability matrix
            Clamp_A: Boolean value, clamping all elements of A between 0. to 1.
        '''
        super(TGCLSTM, self).__init__(config, data_feature)
        self.data_feature = data_feature
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.input_dim = data_feature['feature_dim']
        self.in_features = self.input_dim * self.num_nodes
        self.output_dim = config.get('output_dim', 1)
        self.out_features = self.output_dim * self.num_nodes
        self.K = config.get('K_hop_numbers', 3)
        self.dataset_class = config.get('dataset_class', 'TrafficSpeedDataset')
        self.scaler_type = config.get('scaler', 'standard')
        self.device = config.get('device', torch.device('cpu'))
        self._scaler = self.data_feature.get('scaler')

        self.A_list = []  # Adjacency Matrix List
        adj_mx = data_feature['adj_mx']
        adj_mx[adj_mx > 1e-4] = 1
        adj_mx[adj_mx <= 1e-4] = 0

        A = torch.FloatTensor(adj_mx).to(self.device)
        A_temp = torch.eye(self.num_nodes, self.num_nodes, device=self.device)
        for i in range(self.K):
            A_temp = torch.matmul(A_temp, A)
            if config.get('Clamp_A', True):
                # confine elements of A
                A_temp = torch.clamp(A_temp, max=1.)
            if self.dataset_class == "TGCLSTMDataset":
                self.A_list.append(
                    torch.mul(A_temp, torch.Tensor(data_feature['FFR'][config.get('back_length', 3)]).to(self.device)))
            else:
                self.A_list.append(A_temp)

        # a length adjustable Module List for hosting all graph convolutions
        self.gc_list = nn.ModuleList([FilterLinear(self.device, self.input_dim, self.output_dim, self.in_features,
                                                   self.out_features, self.A_list[i], bias=False) for i in
                                      range(self.K)])

        hidden_size = self.out_features
        input_size = self.out_features * self.K

        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)

        # initialize the neighbor weight for the cell state
        self.Neighbor_weight = Parameter(torch.FloatTensor(self.out_features, self.out_features).to(self.device))
        stdv = 1. / math.sqrt(self.out_features)
        self.Neighbor_weight.data.uniform_(-stdv, stdv)

    def step(self, input, Hidden_State, Cell_State):
        x = input  # [batch_size, in_features]

        gc = self.gc_list[0](x)  # [batch_size, out_features]
        for i in range(1, self.K):
            gc = torch.cat((gc, self.gc_list[i](x)), 1)  # [batch_size, out_features * K]

        combined = torch.cat((gc, Hidden_State), 1)  # [batch_size, out_features * (K+1)]
        # fl: nn.linear(out_features * (K+1), out_features)
        f = torch.sigmoid(self.fl(combined))
        i = torch.sigmoid(self.il(combined))
        o = torch.sigmoid(self.ol(combined))
        C = torch.tanh(self.Cl(combined))

        NC = torch.matmul(Cell_State, torch.mul(
            Variable(self.A_list[-1].repeat(self.output_dim, self.output_dim), requires_grad=False).to(self.device),
            self.Neighbor_weight))

        Cell_State = f * NC + i * C  # [batch_size, out_features]
        Hidden_State = o * torch.tanh(Cell_State)  # [batch_size, out_features]

        return Hidden_State, Cell_State, gc

    def Bi_torch(self, a):
        a[a < 0] = 0
        a[a > 0] = 1
        return a

    def forward(self, batch):
        inputs = batch['X']  # [batch_size,  input_window, num_nodes, input_dim]
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)  # [batch_size, out_features]

        outputs = None

        for i in range(time_step):
            input = torch.squeeze(torch.transpose(inputs[:, i:i + 1, :, :], 2, 3)).reshape(batch_size, -1)
            Hidden_State, Cell_State, gc = self.step(input, Hidden_State, Cell_State)
            # gc: [batch_size, out_features * K]
            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)  # [batch_size, 1, out_features]
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)  # [batch_size, input_window, out_features]
        output = torch.transpose(torch.squeeze(outputs[:, -1, :]).reshape(batch_size, self.output_dim, self.num_nodes),
                                 1, 2).unsqueeze(1)
        return output  # [batch_size, 1, num_nodes, out_dim]

    def get_data_feature(self):
        return self.data_feature

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        x = batch['X']
        y = batch['y']
        output_length = y.shape[1]
        y_preds = []
        x_ = x.clone()
        for i in range(output_length):
            batch_tmp = {'X': x_}
            y_ = self.forward(batch_tmp)
            y_preds.append(y_.clone())
            if y_.shape[3] < x_.shape[3]:
                y_ = torch.cat([y_, y[:, i:i + 1, :, self.output_dim:]], dim=3)
            x_ = torch.cat([x_[:, 1:, :, :], y_], dim=1)
        y_preds = torch.cat(y_preds, dim=1)  # [batch_size, output_window, batch_size, output_dim]
        return y_preds

    def initHidden(self, batch_size):
        Hidden_State = Variable(torch.zeros(batch_size, self.out_features).to(self.device))
        Cell_State = Variable(torch.zeros(batch_size, self.out_features).to(self.device))
        return Hidden_State, Cell_State
