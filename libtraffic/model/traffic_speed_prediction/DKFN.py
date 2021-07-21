import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from libtraffic.model import loss
from libtraffic.model.abstract_traffic_state_model import AbstractTrafficStateModel


class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.cuda(), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    #         print(self.weight.data)
    #         print(self.bias.data)

    def forward(self, input):
        return F.linear(input, self.filter_square_matrix.matmul(self.weight), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')'


class DKFN(AbstractTrafficStateModel):
    # def __init__(self, K, A, feature_size, Clamp_A=True):
    def __init__(self, config, data_feature):
        # GC-LSTM
        '''
        Args:
            K: K-hop graph
            A: adjacency matrix
            feature_size: the dimension of features
            Clamp_A: Boolean value, clamping all elements of A between 0. to 1.
        '''
        super(DKFN, self).__init__(config, data_feature)

        self.device = config.get('device', torch.device('cpu'))
        self._scaler = self.data_feature.get('scaler')

        self.K = config.get('K_hop_numbers', 3)

        # get adjacency matrices
        self.A_list = []
        # binarization
        A = torch.FloatTensor(data_feature['adj_mx']).to(self.device)
        self._eps = 1e-4
        A[A > self._eps] = 1
        A[A <= self._eps] = 0
        # normalization
        D_inverse = torch.diag(1 / torch.sum(A, 0))
        norm_A = torch.matmul(D_inverse, A)
        A = norm_A

        self.feature_size = A.shape[0]
        self.hidden_size = self.feature_size
        self.gc_input_size = self.feature_size * self.K

        # compute its list of powers
        A_temp = torch.eye(self.feature_size, self.feature_size, device=self.device)
        for i in range(self.K):
            A_temp = torch.matmul(A_temp, A)
            if config.get('Clamp_A', True):
                # consider reachability only
                A_temp = torch.clamp(A_temp, max=1.)
            self.A_list.append(A_temp)

        # a length adjustable Module List for hosting all graph convolutions
        self.gc_list = nn.ModuleList([FilterLinear(self.feature_size, self.feature_size, self.A_list[i], bias=False) for i in range(self.K)])

        self.fl = nn.Linear(self.gc_input_size + self.hidden_size, self.hidden_size)
        self.il = nn.Linear(self.gc_input_size + self.hidden_size, self.hidden_size)
        self.ol = nn.Linear(self.gc_input_size + self.hidden_size, self.hidden_size)
        self.Cl = nn.Linear(self.gc_input_size + self.hidden_size, self.hidden_size)

        # initialize the neighbor weight for the cell state
        self.Neighbor_weight = Parameter(torch.FloatTensor(self.feature_size))
        stdv = 1. / math.sqrt(self.feature_size)
        self.Neighbor_weight.data.uniform_(-stdv, stdv)

        # RNN
        self.rnn_input_size = self.feature_size

        self.rfl = nn.Linear(self.rnn_input_size + self.hidden_size, self.hidden_size)
        self.ril = nn.Linear(self.rnn_input_size + self.hidden_size, self.hidden_size)
        self.rol = nn.Linear(self.rnn_input_size + self.hidden_size, self.hidden_size)
        self.rCl = nn.Linear(self.rnn_input_size + self.hidden_size, self.hidden_size)

        # addtional variables
        self.c = torch.nn.Parameter(torch.Tensor([1]))

        self.fc1 = nn.Linear(64, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, 64)
        self.fc5 = nn.Linear(64, self.hidden_size)
        self.fc6 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc7 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc8 = nn.Linear(self.hidden_size, 64)

    def step(self, step_input, Hidden_State, Cell_State, rHidden_State, rCell_State):
        # GC-LSTM
        x = step_input
        gc = self.gc_list[0](x)
        for i in range(1, self.K):
            gc = torch.cat((gc, self.gc_list[i](x)), 1)

        combined = torch.cat((gc, Hidden_State), 1)
        f = torch.sigmoid(self.fl(combined))
        i = torch.sigmoid(self.il(combined))
        o = torch.sigmoid(self.ol(combined))
        C = torch.tanh(self.Cl(combined))

        NC = torch.mul(Cell_State,
                       torch.mv(Variable(self.A_list[-1], requires_grad=False).to(self.device), self.Neighbor_weight))
        Cell_State = f * NC + i * C
        Hidden_State = o * torch.tanh(Cell_State)

        # LSTM
        rcombined = torch.cat((step_input, rHidden_State), 1)
        rf = torch.sigmoid(self.rfl(rcombined))
        ri = torch.sigmoid(self.ril(rcombined))
        ro = torch.sigmoid(self.rol(rcombined))
        rC = torch.tanh(self.rCl(rcombined))
        rCell_State = rf * rCell_State + ri * rC
        rHidden_State = ro * torch.tanh(rCell_State)

        # Kalman Filtering
        var1, var2 = torch.var(step_input), torch.var(gc)

        pred = (Hidden_State * var1 * self.c + rHidden_State * var2) / (var1 + var2 * self.c)

        return Hidden_State, Cell_State, gc, rHidden_State, rCell_State, pred

    def Bi_torch(self, a):
        a[a < 0] = 0
        a[a > 0] = 1
        return a

    def forward(self, batch):
        inputs = batch['X']  # [batch_size, input_window, num_nodes, input_dim]
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State, rHidden_State, rCell_State = self.initHidden(batch_size)

        for i in range(time_step):
            step_input = inputs[:, i:i+1, :, :].transpose(2, 3).squeeze().reshape(batch_size, -1)
            Hidden_State, Cell_State, gc, rHidden_State, rCell_State, pred = self.step(
                torch.squeeze(inputs[:, i:i + 1, :]), Hidden_State, Cell_State, rHidden_State, rCell_State)
        return pred.unsqueeze(1).unsqueeze(3)
        # return Hidden_State, Cell_State, rHidden_State, rCell_State, pred

    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            rHidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            rCell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State, rHidden_State, rCell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            rHidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            rCell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State, rHidden_State, rCell_State

    def reinitHidden(self, batch_size, Hidden_State_data, Cell_State_data):
        return initHidden()

    '''
    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        # pick the first output dimension only as only the first is supported
        y_true = self._scaler.inverse_transform(y_true[..., :1])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :1])
        return loss.masked_mse_torch(y_predicted, y_true)
    '''

    # multi-step prediction
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
                # concatenate with the extra dimensions of original predictions of nodes
                y_ = torch.cat([y_, y[:, i:i+1, :, 1:]], dim=3)
            x_ = torch.cat([x_[:, 1:, :, :], y_], dim=1)
        y_preds = torch.cat(y_preds, dim=1)  # [batch_size, output_window, num_nodes, output_dim]
        return y_preds

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true)

