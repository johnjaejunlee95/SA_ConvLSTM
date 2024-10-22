import torch 
import torch.nn as nn 

class SA_Memory_Module(nn.Module): #SAM 
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
  
        self.layer_qh = nn.Conv2d(input_dim, hidden_dim ,1)
        self.layer_kh = nn.Conv2d(input_dim, hidden_dim,1)
        self.layer_vh = nn.Conv2d(input_dim, hidden_dim, 1)
        
        self.layer_km = nn.Conv2d(input_dim, hidden_dim,1)
        self.layer_vm = nn.Conv2d(input_dim, hidden_dim, 1)
        
        self.layer_z = nn.Conv2d(input_dim * 2, input_dim * 2, 1)
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, 1)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
    def forward(self, h, m):
        batch_size, channel, H, W = h.shape

        K_h = self.layer_kh(h)
        Q_h = self.layer_qh(h)
        V_h = self.layer_vh(h)
        
        K_h = K_h.view(batch_size, self.hidden_dim, H*W)
        Q_h = Q_h.view(batch_size, self.hidden_dim, H*W).transpose(1,2)
        V_h = V_h.view(batch_size, self.hidden_dim, H*W)
        
        A_h = torch.softmax(torch.bmm(Q_h, K_h), dim = -1) #batch_size, H*W, H*W
        Z_h = torch.matmul(A_h, V_h.permute(0,2,1)) 

        K_m = self.layer_km(m)
        V_m = self.layer_vm(m)
        
        K_m = K_m.view(batch_size, self.hidden_dim, H*W)
        V_m = V_m.view(batch_size, self.hidden_dim, H*W)
        A_m = torch.softmax(torch.bmm(Q_h, K_m), dim = -1)
        Z_m = torch.matmul(A_m, V_m.permute(0,2,1))
        
        Z_h = Z_h.transpose(1,2).view(batch_size, self.input_dim, H, W)
        Z_m = Z_m.transpose(1,2).view(batch_size, self.input_dim, H, W)

        W_z = torch.cat([Z_h , Z_m], dim = 1)
        Z = self.layer_z(W_z)
        
        ## Memory Updating
        combined = self.layer_m(torch.cat([Z, h], dim = 1))
        mo, mg, mi = torch.chunk(combined, chunks=3, dim = 1)
        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_h = torch.sigmoid(mo) * new_m 

        return new_h, new_m 


class SA_Convlstm_cell(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super().__init__()
        #hyperparrams 
        self.input_channels = input_dim
        self.hidden_dim = hid_dim
        self.kernel_size= 3
        self.padding = 1
        self.attention_layer = SA_Memory_Module(hid_dim, hid_dim)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels = self.input_channels + self.hidden_dim, out_channels = 4 * self.hidden_dim, kernel_size= self.kernel_size, padding = self.padding),
            nn.GroupNorm(4* self.hidden_dim, 4* self.hidden_dim ))    

    def forward(self, x, hidden):
        c, h, m = hidden
        combined = torch.cat([x, h], dim = 1)
        combined_conv = self.conv2d(combined)
        i, f, g, o = torch.chunk(combined_conv, 4, dim =1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = torch.mul(f,c)+ torch.mul(i,g)
        h_next = torch.mul(o, torch.tanh(c_next))
        
        # Self-Attention
        h_next, m_next = self.attention_layer(h_next, m)
        
        return h_next, (c_next, h_next, m_next)
    

class SA_ConvLSTM_Model(nn.Module):  # self-attention convlstm for spatiotemporal prediction model
    def __init__(self, args):
        super(SA_ConvLSTM_Model, self).__init__()
        # hyperparams
        self.batch_size = args.batch_size//args.gpu_num
        self.img_size = (args.img_size, args.img_size)
        self.cells, self.bns = [], []
        self.n_layers = args.num_layers
        self.frame_num = args.frame_num
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        
        
        self.linear_conv = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.input_dim, kernel_size=1, stride=1)
        
        for i in range(self.n_layers):
            input_dim = self.input_dim if i == 0 else self.hidden_dim
            hidden_dim = self.hidden_dim
            self.cells.append(SA_Convlstm_cell(input_dim, hidden_dim))
            self.bns.append(nn.LayerNorm((self.hidden_dim, *self.img_size)))  # Use layernorm

        self.cells = nn.ModuleList(self.cells)
        self.bns = nn.ModuleList(self.bns)
        
    

    def forward(self, X, hidden = None):
        if hidden == None:
            hidden = self.init_hidden(batch_size = self.batch_size, img_size = self.img_size)
        
        predict =[]
        inputs_x = None
        
        # this process is for the hidden state updates
        for t in range(X.size(1)):
            inputs_x =X[:, t, :, :, :]
            for i, layer in enumerate(self.cells):
                inputs_x, hidden[i] = layer(inputs_x, hidden[i])
                inputs_x = self.bns[i](inputs_x)

        inputs_x = X[:, -1, :, :, :]
        for t in range(X.size(1)):
            for i, layer in enumerate(self.cells):
                inputs_x, hidden[i] = layer(inputs_x, hidden[i])
                inputs_x = self.bns[i](inputs_x)
                
            inputs_x = self.linear_conv(inputs_x)
            predict.append(inputs_x)
        
        predict = torch.stack(predict, dim=1)   

        return torch.sigmoid(predict)

    def init_hidden(self, batch_size, img_size):
        h, w = img_size
        hidden_state = (torch.zeros(batch_size, self.hidden_dim, h, w).cuda(),
                        torch.zeros(batch_size, self.hidden_dim, h, w).cuda(),
                        torch.zeros(batch_size, self.hidden_dim, h, w).cuda())
        states = [] 
        for i in range(self.n_layers):
            states.append(hidden_state)
        return states 