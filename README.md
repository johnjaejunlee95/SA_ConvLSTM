# SA_ConvLSTM



Original Paper: [Self-Attention ConvLSTM for Spatiotemporal Prediction]( https://doi.org/10.1609/aaai.v34i07.6819)



### Download Datasets

##### training / validation datasets:

```
wget https://huggingface.co/spaces/chrisjay/mnist-adversarial/resolve/603879aac618aca69749a8a9172daec23a9dd2c4/files/MNIST/raw/train-images-idx3-ubyte.gz
```

##### test datasets:

```shell
wget https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
```



### Run

##### example:

```
python main.py --model sa_convlstm --batch_size 4 --lr 1e-3 --epochs 50 --num_layers 4 --hidden_dim 64
```

model selection: 
- **convlstm**: ConvLSTM
- **sa_convlstm**: Self-Attention (Memory) ConvLSTM

##### parsing arguments:

```
('--lr', default=1e-3, type=float, help='learning rate')
('--batch_size', default=1, type=int, help='batch size')
('--epochs', type=int, default=50, help='number of epochs to train')
('--hidden_dim', type=int, default=64, help='number of hidden dim for ConvLSTM layers')
('--input_dim', type=int, default=1, help='input channels')
('--model', type=str, default='convlstm', help='name of the model')
('--num_layers', type=int, default=4, help='number of layers')
('--frame_num', type=int, default=10, help='number of frames')
('--img_size', type=int, default=64, help='image size')
('--reload', action='store_true', help='reload model')
```

