import torch
import torch.nn as nn
from torch.autograd import Variable
from convlstmcell import ConvLSTMCell

class LSTMInCell(nn.Module):

    def __init__(self, in_channels, out_channels, batch_size=1):
        super(LSTMInCell, self).__init__()

        out_channels = out_channels // 6

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size

        self.relu = nn.ReLU()

        self._1x1x1_conv_1 = nn.Conv3d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=(1, 1, 1))
        self._1x1x1_conv_2 = nn.Conv3d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=(1, 1, 1))
        self._1x1x1_conv_3 = nn.Conv3d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=(1, 1, 1))

        self._3x3x3_max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3),
                                            stride=(1, 1, 1),
                                            ceil_mode=True)
        self._pad = nn.ConstantPad3d(self.get_padding_shape(filter_shape=(3, 3, 3),
                                                            stride=(1, 1, 1)),
                                     0)

        self._1x1x1_lstm_conv = ConvLSTMCell(input_channels=in_channels,
                                             hidden_channels=out_channels,
                                             kernel_size=1)

        self._3x3x3_lstm_conv = ConvLSTMCell(input_channels=out_channels,
                                             hidden_channels=out_channels * 2,
                                             kernel_size=3)

        self._5x5x5_lstm_conv = ConvLSTMCell(input_channels=out_channels,
                                             hidden_channels=out_channels * 2,
                                             kernel_size=5)

        self._1x1x1_lstm_h = None
        self._1x1x1_lstm_c = None

        self._3x3x3_lstm_h = None
        self._3x3x3_lstm_c = None

        self._5x5x5_lstm_h = None
        self._5x5x5_lstm_c = None

        self._1x1x1_conv_h = None

    def get_padding_shape(self, filter_shape, stride):

        def _pad_top_bottom(filter_dim, stride_val):
            pad_along = max(filter_dim - stride_val, 0)
            pad_top = pad_along // 2
            pad_bottom = pad_along - pad_top
            return pad_top, pad_bottom

        padding_shape = []

        for filter_dim, stride_val in zip(filter_shape, stride):
            pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
            padding_shape.append(pad_top)
            padding_shape.append(pad_bottom)

        depth_top = padding_shape.pop(0)
        depth_bottom = padding_shape.pop(0)
        padding_shape.append(depth_top)
        padding_shape.append(depth_bottom)

        return tuple(padding_shape)

    def forward(self, x, step=0):

        if step == 0:
            self._1x1x1_lstm_h, self._1x1x1_lstm_c = self._1x1x1_lstm_conv.init_hidden(self.batch_size,
                                                                                       self.out_channels,
                                                                                       (*x.shape[2:],))

            self._3x3x3_lstm_h, self._3x3x3_lstm_c = self._3x3x3_lstm_conv.init_hidden(self.batch_size,
                                                                                       self.out_channels * 2,
                                                                                       (*x.shape[2:],))

            self._5x5x5_lstm_h, self._5x5x5_lstm_c = self._5x5x5_lstm_conv.init_hidden(self.batch_size,
                                                                                       self.out_channels * 2,
                                                                                       (*x.shape[2:],))

        self._1x1x1_lstm_h, self._1x1x1_lstm_c = self._1x1x1_lstm_conv(x,
                                                                       self._1x1x1_lstm_h,
                                                                       self._1x1x1_lstm_c)

        self._3x3x3_lstm_h, self._3x3x3_lstm_c = self._3x3x3_lstm_conv(self.relu(self._1x1x1_conv_1(x)),
                                                                       self._3x3x3_lstm_h,
                                                                       self._3x3x3_lstm_c)

        self._5x5x5_lstm_h, self._5x5x5_lstm_c = self._5x5x5_lstm_conv(self.relu(self._1x1x1_conv_2(x)),
                                                                       self._5x5x5_lstm_h,
                                                                       self._5x5x5_lstm_c)

        self._1x1x1_conv_h = self._1x1x1_conv_3(self._3x3x3_max_pool(self._pad(x)))

        return torch.cat((self._1x1x1_lstm_h,
                          self._3x3x3_lstm_h,
                          self._5x5x5_lstm_h,
                          self._1x1x1_conv_h), 1)
