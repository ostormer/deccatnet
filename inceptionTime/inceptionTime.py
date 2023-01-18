import torch
import torch.nn as nn


# Shamelessly copied from https://github.com/TheMrGhostman/InceptionTime-Pytorch
# with only minimal changes

class InceptionTimeModule(nn.Module):

    def __init__(
        self,
        in_channels,  # Number of input channels (input features)
        n_filters,  # Number of filters per convolution layer => out_channels = 4*n_filters
        kernel_sizes=[9, 19, 39],  # List of kernel sizes for each convolution.
        # Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
        # This is nessesery because of padding size.
        # For correction of kernel_sizes use function "correct_sizes".
        bottleneck_channels=32,  # Number of output channels in bottleneck.
        # Bottleneck wont be used if nuber of in_channels is equal to 1.
        # Activation function for output tensor (nn.ReLU()).
        activation=nn.ReLU(),
        # Indices are needed only if we want to create decoder with InceptionTranspose with MaxUnpool1d.
        return_indices=False
    ):
        super(InceptionTimeModule, self).__init__()
        # Initilalize parameters for the model
        self.return_indices = return_indices

        # Define layers
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                bias=False
            )
        else:
            self.bottleneck = self.pass_through
            bottleneck_channels = 1

        # Define the three parallel Conv layers key for inceptionNet
        self.conv_from_bottleneck_0 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0] // 2,
            bias=False,
        )
        self.conv_from_bottleneck_1 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=kernel_sizes[1] // 2,
            bias=False,
        )
        self.conv_from_bottleneck_2 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=kernel_sizes[2] // 2,
            bias=False,
        )

        self.maxpool = nn.MaxPool1d(
            kernel_size=3,
            stride=1,
            padding=1,
            return_indices=return_indices,
        )
        self.conv_from_maxpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(num_features=4*n_filters)
        self.activation = activation

    def forward(self, X):
        Z_bottleneck = self.bottleneck(X)
        if self.return_indices:
            Z_maxpool, indices = self.maxpool(X)
        else:
            Z_maxpool = self.maxpool(X)

        # Parallell conv layers:
        Z0 = self.conv_from_bottleneck_0(Z_bottleneck)
        Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
        Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
        Z3 = self.conv_from_maxpool(Z_maxpool)

        # Concat, BN and activation
        Z = torch.cat((Z0, Z1, Z2, Z3), dim=1)
        Z = self.activation(self.batch_norm(Z))

        # Return
        if self.return_indices:
            return Z, indices
        else:
            return Z

    @staticmethod
    def pass_through(X):
        return X


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters=32,
        kernel_sizes=[9, 19, 39],
        bottleneck_channels=32,
        use_residual=True,
        activation=nn.ReLU(),
        return_indices=False,
    ):
        super(InceptionBlock, self).__init__()
        self.use_residual = use_residual
        self.activation = activation
        self.return_indices = return_indices

        self.inception_0 = InceptionTimeModule(  # First inception block
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_1 = InceptionTimeModule(  # next inception block, 4x channels
            in_channels=4*n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_2 = InceptionTimeModule(  # Last inception block, identical
            in_channels=4*n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )

        if self.use_residual:  # Define residual block if used.
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=4*n_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm1d(
                    num_features=4*n_filters
                )
            )

    def forward(self, X):
        if self.return_indices:
            Z, i0 = self.inception_0(X)
            Z, i1 = self.inception_1(Z)
            Z, i2 = self.inception_2(Z)
        else:
            Z = self.inception_0(X)
            Z = self.inception_1(Z)
            Z = self.inception_2(Z)

        # If we use a residual shortcut connection, add it here.
        # Shortcut from input X, add result to output of
        # the three inception blocks
        if self.use_residual:
            Z = Z + self.residual(X)
            Z = self.activation(Z)
        # Return output
        if self.return_indices:
            return Z, [i0, i1, i2]
        else:
            return Z


# Inception module for transposed variant, not tested or used.
class InceptionTimeTransposeModule(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels (input features)
        out_channels,
        kernel_sizes=[9, 19, 39],  # List of kernel sizes for each convolution.
        # Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
        # This is nessesery because of padding size.
        # For correction of kernel_sizes use function "correct_sizes".
        bottleneck_channels=32,  # Number of output channels in bottleneck.
        # Bottleneck wont be used if nuber of in_channels is equal to 1.
        # Activation function for output tensor (nn.ReLU()).
        activation=nn.ReLU()
    ):

        super(InceptionTimeTransposeModule, self).__init__()
        self.activation = activation
        self.conv_to_bottleneck_0 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0] // 2,
            bias=False
        )
        self.conv_to_bottleneck_1 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=kernel_sizes[1] // 2,
            bias=False
        )
        self.conv_to_bottleneck_2 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=kernel_sizes[2] // 2,
            bias=False
        )
        self.conv_to_maxpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.max_unpool = nn.MaxUnpool1d(kernel_size=3, stride=1, padding=1)
        self.bottleneck = nn.Conv1d(
            in_channels=3*bottleneck_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, X, indices):
        Z0 = self.conv_to_bottleneck_0(X)
        Z1 = self.conv_to_bottleneck_1(X)
        Z2 = self.conv_to_bottleneck_2(X)
        Z3 = self.conv_to_maxpool(X)

        Z = torch.cat([Z0, Z1, Z2], axis=1)
        BN = self.bottleneck(Z)
        MUP = self.max_unpool(Z3, indices)
        # another possibility insted of sum BN and MUP is
        # adding 2nd bottleneck transposed convolution

        return self.activation(self.batch_norm(BN + MUP))


# Inception block for transposed variant, no tested or used.
class InceptionTimeTransposeBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=32,
        kernel_sizes=[9, 19, 39],
        bottleneck_channels=32,
        use_residual=True,
        activation=nn.ReLU()
    ):
        super(InceptionTimeTransposeBlock, self).__init__()
        self.use_residual = use_residual
        self.activation = activation
        self.inception_1 = InceptionTimeTransposeModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation
        )
        self.inception_2 = InceptionTimeTransposeModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation
        )
        self.inception_3 = InceptionTimeTransposeModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation
        )
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm1d(
                    num_features=out_channels
                )
            )

    def forward(self, X, indices):
        assert len(indices) == 3
        Z = self.inception_1(X, indices[2])
        Z = self.inception_2(Z, indices[1])
        Z = self.inception_3(Z, indices[0])
        if self.use_residual:
            Z = Z + self.residual(X)
            Z = self.activation(Z)
        return Z


# Flatten module
class Flatten(nn.Module):
    def __init__(self, out_features):
        super(Flatten, self).__init__()
        self.output_dim = out_features

    def forward(self, x):
        return x.view(-1, self.output_dim)


# Reshape module
class Reshape(nn.Module):
    def __init__(self, out_shape):
        super(Reshape, self).__init__()
        self.out_shape = out_shape

    def forward(self, x):
        return x.view(-1, *self.out_shape)


class InceptionTime(nn.Module):
    def __init__(self, n_classes):
        super(InceptionTime, self).__init__()
        self.model = nn.Sequential(
            Reshape(out_shape=(1, 160)),
            InceptionBlock(
                in_channels=1,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=32*4,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=32*4*1),
            nn.Linear(in_features=4*32*1, out_features=n_classes)
        )

    def forward(self, X):
        # I believe this should be enough, hope I understand pytorch correctly
        return self.model(X)


if __name__ == "__main__":

    model = InceptionTime(n_classes=4)
    print(model)
