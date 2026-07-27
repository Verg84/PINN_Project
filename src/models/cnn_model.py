##############################
# MODEL
##############################
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Simple residual block.
    """

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x

        out = self.block(x)

        out += identity

        out = self.relu(out)

        return out


###########################################################################


class VarCNN(nn.Module):
    """
    CNN for next-frame temperature prediction.

    Input
    -----
    (B,3,H,W)

    Output
    ------
    (B,1,H,W)
    """

    def __init__(
        self,
        in_channels=3,
        hidden_channels=64,
        num_blocks=4,
    ):

        super().__init__()

        ###################################################################
        # Encoder
        ###################################################################

        self.encoder = nn.Sequential(

            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(hidden_channels),

            nn.ReLU(inplace=True),
        )

        ###################################################################
        # Residual backbone
        ###################################################################

        blocks = []

        for _ in range(num_blocks):
            blocks.append(
                ResidualBlock(hidden_channels)
            )

        self.backbone = nn.Sequential(*blocks)

        ###################################################################
        # Decoder
        ###################################################################

        self.decoder = nn.Sequential(

            nn.Conv2d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(hidden_channels // 2),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels // 2,
                1,
                kernel_size=3,
                padding=1,
            ),
        )

        self.initialize_weights()

    #######################################################################

    def initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    #######################################################################

    def forward(self, x):

        """
        x : (B,3,H,W)
        """

        x = self.encoder(x)

        x = self.backbone(x)

        x = self.decoder(x)

        return x

###############################