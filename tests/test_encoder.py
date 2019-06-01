import torch
import attention.encoder as encoder


def test_encoders():
    batch_size = 4
    H = 32
    W = 32

    img1 = torch.rand(batch_size, 1, H, W)
    img2 = torch.rand(batch_size, 3, H, W)

    encoders = [
        encoder.VGGEncoder(),
        encoder.ResNetEncoder(),
        encoder.OnmtEncoder(num_layers=2, bidirectional=True,
                            rnn_size=512, dropout=0.5),
        encoder.DarknetEncoder()
    ]

    for enc in encoders:
        assert enc(img1).size() == (batch_size, enc.out_dim,
                                    H // enc.factor, W // enc.factor)
        assert enc(img2).size() == (batch_size, enc.out_dim,
                                    H // enc.factor, W // enc.factor)
