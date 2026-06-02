import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_ch, out_ch, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class MultiResFusion(nn.Module):
    def __init__(self, in_channels=32, debug=False):
        super(MultiResFusion, self).__init__()
        self.debug = debug  # True로 바꾸면 shape 출력
        C = in_channels      # 32
        C2 = C * 2           # 64

        # High_feature를 Low 해상도로 줄인 뒤 정리하는 conv (해상도 변경은 interpolate가 담당)
        self.high_proj = conv_bn_relu(C, C)

        # U-Net encoder
        self.enc = conv_bn_relu(C2, C2)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = conv_bn_relu(C2, C2)

        # U-Net decoder
        self.up = nn.ConvTranspose2d(C2, C, kernel_size=2, stride=2)
        self.dec = conv_bn_relu(C + C2, C)

        # 최종 skip: U-Net 출력 + 원본 High_feature
        self.final = conv_bn_relu(C + C, C)

    def _log(self, name, tensor):
        if self.debug:
            print(f"[MultiResFusion] {name}: {tuple(tensor.shape)}")

    def forward(self, High_feature, Low_feature):
        self._log("High_feature(input)", High_feature)
        self._log("Low_feature(input)", Low_feature)

        # High를 Low 해상도로 한 번에 downsample
        high_down = F.interpolate(High_feature, size=Low_feature.shape[2:],
                                  mode='bilinear', align_corners=False)
        high_down = self.high_proj(high_down)
        self._log("high_down (after proj)", high_down)

        # concat
        x = torch.cat([high_down, Low_feature], dim=1)  # [B, 64, 96, 320]
        self._log("concat", x)

        # U-Net encoder
        e = self.enc(x)                          # [B, 64, 96, 320]
        self._log("enc", e)
        b = self.bottleneck(self.pool(e))        # [B, 64, 48, 160]
        self._log("pool+bottleneck", b)

        # U-Net decoder + skip
        d = self.up(b)                           # [B, 32, 96, 320]
        if d.shape != e.shape:
            d = F.interpolate(d, size=e.shape[2:], mode='bilinear', align_corners=False)
        d = self.dec(torch.cat([d, e], dim=1))   # [B, 32, 96, 320]
        self._log("dec (after skip)", d)

        # High 해상도로 복원
        d = F.interpolate(d, size=High_feature.shape[2:],
                          mode='bilinear', align_corners=False)  # [B, 32, 160, 512]
        self._log("upsample to High", d)

        # 최종 skip: 원본 High_feature 합치기
        out = self.final(torch.cat([d, High_feature], dim=1))    # [B, 32, 160, 512]
        self._log("final output", out)

        return out