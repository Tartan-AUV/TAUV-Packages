from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

	def __init__(self, in_channels: int, out_channels: int, stride: int):
		super().__init__()

		self.conv1 = nn.Conv2d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=3,
			stride=stride,
			padding=1,
		)
		self.bn1 = nn.BatchNorm2d(out_channels)

		self.conv2 = nn.Conv2d(
			in_channels=out_channels,
			out_channels=out_channels,
			kernel_size=3,
			stride=1,
			padding=1,
		)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.conv_residual = nn.Conv2d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=1,
			stride=stride,
		)
		self.bn_residual = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		residual = self.conv_residual(x)
		residual = self.bn_residual(residual)

		y = self.conv1(x)
		y = self.bn1(y)
		y = F.relu(y)

		y = self.conv2(y)
		y = self.bn2(y)
		y += residual
		y = F.relu(y)

		return y


# When it's time for deform conv stuff, use MMCV package
# https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/csrc/pytorch/deform_conv.cpp

class Root(nn.Module):

	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()

		self.conv = nn.Conv2d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=1,
			stride=1
		)
		self.bn = nn.BatchNorm2d(out_channels)

	def forward(self, children: List[torch.Tensor]) -> torch.Tensor:
		x = self.conv(torch.cat(children, 1))
		x = self.bn(x)
		x = F.relu(x)

		return x


class Tree(nn.Module):

	def __init__(self, in_channels: int, out_channels: int, height: int, root_channels: Optional[int], block: nn.Module, stride: int):
		super().__init__()

		self.height = height

		if root_channels is None:
			root_channels = 2 * out_channels

		if height == 1:
			self.tree_l = block(
				in_channels=in_channels,
				out_channels=out_channels,
				stride=stride,
			)
			self.tree_r = block(
				in_channels=out_channels,
				out_channels=out_channels,
				stride=1,
			)
			self.root = Root(
				in_channels=root_channels,
				out_channels=out_channels
			)
		else:
			self.tree_l = Tree(
				in_channels=in_channels,
				out_channels=out_channels,
				height=height - 1,
				root_channels=None,
				block=block,
				stride=stride,
			)
			self.tree_r = Tree(
				in_channels=out_channels,
				out_channels=out_channels,
				height=height - 1,
				root_channels=root_channels + out_channels,
				block=block,
				stride=1,
			)
			self.root = None

	def forward(self, x: torch.Tensor, children: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
		if children is None:
			children = []

		xl = self.tree_l(x)

		if self.height == 1:
			xr = self.tree_r(xl)
			x = self.root(children + [xl, xr])
		else:
			x = self.tree_r(xl, children=children + [xl])

		return x


class DLADown(nn.Module):

	def __init__(self, heights: List[int], channels: List[int], downsamples: int, block: nn.Module):
		super().__init__()

		self.projection_layer = nn.Sequential(
			nn.Conv2d(
				in_channels=3,
				out_channels=channels[0],
				kernel_size=7,
				stride=1,
				padding=3,
			),
			nn.BatchNorm2d(channels[0]),
			nn.ReLU(),
		)

		self.block_layers = nn.Sequential(
			*[
				ResidualBlock(
					in_channels=channels[0],
					out_channels=channels[0],
					stride=2,
				)
				for _ in range(downsamples)
			]
		)

		tree_layers = []

		for tree_layer_i in range(len(heights)):
			layer = Tree(
				in_channels=channels[tree_layer_i],
				out_channels=channels[tree_layer_i + 1],
				height=heights[tree_layer_i],
				root_channels=None,
				block=block,
				stride=2,
			)

			tree_layers.append(layer)

		self.tree_layers = nn.ModuleList(tree_layers)

	def forward(self, img: torch.Tensor) -> List[torch.Tensor]:
		x = self.projection_layer(img)
		x = self.block_layers(x)

		y = [x]

		for tree_layer in self.tree_layers:
			x = tree_layer(x)
			y.append(x)

		return y


def pad_to_match(feature: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
	if feature.shape == shape:
		return feature

	pad_above = max(0, (feature.shape[2] - shape[2]) // 2)
	pad_below = max(0, shape[2] - feature.shape[2] - pad_above)

	pad_left = max(0, (feature.shape[3] - shape[3]) // 2)
	pad_right = max(0, shape[3] - feature.shape[3] - pad_left)

	padded_feature = F.pad(feature, (pad_above, pad_below, pad_left, pad_right))

	result = padded_feature[:, :, :shape[2], :shape[3]]

	return result


class IDAUp(nn.Module):

	def __init__(self, feature_channels: List[int], scales: List[int]):
		super().__init__()

		assert len(scales) == len(feature_channels) - 1

		projection_layers = []
		output_layers = []
		upsample_layers = []

		self.feature_channels = feature_channels

		for feature_i in range(len(feature_channels) - 1):
			projection_layer = nn.Sequential(
				nn.Conv2d(
					in_channels=feature_channels[feature_i + 1],
					out_channels=feature_channels[feature_i],
					kernel_size=3,
					padding=1,
					stride=1,
				),
				nn.BatchNorm2d(feature_channels[feature_i]),
				nn.ReLU(),
			)

			output_layer = nn.Sequential(
				nn.Conv2d(
					in_channels=feature_channels[feature_i],
					out_channels=feature_channels[feature_i],
					kernel_size=3,
					padding=1,
					stride=1,
				),
				nn.BatchNorm2d(feature_channels[feature_i]),
				nn.ReLU(),
			)

			upsample_layer = nn.ConvTranspose2d(
				in_channels=feature_channels[feature_i],
				out_channels=feature_channels[feature_i],
				kernel_size=scales[feature_i],
				stride=scales[feature_i],
			)

			projection_layers.append(projection_layer)
			output_layers.append(output_layer)
			upsample_layers.append(upsample_layer)

		self.projection_layers = nn.ModuleList(projection_layers)
		self.output_layers = nn.ModuleList(output_layers)
		self.upsample_layers = nn.ModuleList(upsample_layers)

	def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
		assert len(features) == len(self.projection_layers) + 1

		new_features = []

		new_feature = features[-1]

		for feature_i in reversed(range(len(features) - 1)):
			project = self.projection_layers[feature_i]
			output = self.output_layers[feature_i]
			upsample = self.upsample_layers[feature_i]

			upsampled_feature = upsample(project(new_feature))
			new_feature = output(features[feature_i] + pad_to_match(upsampled_feature, features[feature_i].shape))

			new_features.append(new_feature)

		new_features = list(reversed(new_features))

		return new_features


class IDAUpReverse(nn.Module):

	def __init__(self, feature_channels: List[int], scales: List[int]):
		super().__init__()

		assert len(scales) == len(feature_channels) - 1

		projection_layers = []
		output_layers = []
		upsample_layers = []

		self.feature_channels = feature_channels

		for feature_i in range(len(feature_channels) - 1):
			projection_layer = nn.Sequential(
				nn.Conv2d(
					in_channels=feature_channels[feature_i + 1],
					out_channels=feature_channels[0],
					kernel_size=3,
					padding=1,
					stride=1,
				),
				nn.BatchNorm2d(feature_channels[0]),
				nn.ReLU(),
			)

			output_layer = nn.Sequential(
				nn.Conv2d(
					in_channels=feature_channels[0],
					out_channels=feature_channels[0],
					kernel_size=3,
					padding=1,
					stride=1,
				),
				nn.BatchNorm2d(feature_channels[0]),
				nn.ReLU(),
			)

			upsample_layer = nn.ConvTranspose2d(
				in_channels=feature_channels[0],
				out_channels=feature_channels[0],
				kernel_size=scales[feature_i],
				stride=scales[feature_i],
			)

			projection_layers.append(projection_layer)
			output_layers.append(output_layer)
			upsample_layers.append(upsample_layer)

		self.projection_layers = nn.ModuleList(projection_layers)
		self.output_layers = nn.ModuleList(output_layers)
		self.upsample_layers = nn.ModuleList(upsample_layers)

	def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
		assert len(features) == len(self.projection_layers) + 1

		new_features = []

		new_feature = features[0]

		for feature_i in range(len(features) - 1):
			project = self.projection_layers[feature_i]
			output = self.output_layers[feature_i]
			upsample = self.upsample_layers[feature_i]

			upsampled_feature = upsample(project(features[feature_i + 1]))
			new_feature = output(new_feature + pad_to_match(upsampled_feature, new_feature.shape))

			new_features.append(new_feature)

		return new_features


class MultiIDAUp(nn.Module):

	def __init__(self, feature_channels: List[int]):
		super().__init__()

		ida_up_layers = []

		for feature_i in range(len(feature_channels) - 1):
			ida_up_layer = IDAUp(
				feature_channels=feature_channels[:len(feature_channels) - feature_i],
				scales=[2 for _ in range(len(feature_channels) - feature_i - 1)],
			)

			ida_up_layers.append(ida_up_layer)

		self.ida_up_layers = nn.ModuleList(ida_up_layers)

	def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
		assert len(features) == len(self.ida_up_layers) + 1

		out = []

		for i in range(len(features) - 1):
			features = self.ida_up_layers[i](features)
			out.append(features[-1])

		assert len(features) == 1

		out = list(reversed(out))

		return out


class DLABackbone(nn.Module):

	def __init__(self, heights: List[int], channels: List[int], downsamples: int):
		super().__init__()

		block = ResidualBlock

		self.dla_down = DLADown(heights, channels, downsamples, block)
		self.multi_ida_up = MultiIDAUp(channels)
		self.ida_up_reverse = IDAUpReverse(
			feature_channels=channels[:len(channels) - 1],
			scales=[2 ** i for i in range(1, len(channels) - 1)],
		)

		self.out_channels = channels[0]

	def forward(self, img: torch.Tensor) -> torch.Tensor:
		features = self.dla_down.forward(img)
		features = self.multi_ida_up.forward(features)

		# 128, 128, 64, 64
		features = self.ida_up_reverse(features)

		return features[-1]


def main():
	img = torch.rand((2, 3, 512, 512))

	heights = [1, 2, 2, 1]
	channels = [64, 64, 128, 128, 256]

	backbone = DLABackbone(heights, channels)

	out = backbone.forward(img)

	pass


if __name__ == "__main__":
	main()
