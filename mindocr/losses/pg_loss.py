import mindspore as ms
from mindspore import nn
import mindspore.ops as ops
import numpy as np
from mindocr.losses.det_loss import DiceLoss
from mindocr.utils.e2e_utils.extract_batchsize import pre_process

__all__ = ["PGLoss"]

class PGLoss(nn.LossBase):
    def __init__(
        self, tcl_bs, max_text_length, max_text_nums, pad_num, eps=1e-6, **kwargs
    ):
        super(PGLoss, self).__init__()
        self.tcl_bs = tcl_bs
        self.max_text_nums = max_text_nums
        self.max_text_length = max_text_length
        self.pad_num = pad_num
        self.dice_loss = DiceLoss(eps=eps)
        self.cast = ops.Cast()

    def border_loss(self, f_border, l_border, l_score, l_mask):
        # 可以看下wuhao的实现，很多函数可以简化
        l_border_split, l_border_norm = ops.split(
            l_border, (4, 1), axis=1
        )
        f_border_split = f_border
        b, c, h, w = ops.shape(l_border_norm)
        l_border_norm_split = ops.broadcast_to(l_border_norm, (b, 4 * c, h, w))
        b, c, h, w = ops.shape(l_score)
        l_border_score = ops.broadcast_to(l_score, (b, 4 * c, h, w))
        b, c, h, w = ops.shape(l_mask)
        l_border_mask = ops.broadcast_to(l_mask, (b, 4 * c, h, w))
        border_diff = l_border_split - f_border_split
        abs_border_diff = ops.abs(border_diff)
        border_sign = abs_border_diff < 1.0
        border_sign = self.cast(border_sign, ms.float32)
        border_sign.stop_gradient = True
        border_in_loss = 0.5 * abs_border_diff * abs_border_diff * border_sign + (
            abs_border_diff - 0.5
        ) * (1.0 - border_sign)
        border_out_loss = l_border_norm_split * border_in_loss
        border_loss = ops.sum(border_out_loss * l_border_score * l_border_mask) / (
            ops.sum(l_border_score * l_border_mask) + 1e-5
        )
        return border_loss

    def direction_loss(self, f_direction, l_direction, l_score, l_mask):
        l_direction_split, l_direction_norm = ops.split(
            l_direction, (2, 1), axis=1
        )
        f_direction_split = f_direction
        b, c, h, w = l_direction_norm.shape
        l_direction_norm_split = ops.broadcast_to(
            l_direction_norm, (b, 2 * c, h, w)
        )
        b, c, h, w = ops.shape(l_score)
        l_direction_score = ops.broadcast_to(l_score, (b, 2 * c, h, w))
        b, c, h, w = ops.shape(l_mask)
        l_direction_mask = ops.broadcast_to(l_mask, (b, 2 * c, h, w))
        direction_diff = l_direction_split - f_direction_split
        abs_direction_diff = ops.abs(direction_diff)
        direction_sign = abs_direction_diff < 1.0
        direction_sign = self.cast(direction_sign, ms.int32)
        direction_sign.stop_gradient = True
        direction_in_loss = (
            0.5 * abs_direction_diff * abs_direction_diff * direction_sign
            + (abs_direction_diff - 0.5) * (1.0 - direction_sign)
        )
        direction_out_loss = l_direction_norm_split * direction_in_loss
        direction_loss = ops.sum(
            direction_out_loss * l_direction_score * l_direction_mask
        ) / (ops.sum(l_direction_score * l_direction_mask) + 1e-5)
        return direction_loss

    #wuhao的ctcloss和该函数差别很大
    def ctcloss(self, f_char, tcl_pos, tcl_mask, tcl_label, label_t):
        # 将输入特征图f_char的维度从[N, C, H, W]转换为[N, H, W, C]，  
        # 通常是为了符合后续操作（如gather_nd）对维度顺序的要求
        f_char = ops.transpose(f_char, (0, 2, 3, 1))
        # 将tcl_pos重塑为[-1, 3]的形状，  
        # 通常tcl_pos包含了用于从f_char中选取特定位置的索引。
        tcl_pos = ops.reshape(tcl_pos, (-1, 3))
        # 将tcl_pos的数据类型转换为整型，  
        # 因为索引操作通常需要整型数据。
        tcl_pos = self.cast(tcl_pos, ms.int64)
        # 使用tcl_pos作为索引，从f_char中选取特定的数据，  
        # 然后重塑为[-1, 64, self.pad_num + 1]的形状
        f_tcl_char = ops.gather_nd(f_char, tcl_pos)
        f_tcl_char = ops.reshape(
            f_tcl_char, (-1, 64, self.pad_num + 1)
        )  # len(Lexicon_Table)+1
        # 将f_tcl_char分割成前景（f_tcl_char_fg）和背景（f_tcl_char_bg），  
        # 沿着类别维度（axis=2）进行分割，  
        # self.pad_num是前景类别的数量，1是背景（空白）类别的数量
        f_tcl_char_fg, f_tcl_char_bg = ops.split(
            f_tcl_char, (self.pad_num, 1), axis=2
        )
         # 使用tcl_mask调整f_tcl_char_bg的值，  
        # 对于被掩码的位置，乘以一个较大的数（如20.0），  
        # 对于未被掩码的位置，保持其值不变（但由于后面的加法操作，这实际上会被覆盖）
        f_tcl_char_bg = f_tcl_char_bg * tcl_mask + (1.0 - tcl_mask) * 20.0
        # 获取tcl_mask的形状，并据此扩展tcl_mask以匹配f_tcl_char_fg的形状，  
        # 用于后续的前景掩码操作
        b, c, l = ops.shape(tcl_mask)
        tcl_mask_fg = ops.broadcast_to(tcl_mask, (b, c, self.pad_num * l))
        tcl_mask_fg.stop_gradient = True
        # 使用tcl_mask_fg调整f_tcl_char_fg的值，  
        # 对于被掩码的位置，保持其值不变，  
        # 对于未被掩码的位置，乘以一个较小的数（如-20.0），以抑制这些位置的损失。
        f_tcl_char_fg = f_tcl_char_fg * tcl_mask_fg + (1.0 - tcl_mask_fg) * (-20.0)
        # 将前景和背景合并回一个张量，准备进行CTC损失计算
        f_tcl_char_mask = ops.cat((f_tcl_char_fg, f_tcl_char_bg), axis=2)
        # 将f_tcl_char_mask的维度从[N, T, C]转换为[T, N, C]，  
        # 以符合CTC损失函数的输入要求
        f_tcl_char_ld = ops.transpose(f_tcl_char_mask, (1, 0, 2))
        log_softmax = nn.LogSoftmax()
        # 获取f_tcl_char_ld的形状，并据此计算每个输入序列的长度（这里假设所有序列长度相同为N）
        N, B, _ = f_tcl_char_ld.shape
        input_lengths = ms.Tensor(np.array([N] * B, dtype=np.int64), ms.int64)
        f_tcl_char_ld = log_softmax(f_tcl_char_ld)
        # 计算CTC损失，使用指定的参数
        ctc_loss = nn.CTCLoss(blank=self.pad_num, reduction="mean", zero_infinity=False)
        cost= ctc_loss(f_tcl_char_ld, tcl_label, input_lengths, label_t)
        return cost

    def construct(self, predicts, tcl_maps, tcl_label_maps, border_maps, direction_maps, training_masks, label_list, pos_list, pos_mask):
        # for all the batch_size
        pos_list, pos_mask, label_list, label_t = pre_process(
            label_list,
            pos_list,
            pos_mask,
            self.max_text_length,
            self.max_text_nums,
            self.pad_num,
            self.tcl_bs,
        )
        f_score, f_border, f_direction, f_char = (
            predicts["f_score"],
            predicts["f_border"],
            predicts["f_direction"],
            predicts["f_char"],
        )
        score_loss = self.dice_loss(f_score, tcl_maps, training_masks)
        border_loss = self.border_loss(f_border, border_maps, tcl_maps, training_masks)
        direction_loss = self.direction_loss(
            f_direction, direction_maps, tcl_maps, training_masks
        )
        ctc_loss = self.ctcloss(f_char, pos_list, pos_mask, label_list, label_t)
        loss_all = score_loss + border_loss + direction_loss + 5 * ctc_loss
        # losses = {
        #     "loss": loss_all,
        #     "score_loss": score_loss,
        #     "border_loss": border_loss,
        #     "direction_loss": direction_loss,
        #     "ctc_loss": ctc_loss,
        # }
        return loss_all
