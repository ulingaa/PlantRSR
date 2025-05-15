import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
import MinkowskiEngine as ME

# import torchsparse
# import torchsparse.nn as spnn
# from torchsparse import SparseTensor


def sample_patches(inputs, patch_size=3, stride=1):
    """Extract sliding local patches from an input feature tensor.
    The sampled pathes are row-major.

    Args:
        inputs (Tensor): the input feature maps, shape: (c, h, w).
        patch_size (int): the spatial size of sampled patches. Default: 3.
        stride (int): the stride of sampling. Default: 1.

    Returns:
        patches (Tensor): extracted patches, shape: (c, patch_size,
            patch_size, n_patches).
    """

    c, h, w = inputs.shape


    patches_ref = inputs.unsqueeze(0)
    res_ref = patches_ref - F.interpolate(F.interpolate(patches_ref, (h//4, w//4), mode='bilinear', align_corners=False), (h,w), mode='bilinear', align_corners=False) 
    ref_mag = res_ref.abs().mean(dim=1)
    threshold_ref = ref_mag.mean() + ref_mag.std()
    mask_ref = ref_mag > threshold_ref

    sparse_ref_input = patches_ref * mask_ref.unsqueeze(1).float()  # [1, C, H, W]
    patches_ref = F.unfold(sparse_ref_input, kernel_size=3, padding=0, stride=1)  # [B, C*3*3, L] where L = (H-2)*(W-2)
    patches_ref = patches_ref.squeeze(0).transpose(0, 1)  # [L, C*9]

    nonzero_mask = patches_ref.abs().sum(dim=1) > 0  # [L] boolean mask

    valid_patches_ref = patches_ref[nonzero_mask]        # [N_valid, C*9]
    valid_indices = torch.arange(patches_ref.size(0))[nonzero_mask]  # 原始索引 [N_valid]


    patches_ref = valid_patches_ref.view(-1, c, 3, 3).permute(1, 2, 3, 0)


    # patches = inputs.unfold(1, patch_size, stride)\ #(256,38,40,3)
    #                 .unfold(2, patch_size, stride)\ #(256,38,38,3,3)
    #                 .reshape(c, -1, patch_size, patch_size)\ #(256,1444,3,3)
    #                 .permute(0, 2, 3, 1) #(256,3,3,1444)
    # patches = inputs.unfold(1, patch_size, stride).unfold(2, patch_size, stride).reshape(c, -1, patch_size, patch_size).permute(0, 2, 3, 1)
    # return patches
    return patches_ref, valid_indices


def feature_match_index(feat_input,
                        feat_ref,
                        patch_size=3,
                        input_stride=1,
                        ref_stride=1,
                        is_norm=True,
                        norm_input=False):

    # patch decomposition, shape: (c, patch_size, patch_size, n_patches)
    patches_ref, valid_indices = sample_patches(feat_ref, patch_size, ref_stride) #(256,40,40)-->(256,3,3,38*38)

    # batch-wise matching because of memory limitation
    _, h, w = feat_input.shape

    input_c,_,_,output_c = patches_ref.shape
    if is_norm:
        patches_ref = patches_ref / (patches_ref.norm(p=2, dim=(0, 1, 2)) + 1e-5)

    #corr1 = F.conv2d(feat_input.unsqueeze(0),batch.permute(3, 0, 1, 2),stride=input_stride)

    #dense-->sparse
    feature_input = feat_input.unsqueeze(0) #[256,40,40]-->[1,256,40,40] # dense feature
    res = feature_input - F.interpolate(F.interpolate(feature_input, (h//2, w//2), mode='bilinear', align_corners=False), (h,w), mode='bilinear', align_corners=False) 
    res_mag = res.abs().mean(dim=1)
    threshold = res_mag.mean() + res_mag.std()
    mask = res_mag > threshold
    m = mask
  
    mask[[0,-1], :] = True
    mask[:, [0,-1]] = True

    coord = torch.nonzero(mask)  
    batch_indices = coord[:,0]
    x_indices = coord[:,1]
    y_indices = coord[:,2]   

    #[1,256,40,40]-->[117,256]
    feature_input = feature_input[batch_indices,:,x_indices,y_indices] #sparse feature
    #[117,256]
    sparse_input = ME.SparseTensor(
        features=feature_input,   # [N, 256] #N
        coordinates=coord.int().contiguous(), # [N, 3] (batch_idx, x, y)
        tensor_stride=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    sparse_conv = ME.MinkowskiConvolution(
        in_channels=input_c, 
        out_channels=output_c,
        kernel_size=3,
        stride=1,
        dimension=2,  
        bias= False,
    ).cuda()

    custom_weight = patches_ref.permute(2, 1, 0, 3).contiguous().view(9, input_c, output_c)  # (9, 256, 1444)


    with torch.no_grad():
        sparse_conv.kernel.data.copy_(custom_weight.to("cuda"))

    #[117,256],[38*38,256,3*3]
    corr = sparse_conv(sparse_input) # [1600,9]
    corr= corr.dense()[0] #[1,38*38,37,37]
    corr = corr[:, :, 1:-1, 1:-1] #[1,38*38,38,38]
    _, max_idx_tmp = corr.squeeze(0).max(dim=0) #[38,38],[38,38]
    max_idx = valid_indices[max_idx_tmp]

    return max_idx, m







