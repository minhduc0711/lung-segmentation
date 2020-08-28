import torch


def dice_coeff(input, target):
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    input = input.to(dtype=torch.float)
    target = target.to(dtype=torch.float)

    batch_size = input.shape[0]
    for i in range(batch_size):
        eps = 0.0001  # for numerical stability
        inter = torch.dot(input[i].reshape(-1), target[i].reshape(-1))
        # this is not actually union
        union = torch.sum(input[i]) + torch.sum(target[i])
        s += (2 * inter.float() + eps) / (union.float() + eps)
    return s / batch_size


def dice_coeff_vectorized(input, target):
    eps = 1e-4
    batch_size = input.shape[0]
    input = input.float().reshape(batch_size, -1)
    target = target.float().reshape(batch_size, -1)

    inter = (input * target).sum(dim=1)
    union = input.sum(dim=1) + target.sum(dim=1)
    dice_coeffs = (2 * inter + eps) / (union + eps)
    avg_dice = dice_coeffs.mean()
    return avg_dice
