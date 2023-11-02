import torch

###################################################
# Transformation Matrices Creation
###################################################


def makeTmats(ctheta, stheta, cchi, schi, lens):
    """Make transformation matrices given angles/lens.

    ctheta is used to determine device and number of matrices.
    """
    N = ctheta.shape[0]
    mats = torch.zeros((N, 4, 4), device=ctheta.device, dtype=ctheta.dtype)

    mats[:, 0, 0] = ctheta
    mats[:, 0, 1] = -stheta
    mats[:, 0, 3] = ctheta * lens

    mats[:, 1, 0] = cchi * stheta
    mats[:, 1, 1] = ctheta * cchi
    mats[:, 1, 2] = schi
    mats[:, 1, 3] = cchi * lens * stheta

    mats[:, 2, 0] = -stheta * schi
    mats[:, 2, 1] = -ctheta * schi
    mats[:, 2, 2] = cchi
    mats[:, 2, 3] = -lens * stheta * schi

    mats[:, 3, 3] = 1.0

    return mats


def makeTmats_backward(grad_outputs, ctheta, stheta, cchi, schi, lens):
    """Backwards function for makeTmats."""
    # yapf: disable
    grad = grad_outputs
    grad_ctheta = grad[:,0,0] + lens*grad[:,0,3] + cchi*grad[:,1,1] + -schi*grad[:,2,1]
    grad_stheta = -grad[:,0,1] + cchi*grad[:,1,0] + cchi*lens*grad[:,1,3] - schi*grad[:,2,0] - lens*schi*grad[:,2,3]
    grad_cchi = stheta*grad[:,1,0] + ctheta*grad[:,1,1] + lens*stheta*grad[:,1,3] + grad[:,2,2]
    grad_schi = grad[:,1,2] + -stheta*grad[:,2,0] - ctheta*grad[:,2,1] - lens*stheta*grad[:,2,3]
    # lengths are typically constant.. should we check requires grad?
    grad_lens = ctheta*grad[:,0,3] + cchi*stheta*grad[:,1,3] - stheta*schi*grad[:,2,3]
    # yapf: enable

    return grad_ctheta, grad_stheta, grad_cchi, grad_schi, grad_lens


class MakeTMats(torch.autograd.Function):

    @staticmethod
    def forward(ctx, ctheta, stheta, cchi, schi, lens):
        ctx.save_for_backward(ctheta, stheta, cchi, schi, lens)
        return makeTmats(ctheta, stheta, cchi, schi, lens)

    @staticmethod
    def backward(ctx, grad):
        return makeTmats_backward(grad, *ctx.saved_tensors)


###################################################
# Backbone
###################################################


def make_backbone_serial(ncacM):
    """Make backbone matrices from indiv. transformation matrices by multiplying in order.

    Modifies in-place.
    """
    for i in range(1, len(ncacM)):
        ncacM[i] = ncacM[i - 1] @ ncacM[i]
    return ncacM


def make_backbone_(M):
    """Make backbone matrices from indiv. transformation matrices using log(N) multiplies.

    Backbone matrices are created from inidividual transformation matrices via log(N)
    parallel matrix multiplies operations. Modifies in place.
    """
    N = len(M)
    indices = torch.arange(0, N, device=M.device)

    for i in range(N.bit_length()):
        dstmask = ((1 << i) & indices) != 0
        srci = ((-1 << i) & indices) - 1

        src = M[srci[dstmask]]
        dst = M[dstmask]

        M[dstmask] = src @ dst

    return M


def make_backbone_save(M):
    """Make backbone matrices from indiv. transformation matrices using log(N) multiplies.

    Make backbone matrices from individual transformation matrices using log(N) parallel
    matrix multiplies. Does not modify in place. Not very memory efficient in the
    interest of simplicity.

    Returns transformation matrices at each iteration (last is final result).
    """
    N = len(M)
    bit_length = N.bit_length()

    indices = torch.arange(0, N, device=M.device)
    ret = torch.zeros(bit_length, N, 4, 4, device=M.device, dtype=M.dtype)

    for i in range(N.bit_length()):
        ret[i] = M
        dstmask = ((1 << i) & indices) != 0
        srci = ((-1 << i) & indices) - 1

        src = M[srci[dstmask]]
        dst = M[dstmask]

        M = M.clone()
        M[dstmask] = src @ dst

    return M, ret


def make_backbone_backward(grad, M, saved):
    """Compute gradient of M."""
    N = len(M)
    indices = torch.arange(0, N, device=grad.device)
    grad = grad.clone()  # make sure contiguous
    for i in range(N.bit_length() - 1, -1, -1):
        dstmask = ((1 << i) & indices) != 0
        srci = ((-1 << i) & indices) - 1
        srcmask = srci[dstmask]
        src = saved[i][srcmask]
        dst = saved[i][dstmask]

        srcgrad = grad[dstmask] @ dst.transpose(1, 2)  # src grad
        grad[dstmask] = src.transpose(1, 2) @ grad[dstmask]

        index = srcmask[:, None, None].repeat(1, 4, 4)
        grad.scatter_add_(0, index, srcgrad)

    return grad


class MakeBackbone(torch.autograd.Function):

    @staticmethod
    def forward(ctx, ncac):
        if ncac.requires_grad:
            M, saved = make_backbone_save(ncac)
            ctx.save_for_backward(ncac, saved)
            return M
        else:
            return make_backbone_(ncac.clone())

    @staticmethod
    def backward(ctx, grad):
        return make_backbone_backward(grad, *ctx.saved_tensors)


class MakeSCCoords(torch.autograd.Function):

    @staticmethod
    def forward(ctx, backbone, seq_aa_index, ang, build_info):
        ctx.save_for_backward(backbone, ang)
        ctx.build_info = build_info
        ctx.seq_aa_index = seq_aa_index
        return make_sidechain_coords(backbone, seq_aa_index, ang, build_info)

    @staticmethod
    def backward(ctx, grad):
        # implementing this can make make_coords 3X faster
        raise NotImplementedError
