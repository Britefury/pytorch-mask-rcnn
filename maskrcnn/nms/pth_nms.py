import torch
from ._ext import nms


def pth_nms(dets, thresh):
  """
  dets has to be a tensor
  """
  device = dets.device

  if not dets.is_cuda:
    x1 = dets[:, 1]
    y1 = dets[:, 0]
    x2 = dets[:, 3]
    y2 = dets[:, 2]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.sort(0, descending=True)[1]
    # order = torch.from_numpy(np.ascontiguousarray(scores.numpy().argsort()[::-1])).long()

    keep = torch.zeros(dets.size(0), dtype=torch.long, device=device)
    num_out = torch.zeros(1, dtype=torch.long, device=device)
    nms.cpu_nms(keep, num_out, dets, order, areas, thresh)

    return keep[:num_out[0]]
  else:
    x1 = dets[:, 1]
    y1 = dets[:, 0]
    x2 = dets[:, 3]
    y2 = dets[:, 2]
    scores = dets[:, 4]

    dets_temp = torch.zeros(dets.size(), dtype=torch.float, device=device)
    dets_temp[:, 0] = dets[:, 1]
    dets_temp[:, 1] = dets[:, 0]
    dets_temp[:, 2] = dets[:, 3]
    dets_temp[:, 3] = dets[:, 2]
    dets_temp[:, 4] = dets[:, 4]

    # areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.sort(0, descending=True)[1]
    # order = torch.from_numpy(np.ascontiguousarray(scores.cpu().numpy().argsort()[::-1])).long().cuda()

    dets = dets[order].contiguous()

    # keep and num_out are CPU tensors, NOT GPU
    keep = torch.zeros(dets.size(0), dtype=torch.long)
    num_out = torch.zeros(1, dtype=torch.long)
    nms.gpu_nms(keep, num_out, dets_temp, thresh)

    return order[keep[:num_out[0]].to(device)].contiguous()
    # return order[keep[:num_out[0]]].contiguous()

