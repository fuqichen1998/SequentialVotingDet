import torch


def box_field_voting(xs, ys, hs, ws, confidences, h, w):
    device = xs.device
    # discretize
    xs = torch.clamp(torch.round(xs), 0, w-1).long()
    ys = torch.clamp(torch.round(ys), 0, h-1).long()
    hs = torch.clamp(torch.round(hs), 0, h-1).long()
    ws = torch.clamp(torch.round(ws), 0, w-1).long()
    # vote for center
    center_hm = torch.zeros((h, w)).to(device)
    center_hm.index_put_((ys, xs), confidences, accumulate=True)
    max_idx = center_hm.argmax()
    y_pred = max_idx // w
    x_pred = max_idx % w
    # vote for height
    h_hm = torch.zeros(h).to(device)
    h_hm.index_put_((hs,), confidences, accumulate=True)
    h_pred = h_hm.argmax()
    # vote for width
    w_hm = torch.zeros(w).to(device)
    w_hm.index_put_((ws,), confidences, accumulate=True)
    w_pred = w_hm.argmax()
    return x_pred, y_pred, h_pred, w_pred


def maksed_inliers_gaussian_sum(xs, ys, hs, ws, confidences, x, y, h, w, tol=0.1, eps=1e-7):
    mask = (((xs - x) / (w + eps)) < tol) * (((ys - y) / (h + eps)) < tol) * \
        (((ws - w) / (w + eps)) < tol) * (((hs - h) / (h + eps)) < tol)
    x_pred = torch.sum(xs[mask] * confidences[mask]) / (torch.sum(confidences[mask]) + eps)
    y_pred = torch.sum(ys[mask] * confidences[mask]) / (torch.sum(confidences[mask]) + eps)
    h_pred = torch.sum(hs[mask] * confidences[mask]) / (torch.sum(confidences[mask]) + eps)
    w_pred = torch.sum(ws[mask] * confidences[mask]) / (torch.sum(confidences[mask]) + eps)
    confidence = torch.mean(confidences[mask])
    return x_pred, y_pred, h_pred, w_pred, confidence