import torch


def add(ws1, ws2):
    return {k: ws1[k] + ws2[k] for k in ws1.keys()}


def mul(ws, c):
    return {k: ws[k] * c for k in ws.keys()}


def diff(ws1, ws2):
    return add(ws1, mul(ws2, -1))


def norm(ws):
    dot = inner(ws, ws)

    return torch.sqrt(dot)


def rad(ws1, ws2):
    return norm(diff(ws1, ws2))


def inner(ws1, ws2):
    dot = {k: torch.sum(ws1[k] * ws2[k]) for k in ws1.keys()}
    dot = torch.sum(torch.tensor(list(dot.values())))

    return dot


def cos(ws1, ws2):
    norm1 = norm(ws1)
    norm2 = norm(ws2)
    dot = inner(ws1, ws2)

    return dot / (norm1 * norm2 + 1e-7)


def sin(ws1, ws2):
    cosv = cos(ws1, ws2)

    return torch.sqrt((1 + cosv) * (1 - cosv + 1e-7)) if cosv < 1.0 else torch.tensor(0.0)
