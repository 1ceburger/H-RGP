"""Hydrological model implementation."""
import copy, torch

class EMAWeights:

    def __init__(self, model, decay=0.99):
        assert 0 < decay < 1
        self.decay = float(decay)
        self.updates = 0
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def update(self, online):
        source = dict(online.named_parameters())
        for name, p in self.model.named_parameters():
            q = source[name]
            if q.requires_grad:
                p.mul_(self.decay).add_(q.detach(), alpha=1 - self.decay)
            else:
                p.copy_(q.detach())
        for name, buf in self.model.named_buffers():
            buf.copy_(dict(online.named_buffers())[name])
        self.updates += 1

    def state_dict(self):
        return dict(model=self.model.state_dict(), decay=self.decay, updates=self.updates)

    def load_state_dict(self, state):
        assert state['decay'] == self.decay
        self.model.load_state_dict(state['model'])
        self.updates = int(state['updates'])
        self.model.requires_grad_(False)
        self.model.eval()
