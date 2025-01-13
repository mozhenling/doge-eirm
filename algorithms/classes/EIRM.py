import torch
import torch.autograd as autograd
from algorithms.classes.ERM import ERM

class EIRM(ERM):
    """ Extended Invariant Risk Minimization """
    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(EIRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams,args)

    def update(self, minibatches, unlabeled=None):
        loss = 0
        grads = [0 for _ in self.network.parameters()]
        # save original sate dict
        state_dict = self.network.state_dict()
        self.network.zero_grad()
        for i, (x, y) in enumerate(minibatches):
            # load original state
            if i >0:
                self.network.load_state_dict(state_dict)
                self.network.zero_grad()
            # loss 1 and grad 1
            loss_1 = self.erm_loss(self.predict(x), y)
            grads_1 = autograd.grad(loss_1, self.network.parameters())
            # temp model update
            with torch.no_grad():
                for p, g in zip(self.network.parameters(), grads_1):
                    p.add_(g * self.hparams["lr"] ) # can be another independent hyper-params to increase flexibility
            # loss 2 and grad 2
            loss_2 = self.erm_loss(self.predict(x), y)
            grads_2 = autograd.grad(loss_2, self.network.parameters())

            # final loss and grads
            with torch.no_grad():
                for j, (g1, g2) in enumerate(zip( grads_1, grads_2)):
                    grads[j] += g1 + self.hparams["p_weight"] * (loss_2-loss_1)*(g2-g1)
            loss += loss_2 + loss_1

        # average
        loss /= len(minibatches)
        grads =[g/len(minibatches) for g in grads]
        # back to original
        self.network.load_state_dict(state_dict)
        self.network.zero_grad()
        self.optimizer.zero_grad()
        # update model by grads
        with torch.no_grad():
            for p, g in zip(self.network.parameters(), grads):
                p.grad = g
        # optimizer uses the p.grad to update the p
        self.optimizer.step()

        return {"loss": loss.item()}