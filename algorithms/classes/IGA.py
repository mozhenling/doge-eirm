

import torch.nn.functional as F
import torch.autograd as autograd
from algorithms.classes.ERM import ERM

class IGA(ERM):
    """
    Inter-environmental Gradient Alignment
    From https://arxiv.org/abs/2008.01883v2
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, args ):
        super(IGA, self).__init__(input_shape, num_classes, num_domains, hparams, args )

    def update(self, minibatches, unlabeled=None):
        total_loss = 0
        grads = []
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            total_loss += env_loss

            env_grad = autograd.grad(env_loss, self.network.parameters(),
                                        create_graph=True)

            grads.append(env_grad)

        mean_loss = total_loss / len(minibatches)
        mean_grad = autograd.grad(mean_loss, self.network.parameters(),
                                        retain_graph=True)

        # compute trace penalty
        penalty_value = 0
        for grad in grads:
            for g, mean_g in zip(grad, mean_grad):
                penalty_value += (g - mean_g).pow(2).sum()

        objective = mean_loss + self.hparams['iga_penalty_weight'] * penalty_value

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        if self.args.scheduler:
            self.scheduler.step()

        return {'loss': mean_loss.item(), 'penalty': penalty_value.item()}