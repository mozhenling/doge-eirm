import torch
import torch.nn.functional as F
from algorithms.classes.ERM import ERM
from algorithms.optimization import get_optimizer

class IB_ERM(ERM):
    """Information Bottleneck based ERM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(IB_ERM, self).__init__(input_shape, num_classes, num_domains, hparams, args)

        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None, doyojo=None):

        ib_penalty_weight = (self.hparams['ib_penalty_weight'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)

        nll = 0.
        ib_penalty = 0.
        all_x = torch.cat([x for x, y in minibatches])
        # ------------------------------------------------------------------------
        # ------------------------- DoYoJo subalgorithm --------------------------
        all_features = self.featurizer(all_x) if doyojo is None else doyojo.subalg_all_featurizer_outs(all_x)
        all_logits = self.classifier(all_features) if doyojo is None else doyojo.subalg_all_classifier_outs(all_features)
        # ------------------------------------------------------------------------

        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += ib_penalty_weight * ib_penalty

        # ------------------------------------------------------------------------
        # ------------------------- DoYoJo subalgorithm --------------------------
        if doyojo is not None:
            doyojo.hparams['ib_penalty_weight'] = ib_penalty_weight
            if self.update_count == self.hparams['ib_penalty_anneal_iters']:
                # Reset Adam, because it doesn't like the sharp jump in gradient
                # magnitudes that happens at this step.
                doyojo.optimizer = get_optimizer(params=doyojo.parameters(),
                                                 hparams=doyojo.hparams, args=doyojo.args)

            return {'erm_alpha': nll, 'ib_penalty': ib_penalty}
        # ------------------------------------------------------------------------
        else:
            if self.update_count == self.hparams['ib_penalty_anneal_iters']:
                # Reset Adam, because it doesn't like the sharp jump in gradient
                # magnitudes that happens at this step.
                self.optimizer = get_optimizer(params =self.network.parameters(), hparams=self.hparams, args=self.args)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.args.scheduler:
                self.scheduler.step()

            self.update_count += 1
            return {'loss': loss.item(),
                    'nll': nll.item(),
                    'IB_penalty': ib_penalty.item()}