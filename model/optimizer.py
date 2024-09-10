import torch
import numpy as np

import math
import torch
from torch.optim import Optimizer

#taken from https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch/tree/main

class AdEMAMix(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9999), eps=1e-8,
                 weight_decay=0, alpha=5.0, T_alpha_beta3=None):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        assert len(betas) == 3, f"Invalid beta parameters: {betas}, expected 3"
        assert all(0.0 <= beta < 1.0 for beta in betas), f"Invalid beta parameters: {betas}"
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        alpha=alpha, T_alpha_beta3=T_alpha_beta3)
        super(AdEMAMix, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdEMAMix, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_slow = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('AdEMAMix does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Slow exponential moving average
                        state['exp_avg_slow'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    exp_avg_slow.append(state['exp_avg_slow'])
                    state['step'] += 1
                    state_steps.append(state['step'])

            beta1, beta2, beta3 = group['betas']
            alpha = group['alpha']
            T_alpha_beta3 = group['T_alpha_beta3']

            self._update_adamemix(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                exp_avg_slow,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                alpha=alpha,
                T_alpha_beta3=T_alpha_beta3,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
            )

        return loss

    def _update_adamemix(self, params, grads, exp_avgs, exp_avg_sqs, exp_avg_slow, state_steps,
                         beta1, beta2, beta3, alpha, T_alpha_beta3, lr, weight_decay, eps):

        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            exp_avg_slow_i = exp_avg_slow[i]
            step = state_steps[i]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            if T_alpha_beta3 is not None:
                alpha_t = min(step * alpha / T_alpha_beta3, alpha)
                beta3_t = min(math.exp(math.log(beta1) * math.log(beta3) /
                                       ((1 - step / T_alpha_beta3) * math.log(beta3) +
                                        (step / T_alpha_beta3) * math.log(beta1))), beta3)
            else:
                alpha_t = alpha
                beta3_t = beta3

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            exp_avg_slow_i.mul_(beta3_t).add_(grad, alpha=1 - beta3_t)

            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1

            if weight_decay != 0:
                param.add_(param, alpha=-weight_decay * lr)

            param.addcdiv_(exp_avg + alpha_t * exp_avg_slow_i, denom, value=-step_size)


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, train_config, model_config, current_step):

        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        self.n_warmup_steps = train_config["optimizer"]["warm_up_step"]
        self.anneal_steps = train_config["optimizer"]["anneal_steps"]
        self.anneal_rate = train_config["optimizer"]["anneal_rate"]
        self.current_step = current_step
        self.init_lr = np.power(model_config["transformer"]["encoder_hidden"], -0.5)

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    def step(self, *args, **kwargs):
        return self.step_and_update_lr()

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
