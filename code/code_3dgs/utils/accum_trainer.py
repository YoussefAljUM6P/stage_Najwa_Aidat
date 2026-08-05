# -*- coding: utf-8 -*-
import torch


class AccumTrainer(object):
    def __init__(self,
                 device="cpu",
                 name="AccumTrainer"):
        self._name   = name
        self._device = device

    def prepare_minimize(self, loss, var_list):
        """
        BUG FIX: loss est passé à None depuis training_thread car la loss
        n'existe pas encore au moment de l'init. C'est correct : on stocke
        seulement var_list ici, la loss est passée à accumulate_gradients().
        """
        self._var_list        = var_list
        self._accum_grad_list = [torch.zeros_like(var) for var in var_list]

    def get_accum_grad_list(self):
        return self._accum_grad_list

    def accumulate_gradients(self, loss):
        grads = torch.autograd.grad(
            loss,
            self._var_list,
            retain_graph=True,
            allow_unused=True
        )
        for accum_grad, grad in zip(self._accum_grad_list, grads):
            if grad is not None:
                accum_grad.add_(grad.detach())

    def reset_gradients(self):
        for accum_grad in self._accum_grad_list:
            accum_grad.zero_()
