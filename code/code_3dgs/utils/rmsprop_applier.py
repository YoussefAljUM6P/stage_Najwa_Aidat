# -*- coding: utf-8 -*-
import torch
import threading


class RMSPropApplier(object):

    def __init__(self,
                 learning_rate,
                 decay=0.9,
                 momentum=0.0,
                 epsilon=1e-10,
                 clip_norm=40.0,
                 device="cpu",
                 name="RMSPropApplier"):

        self._name          = name
        self._learning_rate = learning_rate
        self._decay         = decay
        self._momentum      = momentum
        self._epsilon       = epsilon
        self._clip_norm     = clip_norm
        self._device        = device
        self._optimizer     = None
        # BUG FIX: plusieurs threads appellent apply_gradients simultanément
        # => verrou pour protéger l'optimiseur partagé
        self._lock          = threading.Lock()

    def apply_gradients(self, var_list, accum_grad_list, lr_override=None):
        """
        BUG FIX 1: le paramètre lr_override était absent alors que
        training_thread l'utilise pour l'annealing du learning rate.

        BUG FIX 2: verrou threading pour éviter les race conditions quand
        plusieurs threads appellent apply_gradients en parallèle.
        """
        lr = lr_override if lr_override is not None else self._learning_rate

        with self._lock:
            # Créer l'optimiseur à la première utilisation
            if self._optimizer is None:
                self._optimizer = torch.optim.RMSprop(
                    var_list,
                    lr=lr,
                    alpha=self._decay,
                    momentum=self._momentum,
                    eps=self._epsilon
                )
            else:
                # Mettre à jour le learning rate annealed
                for pg in self._optimizer.param_groups:
                    pg['lr'] = lr

            # Remettre les gradients à zéro
            self._optimizer.zero_grad()

            # Clipping et assignation des gradients accumulés
            for var, accum_grad in zip(var_list, accum_grad_list):
                if accum_grad is not None:
                    clipped = torch.clamp(accum_grad, -self._clip_norm, self._clip_norm)
                    var.grad = clipped.clone()

            # Mise à jour des poids
            self._optimizer.step()
