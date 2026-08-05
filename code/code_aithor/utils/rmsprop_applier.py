# -*- coding: utf-8 -*-
import torch
import threading


class RMSPropApplier(object):
    """
    Applique les gradients accumulés sur une liste fixe de paramètres
    via RMSProp. L'optimizer est créé une seule fois sur var_list.

    IMPORTANT : var_list doit être passé à la construction (paramètres du
    réseau global). apply_gradients reçoit un sous-ensemble de ces paramètres
    avec leurs gradients correspondants — il injecte les gradients puis appelle
    optimizer.step() sur l'ensemble (les params sans gradient ne bougent pas).
    """

    def __init__(self,
                 var_list,
                 learning_rate=7e-4,
                 decay=0.99,
                 momentum=0.0,
                 epsilon=1e-10,
                 clip_norm=40.0,
                 device=None):

        self._learning_rate = learning_rate
        self._clip_norm     = clip_norm
        self._lock          = threading.Lock()

        self._optimizer = torch.optim.RMSprop(
            var_list,
            lr=learning_rate,
            alpha=decay,
            momentum=momentum,
            eps=epsilon
        )

    def apply_gradients(self, param_list, accum_grad_list, lr_override=None):
        """
        param_list      : sous-liste des paramètres globaux à mettre à jour
        accum_grad_list : gradients accumulés correspondants (même ordre)
        lr_override     : taux d'apprentissage annealed (optionnel)
        """
        with self._lock:
            # Mise à jour du learning rate si nécessaire
            if lr_override is not None:
                for pg in self._optimizer.param_groups:
                    pg['lr'] = lr_override

            # Remettre tous les gradients à zéro
            self._optimizer.zero_grad()

            # Injecter les gradients clippés dans les paramètres concernés
            for param, accum_grad in zip(param_list, accum_grad_list):
                if accum_grad is None:
                    continue
                clipped = torch.clamp(accum_grad, -self._clip_norm, self._clip_norm)
                # Vérification de cohérence (ne devrait plus arriver avec le fix)
                if param.shape != clipped.shape:
                    print(
                        f"[RMSProp] AVERTISSEMENT shape mismatch : "
                        f"param={param.shape} grad={clipped.shape} — ignoré",
                        flush=True
                    )
                    continue
                param.grad = clipped.clone().detach()

            # Un seul step sur l'optimizer global
            self._optimizer.step()