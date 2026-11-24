"""La penalización adicional debe corregir bordes y admitir fondos vacíos."""
import unittest

import torch

from scripts.train_detector_refine import complete_object_loss
from training.detector_own import detection_loss, make_targets


class RefinementLossTests(unittest.TestCase):
    def test_extent_gradient_expands_partial_boxes_and_shrinks_oversized_boxes(self):
        config={'negative_quality_weight':0}
        for size_logit,expected_sign in [(0.,-1),(2.,1)]:
            outputs=[torch.zeros(1,6,n,n,requires_grad=True) for n in (16,8,4)]
            with torch.no_grad():
                for o in outputs:o[:,3:5].fill_(size_logit)
            targets,collisions=make_targets(outputs,[torch.tensor([[40.,40.,104.,104.]])])
            original,_=detection_loss(outputs,prepared_targets=(targets,collisions))
            refined,_=complete_object_loss(outputs,targets,collisions,config)
            extra=refined-original
            extra.backward()
            gradient=outputs[1].grad[0,3:5,4,4]
            self.assertTrue(torch.isfinite(extra))
            self.assertTrue(torch.all(gradient*expected_sign>0))

    def test_empty_background_has_finite_loss_without_extent_penalty(self):
        outputs=[torch.zeros(2,6,n,n,requires_grad=True) for n in (16,8,4)]
        targets,collisions=make_targets(outputs,[torch.empty(0,4),torch.empty(0,4)])
        original,_=detection_loss(outputs,prepared_targets=(targets,collisions))
        refined,_=complete_object_loss(outputs,targets,collisions,{'negative_quality_weight':0})
        self.assertEqual(float(original.detach()),float(refined.detach()))
        refined.backward()
        self.assertTrue(all(torch.isfinite(o.grad).all() for o in outputs))


if __name__=='__main__':unittest.main()
