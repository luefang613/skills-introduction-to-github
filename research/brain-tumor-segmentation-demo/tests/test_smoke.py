"""Deterministic CPU smoke tests; do not test any clinical/real dataset claims."""
import unittest
import torch
from segmentation_demo import SyntheticSegmentation, TinyUNet, segmentation_metrics, run_demo


class SmokeTests(unittest.TestCase):
    def test_data_shapes_and_labels(self):
        x,y = SyntheticSegmentation(count=2)[0]
        self.assertEqual(tuple(x.shape),(1,32,32))
        self.assertEqual(tuple(y.shape),(32,32))
        self.assertTrue(set(y.unique().tolist()).issubset({0,1,2}))

    def test_forward_backward(self):
        model=TinyUNet()
        x=torch.randn(2,1,32,32)
        y=torch.randint(0,3,(2,32,32))
        out=model(x)
        self.assertEqual(tuple(out.shape),(2,3,32,32))
        loss=torch.nn.functional.cross_entropy(out,y)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        scores=segmentation_metrics(out,y)
        self.assertTrue(all(0 <= v <= 1 for v in scores.values()))

    def test_training_loop(self):
        out=run_demo(epochs=1)
        self.assertTrue(0 < out['loss'] < 10)


if __name__=='__main__': unittest.main()
