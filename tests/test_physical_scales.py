"""Physical losses and residual increments must not depend on input units."""
import io
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ball_retention.world_model import ResidualMLP
from ball_retention.pair_loss import paired_loss


class PhysicalScaleTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(491)
        self.units=np.repeat([.005,.05,2.5],3)
        self.gain=self.units*np.repeat([.05,.25,.25],3)
        self.a=ResidualMLP(2,np.zeros(9),self.units,1/60,41,hidden=5,residual_gain=self.gain/self.units)
        scale=self.units*np.linspace(.4,3,9)
        mean=self.units*np.linspace(-2,2,9)
        self.b=ResidualMLP(2,mean,scale,1/60,41,hidden=5,residual_gain=self.gain/scale)
        # Reparameterize only the first layer to represent the same physical
        # network under different affine input coordinates.
        for k,v in self.a.params.items(): self.b.params[k][...]=v
        self.b.params['W1'][:9]=scale[:,None]/self.a.state_scale[:,None]*self.a.params['W1'][:9]
        self.b.params['b1'][...]=mean/self.a.state_scale@self.a.params['W1'][:9]
        output=self.rng.normal(scale=.1,size=self.a.params['W3'].shape)
        self.a.params['W3'][...]=output;self.b.params['W3'][...]=output
        self.x=self.rng.normal(size=(2,9))*self.units
        self.k=self.rng.normal(size=(2,4,2))
        self.y=self.rng.normal(size=(2,4,9))*self.units

    def test_physical_rollout_loss_and_reload_invariant_to_affine_input_units(self):
        predictions=[];losses=[]
        for model in (self.a,self.b):
            predictions.append(model.denormalize(model.rollout(model.normalize(self.x),self.k)))
            losses.append(model.loss_and_grad(model.normalize(self.x),self.k,model.normalize(self.y),
                state_weights=np.arange(1,10),state_error_scale_si=self.units)[0])
            np.testing.assert_allclose(model.residual_gain*model.state_scale,self.gain,rtol=1e-15)
            stream=io.BytesIO();np.savez(stream,**model.state_dict());stream.seek(0)
            with np.load(stream,allow_pickle=False) as saved: restored=ResidualMLP.from_state_dict(saved)
            np.testing.assert_array_equal(restored.rollout(model.normalize(self.x),self.k),model.rollout(model.normalize(self.x),self.k))
        np.testing.assert_allclose(predictions[0],predictions[1],rtol=1e-12,atol=1e-14)
        self.assertAlmostEqual(losses[0],losses[1],places=12)

    def test_anchor_parameter_gradient_with_physical_units(self):
        model=self.b;x=model.normalize(self.x);y=model.normalize(self.y)
        def objective():return model.loss_and_grad(x,self.k,y,state_error_scale_si=self.units)
        _,grads=objective()
        for key,index in [('W1',(1,2)),('W2',(2,3)),('W3',(3,4)),('b3',(0,))]:
            value=model.params[key][index];eps=1e-6
            model.params[key][index]=value+eps;plus=objective()[0]
            model.params[key][index]=value-eps;minus=objective()[0]
            model.params[key][index]=value
            self.assertAlmostEqual(grads[key][index],(plus-minus)/(2*eps),delta=2e-8)

    def test_paired_physical_loss_invariance_and_masked_gradient(self):
        known=np.stack((self.k,self.k+.1));target=np.stack((self.y,self.y+.1*self.units))
        mask=np.ones((2,2,4),bool);mask[1,0,-1]=False
        response=np.ones((2,4,9),bool)
        def objective(model,gradients=True):
            return paired_loss(model,model.normalize(self.x),known,target,mask,response,mask.all(0),
                np.zeros((2,4,9)),np.array([0,.2,.3,.5]),np.arange(1,10),self.units,.5,.2,
                absolute_scale_si=self.units,sample_weights=np.array([.3,.7]),gradients=gradients)
        first,_=objective(self.a);second,grads=objective(self.b)
        for key in first:self.assertAlmostEqual(first[key],second[key],places=12)
        for key,index in [('W1',(1,2)),('W3',(3,4)),('b3',(0,))]:
            value=self.b.params[key][index];eps=1e-6
            self.b.params[key][index]=value+eps;plus=objective(self.b,False)[0]['loss']
            self.b.params[key][index]=value-eps;minus=objective(self.b,False)[0]['loss']
            self.b.params[key][index]=value
            self.assertAlmostEqual(grads[key][index],(plus-minus)/(2*eps),delta=2e-8)

    def test_invalid_scales_and_legacy_default(self):
        model=self.a;x=model.normalize(self.x);y=model.normalize(self.y)
        legacy=model.loss_and_grad(x,self.k,y)
        equivalent=model.loss_and_grad(x,self.k,y,state_error_scale_si=model.state_scale)
        self.assertEqual(legacy[0],equivalent[0])
        for key in legacy[1]:np.testing.assert_array_equal(legacy[1][key],equivalent[1][key])
        for scales in (np.zeros(9),np.full(9,np.nan),np.ones(8)):
            with self.assertRaises(ValueError): model.state_error_factor(scales)
