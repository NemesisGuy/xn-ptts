//! Deep copies of streaming state, for forking a stream (`asr::AsrStream::fork`).
//!
//! `#[derive(Clone)]` on a state is not a copy: an `xn::Tensor` is an `Arc` around its
//! buffer, so a cloned state shares every cache with the original, and a cache updated in
//! place (`slice_set`, as the flow LM's KV cache is) would write into both. [`DeepCopy`]
//! allocates fresh storage for every tensor instead, so the two states share nothing,
//! whatever the forward passes later do to them.

use xn::{Backend, Result, Tensor, WithDType};

pub(crate) trait DeepCopy: Sized {
    /// A copy that shares no tensor storage with `self`.
    fn deep_copy(&self) -> Result<Self>;
}

impl<T: WithDType, B: Backend> DeepCopy for Tensor<T, B> {
    fn deep_copy(&self) -> Result<Self> {
        self.copy()
    }
}

impl<X: DeepCopy> DeepCopy for Option<X> {
    fn deep_copy(&self) -> Result<Self> {
        self.as_ref().map(X::deep_copy).transpose()
    }
}

impl<X: DeepCopy> DeepCopy for Vec<X> {
    fn deep_copy(&self) -> Result<Self> {
        self.iter().map(X::deep_copy).collect()
    }
}

impl<X: DeepCopy, Y: DeepCopy> DeepCopy for (X, Y) {
    fn deep_copy(&self) -> Result<Self> {
        Ok((self.0.deep_copy()?, self.1.deep_copy()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xn::CpuDevice;

    #[test]
    fn a_copied_tensor_does_not_share_storage() {
        let a = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 4), &CpuDevice).unwrap();
        let shallow = a.clone();
        let deep = a.deep_copy().unwrap();
        let poison = Tensor::from_vec(vec![9f32], (1, 1), &CpuDevice).unwrap();
        deep.slice_set(&poison, 1usize, 0).unwrap();
        assert_eq!(a.to_vec().unwrap(), [1., 2., 3., 4.]);
        assert_eq!(deep.to_vec().unwrap(), [9., 2., 3., 4.]);
        // What `Clone` would have done: the write reaches the original.
        shallow.slice_set(&poison, 1usize, 1).unwrap();
        assert_eq!(a.to_vec().unwrap(), [1., 9., 3., 4.]);
    }

    #[test]
    fn a_copied_kv_cache_updated_in_place_leaves_the_source_alone() {
        use crate::transformer::{
            LayerAttentionState, StreamingMHAState, StreamingTransformerState,
        };
        let zeros = || Tensor::<f32, CpuDevice>::zeros((1, 4, 1, 2), &CpuDevice).unwrap();
        let src = StreamingTransformerState {
            layer_states: vec![LayerAttentionState::FlowLm(StreamingMHAState {
                k_cache: zeros(),
                v_cache: zeros(),
                current_end: 1,
            })],
        };
        let mut copy = src.deep_copy().unwrap();
        let LayerAttentionState::FlowLm(mha) = &mut copy.layer_states[0] else { panic!() };
        let kv = Tensor::from_vec(vec![5f32, 6.], (1, 1, 1, 2), &CpuDevice).unwrap();
        mha.complete_kv(&kv, &kv).unwrap();
        assert_eq!(mha.k_cache.to_vec().unwrap()[2..4], [5., 6.]);
        let LayerAttentionState::FlowLm(src) = &src.layer_states[0] else { panic!() };
        assert_eq!(src.current_end, 1);
        assert!(src.k_cache.to_vec().unwrap().iter().all(|&x| x == 0.0));
        assert!(src.v_cache.to_vec().unwrap().iter().all(|&x| x == 0.0));
    }
}
