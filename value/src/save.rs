use std::io::Write;

use tch::nn;

use crate::arch::{DK, DV, INPUTS, PIECE_TOKENS, TOKENS};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Accumulator<const N: usize>(pub [f32; N]);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Layer<const M: usize, const N: usize> {
    pub weights: [Accumulator<N>; M],
    pub biases: Accumulator<N>,
}

#[repr(C)]
pub struct SavedNetworkFormat {
    pub wq: [[Accumulator<{DK as usize}>; INPUTS as usize]; PIECE_TOKENS as usize],
    pub wk: [[Accumulator<{DK as usize}>; INPUTS as usize]; PIECE_TOKENS as usize],
    pub wv: [[Accumulator<{DV as usize}>; INPUTS as usize]; PIECE_TOKENS as usize],
    pub wq_board: [Accumulator<{DK as usize}>; 768],
    pub wk_board: [Accumulator<{DK as usize}>; 768],
    pub wv_board: [Accumulator<{DV as usize}>; 768],
    pub l1: Layer<{(TOKENS * DV) as usize}, 16>,
    pub l2: Layer<16, 1>,
}

impl SavedNetworkFormat {
    pub fn boxed_and_zeroed() -> Box<Self> {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        }
    }

    pub fn write_to_bin(&self, path: &str) {
        let size_of = std::mem::size_of::<Self>();

        let mut file = std::fs::File::create(path).unwrap();

        unsafe {
            let ptr: *const Self = self;
            let slice_ptr: *const u8 = std::mem::transmute(ptr);
            let slice = std::slice::from_raw_parts(slice_ptr, size_of);
            file.write_all(slice).unwrap();
        }
    }

    pub fn write_linear_into_matrix<const M: usize, const N: usize>(linear: &nn::Linear, out: &mut [Accumulator<N>; M]) {
        let contiguous = tch::no_grad(|| linear.ws.reshape([(M * N) as i64]));

        let mut buffer = vec![0f32; M * N];
        contiguous.copy_data(&mut buffer, M * N);

        for i in 0..M {
            for j in 0..N {
                out[i].0[j] = buffer[M * j + i];
            }
        }
    }

    pub fn write_linear_into_layer<const M: usize, const N: usize>(linear: &nn::Linear, layer: &mut Layer<M, N>) {
        Self::write_linear_into_matrix(linear, &mut layer.weights);
        linear.bs.as_ref().unwrap().copy_data(&mut layer.biases.0, N);
    }
}
