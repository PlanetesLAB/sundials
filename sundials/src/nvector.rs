use sundials_sys::{
    N_VClone, N_VDestroy_Serial, N_VGetArrayPointer_Serial, N_VNew_Serial, N_VScale, N_Vector,
    SUNContext,
};

use crate::SunContext;

pub struct NVector {
    ptr: N_Vector,
    len: usize,
    ctx: SUNContext,
}

impl NVector {
    /// Create a new serial [`NVector`] of the given length, associated with the provided [`SunContext`].
    ///
    /// # Safety
    /// `ctx` must outlive the vector and all of its clones.
    ///
    /// # Errors
    /// Returns an error for zero or unrepresentable length, or allocation failure.
    pub unsafe fn new_serial(len: usize, ctx: &SunContext) -> Result<Self, NVectorError> {
        if len == 0 {
            return Err(NVectorError::InvalidLength);
        }
        let len_c = i64::try_from(len).map_err(|_| NVectorError::InvalidLength)?;
        let ptr = unsafe { N_VNew_Serial(len_c, ctx.as_raw()) };
        if ptr.is_null() {
            return Err(NVectorError::AllocationFailed);
        }
        Ok(Self {
            ptr,
            len,
            ctx: ctx.as_raw(),
        })
    }

    #[must_use]
    #[inline]
    pub fn as_raw(&self) -> N_Vector {
        self.ptr
    }

    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn context_raw(&self) -> SUNContext {
        self.ctx
    }

    #[must_use]
    #[inline]
    pub fn slice(&self) -> &[f64] {
        let p = unsafe { N_VGetArrayPointer_Serial(self.ptr) };
        unsafe { std::slice::from_raw_parts(p, self.len) }
    }

    #[inline]
    pub fn slice_mut(&mut self) -> &mut [f64] {
        let p = unsafe { N_VGetArrayPointer_Serial(self.ptr) };
        unsafe { std::slice::from_raw_parts_mut(p, self.len) }
    }
}

impl Drop for NVector {
    #[inline]
    fn drop(&mut self) {
        unsafe { N_VDestroy_Serial(self.ptr) };
    }
}

impl Clone for NVector {
    fn clone(&self) -> Self {
        unsafe {
            let new = N_VClone(self.ptr);
            assert!(!new.is_null(), "SUNDIALS failed to clone NVector");
            N_VScale(1.0, self.ptr, new);

            Self {
                ptr: new,
                len: self.len,
                ctx: self.ctx,
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NVectorError {
    InvalidLength,
    AllocationFailed,
}

impl std::fmt::Display for NVectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for NVectorError {}
