use sundials_sys::{
    N_VGetArrayPointer_Serial, N_VGetLength_Serial, N_VGetVectorID, N_Vector,
    N_Vector_ID_SUNDIALS_NVEC_SERIAL,
};

/// Read-only view of a serial vector passed to a SUNDIALS callback.
pub struct SerialVectorView<'a> {
    data: &'a [f64],
}

/// Mutable view of a serial vector passed to a SUNDIALS callback.
pub struct SerialVectorViewMut<'a> {
    data: &'a mut [f64],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SerialVectorViewError {
    NullVector,
    NotSerial,
    InvalidLength,
    NullData,
}

impl std::fmt::Display for SerialVectorViewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SerialVectorViewError {}

unsafe fn checked_parts(raw: N_Vector) -> Result<(*mut f64, usize), SerialVectorViewError> {
    if raw.is_null() {
        return Err(SerialVectorViewError::NullVector);
    }
    if unsafe { N_VGetVectorID(raw) } != N_Vector_ID_SUNDIALS_NVEC_SERIAL {
        return Err(SerialVectorViewError::NotSerial);
    }
    let len = usize::try_from(unsafe { N_VGetLength_Serial(raw) })
        .map_err(|_| SerialVectorViewError::InvalidLength)?;
    let data = unsafe { N_VGetArrayPointer_Serial(raw) };
    if data.is_null() {
        return Err(SerialVectorViewError::NullData);
    }
    Ok((data, len))
}

impl<'a> SerialVectorView<'a> {
    /// Borrow a callback vector as a read-only slice.
    ///
    /// # Safety
    /// `raw` must be a live serial vector for the lifetime of this view. No
    /// code may mutate or free its data while this view exists.
    ///
    /// # Errors
    /// Returns an error for a null, non-serial, or invalid vector.
    pub unsafe fn from_raw(raw: N_Vector) -> Result<Self, SerialVectorViewError> {
        let (data, len) = unsafe { checked_parts(raw) }?;
        Ok(Self {
            data: unsafe { std::slice::from_raw_parts(data, len) },
        })
    }

    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        self.data
    }
}

impl<'a> SerialVectorViewMut<'a> {
    /// Borrow a callback vector as a mutable slice.
    ///
    /// # Safety
    /// `raw` must be a live serial vector for the lifetime of this view. The
    /// caller must have exclusive access to its data during that time.
    ///
    /// # Errors
    /// Returns an error for a null, non-serial, or invalid vector.
    pub unsafe fn from_raw(raw: N_Vector) -> Result<Self, SerialVectorViewError> {
        let (data, len) = unsafe { checked_parts(raw) }?;
        Ok(Self {
            data: unsafe { std::slice::from_raw_parts_mut(data, len) },
        })
    }

    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NVector, SunContext};

    #[test]
    fn checked_serial_views_expose_slices() {
        let ctx = SunContext::new();
        let vector = unsafe { NVector::new_serial(2, &ctx) }.unwrap();
        {
            let mut view = unsafe { SerialVectorViewMut::from_raw(vector.as_raw()) }.unwrap();
            view.as_mut_slice().copy_from_slice(&[2.0, 3.0]);
        }
        let view = unsafe { SerialVectorView::from_raw(vector.as_raw()) }.unwrap();
        assert_eq!(view.as_slice(), &[2.0, 3.0]);
        assert!(matches!(
            unsafe { SerialVectorView::from_raw(std::ptr::null_mut()) },
            Err(SerialVectorViewError::NullVector)
        ));
    }
}
