use sundials_sys::{
    SUNContext, SUNDenseMatrix, SUNDenseMatrix_Columns, SUNDenseMatrix_Data, SUNDenseMatrix_LData,
    SUNDenseMatrix_Rows, SUNMatDestroy, SUNMatGetID, SUNMatrix, SUNMatrix_ID_SUNMATRIX_DENSE,
};

use crate::SunContext;

/// An owned SUNDIALS dense matrix.
pub struct DenseMatrix {
    ptr: SUNMatrix,
    rows: usize,
    columns: usize,
    ctx: SUNContext,
}

impl DenseMatrix {
    /// Allocate a column-major dense matrix in `ctx`.
    ///
    /// # Safety
    /// `ctx` must outlive this matrix and any solver that owns it.
    ///
    /// # Errors
    /// Returns an error for invalid dimensions or allocation failure.
    pub unsafe fn new(
        rows: usize,
        columns: usize,
        ctx: &SunContext,
    ) -> Result<Self, DenseMatrixError> {
        if rows == 0 || columns == 0 {
            return Err(DenseMatrixError::InvalidDimensions);
        }
        let elements = rows
            .checked_mul(columns)
            .ok_or(DenseMatrixError::InvalidDimensions)?;
        i64::try_from(elements).map_err(|_| DenseMatrixError::InvalidDimensions)?;
        let rows_c = i64::try_from(rows).map_err(|_| DenseMatrixError::InvalidDimensions)?;
        let columns_c = i64::try_from(columns).map_err(|_| DenseMatrixError::InvalidDimensions)?;
        let ptr = unsafe { SUNDenseMatrix(rows_c, columns_c, ctx.as_raw()) };
        if ptr.is_null() {
            return Err(DenseMatrixError::AllocationFailed);
        }
        Ok(Self {
            ptr,
            rows,
            columns,
            ctx: ctx.as_raw(),
        })
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub fn columns(&self) -> usize {
        self.columns
    }

    #[must_use]
    pub fn as_raw(&self) -> SUNMatrix {
        self.ptr
    }

    pub fn view_mut(&mut self) -> DenseMatrixViewMut<'_> {
        // The matrix is owned and exclusively borrowed for this view.
        unsafe { DenseMatrixViewMut::from_raw(self.ptr) }
            .expect("an owned DenseMatrix must remain a valid dense matrix")
    }

    pub(crate) fn context_raw(&self) -> SUNContext {
        self.ctx
    }
}

impl Drop for DenseMatrix {
    fn drop(&mut self) {
        unsafe { SUNMatDestroy(self.ptr) };
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseMatrixError {
    InvalidDimensions,
    AllocationFailed,
}

impl std::fmt::Display for DenseMatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for DenseMatrixError {}

/// A checked, non-owning view of a SUNDIALS dense matrix.
///
/// Entries are stored in column-major order. This view is intended for use
/// inside a SUNDIALS Jacobian callback, where SUNDIALS owns the matrix.
pub struct DenseMatrixViewMut<'a> {
    rows: usize,
    columns: usize,
    data: &'a mut [f64],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseMatrixViewError {
    NullMatrix,
    NotDense,
    InvalidDimensions,
    NullData,
}

impl std::fmt::Display for DenseMatrixViewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NullMatrix => write!(f, "null SUNMatrix"),
            Self::NotDense => write!(f, "SUNMatrix is not a dense matrix"),
            Self::InvalidDimensions => write!(f, "invalid dense SUNMatrix dimensions"),
            Self::NullData => write!(f, "dense SUNMatrix has null data"),
        }
    }
}

impl std::error::Error for DenseMatrixViewError {}

impl<'a> DenseMatrixViewMut<'a> {
    /// Borrow a SUNDIALS-owned dense matrix from a callback pointer.
    ///
    /// # Safety
    /// `raw` must point to a live `SUNMatrix` for the entire lifetime `'a`.
    /// The caller must have exclusive access to its entries during that time;
    /// in particular, no other Rust or C code may access or resize the matrix
    /// while this view exists.
    ///
    /// # Errors
    /// Returns an error if the pointer is null, the matrix is not dense, or
    /// its dimensions and data pointer are inconsistent.
    pub unsafe fn from_raw(raw: SUNMatrix) -> Result<Self, DenseMatrixViewError> {
        if raw.is_null() {
            return Err(DenseMatrixViewError::NullMatrix);
        }
        if unsafe { SUNMatGetID(raw) } != SUNMatrix_ID_SUNMATRIX_DENSE {
            return Err(DenseMatrixViewError::NotDense);
        }

        let rows = usize::try_from(unsafe { SUNDenseMatrix_Rows(raw) })
            .map_err(|_| DenseMatrixViewError::InvalidDimensions)?;
        let columns = usize::try_from(unsafe { SUNDenseMatrix_Columns(raw) })
            .map_err(|_| DenseMatrixViewError::InvalidDimensions)?;
        let len = rows
            .checked_mul(columns)
            .ok_or(DenseMatrixViewError::InvalidDimensions)?;
        let allocated_len = usize::try_from(unsafe { SUNDenseMatrix_LData(raw) })
            .map_err(|_| DenseMatrixViewError::InvalidDimensions)?;
        if len != allocated_len {
            return Err(DenseMatrixViewError::InvalidDimensions);
        }
        let ptr = unsafe { SUNDenseMatrix_Data(raw) };
        if ptr.is_null() {
            return Err(DenseMatrixViewError::NullData);
        }

        Ok(Self {
            rows,
            columns,
            data: unsafe { std::slice::from_raw_parts_mut(ptr, len) },
        })
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub fn columns(&self) -> usize {
        self.columns
    }

    #[must_use]
    pub fn data(&self) -> &[f64] {
        self.data
    }

    pub fn data_mut(&mut self) -> &mut [f64] {
        self.data
    }

    #[must_use]
    pub fn get(&self, row: usize, column: usize) -> Option<&f64> {
        if row < self.rows && column < self.columns {
            self.data.get(column * self.rows + row)
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, row: usize, column: usize) -> Option<&mut f64> {
        if row < self.rows && column < self.columns {
            self.data.get_mut(column * self.rows + row)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use sundials_sys::{SUNDenseMatrix, SUNMatDestroy};

    use super::*;
    use crate::SunContext;

    #[test]
    fn dense_matrix_view_checks_shape_and_column_major_entries() {
        let ctx = SunContext::new();
        let raw = unsafe { SUNDenseMatrix(2, 3, ctx.as_raw()) };
        assert!(!raw.is_null());

        {
            let mut view = unsafe { DenseMatrixViewMut::from_raw(raw) }.unwrap();
            assert_eq!((view.rows(), view.columns()), (2, 3));
            *view.get_mut(1, 2).unwrap() = 7.0;
            assert_eq!(view.data()[5], 7.0);
            assert_eq!(view.get(1, 2), Some(&7.0));
            assert!(view.get_mut(2, 0).is_none());
            assert!(view.get_mut(0, 3).is_none());
        }

        unsafe { SUNMatDestroy(raw) };
        assert!(matches!(
            unsafe { DenseMatrixViewMut::from_raw(std::ptr::null_mut()) },
            Err(DenseMatrixViewError::NullMatrix)
        ));
    }
}
