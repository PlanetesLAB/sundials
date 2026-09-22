use sundials_sys::{
    SUNContext, SUNGramSchmidtType_SUN_CLASSICAL_GS, SUNGramSchmidtType_SUN_MODIFIED_GS,
    SUNLinSol_Dense, SUNLinSol_SPBCGS, SUNLinSol_SPFGMR, SUNLinSol_SPGMR, SUNLinSol_SPGMRSetGSType,
    SUNLinSol_SPGMRSetMaxRestarts, SUNLinSol_SPTFQMR, SUNLinSolFree, SUNLinearSolver,
    SUNPrecType_SUN_PREC_BOTH, SUNPrecType_SUN_PREC_LEFT, SUNPrecType_SUN_PREC_NONE,
    SUNPrecType_SUN_PREC_RIGHT,
};

use crate::{DenseMatrix, NVector};

pub struct LinearSolver {
    ptr: SUNLinearSolver,
    ctx: SUNContext,
    matrix: Option<DenseMatrix>,
    dimension: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PreconditionerType {
    None,
    #[default]
    Left,
    Right,
    Both,
}

impl PreconditionerType {
    #[inline]
    fn as_raw(self) -> i32 {
        match self {
            Self::None => SUNPrecType_SUN_PREC_NONE.try_into().unwrap(),
            Self::Left => SUNPrecType_SUN_PREC_LEFT.try_into().unwrap(),
            Self::Right => SUNPrecType_SUN_PREC_RIGHT.try_into().unwrap(),
            Self::Both => SUNPrecType_SUN_PREC_BOTH.try_into().unwrap(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GramSchmidtType {
    #[default]
    Modified,
    Classical,
}

impl GramSchmidtType {
    #[inline]
    fn as_raw(self) -> i32 {
        match self {
            Self::Modified => SUNGramSchmidtType_SUN_MODIFIED_GS.try_into().unwrap(),
            Self::Classical => SUNGramSchmidtType_SUN_CLASSICAL_GS.try_into().unwrap(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpgmrOptions {
    pub preconditioner: PreconditionerType,
    /// Maximum Krylov subspace dimension. Zero selects the SUNDIALS default.
    pub max_krylov_dimension: usize,
    pub max_restarts: usize,
    pub gram_schmidt: GramSchmidtType,
}

impl Default for SpgmrOptions {
    fn default() -> Self {
        Self {
            preconditioner: PreconditionerType::Left,
            max_krylov_dimension: 0,
            max_restarts: 0,
            gram_schmidt: GramSchmidtType::Modified,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinearSolverError {
    ValueOutOfRange(&'static str),
    CreationFailed,
    ConfigurationFailed(i32),
    ContextMismatch,
    MatrixShapeMismatch,
}

impl std::fmt::Display for LinearSolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ValueOutOfRange(name) => write!(f, "{name} does not fit in a C integer"),
            Self::CreationFailed => write!(f, "SUNDIALS failed to create the linear solver"),
            Self::ConfigurationFailed(code) => {
                write!(
                    f,
                    "SUNDIALS rejected the linear-solver configuration ({code})"
                )
            }
            Self::ContextMismatch => write!(f, "linear solver objects use different contexts"),
            Self::MatrixShapeMismatch => write!(f, "dense matrix is not square with the vector"),
        }
    }
}

impl std::error::Error for LinearSolverError {}

impl LinearSolver {
    /// Create an SPGMR solver with left preconditioning.
    ///
    /// # Errors
    /// Returns an error if SUNDIALS cannot create or configure the solver.
    /// # Safety
    /// The vector's context must outlive the solver.
    pub unsafe fn spgmr(y: &NVector) -> Result<Self, LinearSolverError> {
        unsafe { Self::spgmr_with_options(y, SpgmrOptions::default()) }
    }

    /// Create a configurable SPGMR solver. KINSOL requires right preconditioning.
    ///
    /// # Errors
    /// Returns an error for out-of-range options or a SUNDIALS failure.
    /// # Safety
    /// The vector's context must outlive the solver.
    pub unsafe fn spgmr_with_options(
        y: &NVector,
        options: SpgmrOptions,
    ) -> Result<Self, LinearSolverError> {
        let maxl = i32::try_from(options.max_krylov_dimension)
            .map_err(|_| LinearSolverError::ValueOutOfRange("max_krylov_dimension"))?;
        let max_restarts = i32::try_from(options.max_restarts)
            .map_err(|_| LinearSolverError::ValueOutOfRange("max_restarts"))?;
        let ctx = y.context_raw();
        let ptr =
            unsafe { SUNLinSol_SPGMR(y.as_raw(), options.preconditioner.as_raw(), maxl, ctx) };
        if ptr.is_null() {
            return Err(LinearSolverError::CreationFailed);
        }

        let solver = Self {
            ptr,
            ctx,
            matrix: None,
            dimension: y.len(),
        };
        let status = unsafe { SUNLinSol_SPGMRSetGSType(ptr, options.gram_schmidt.as_raw()) };
        if status != 0 {
            return Err(LinearSolverError::ConfigurationFailed(status));
        }
        let status = unsafe { SUNLinSol_SPGMRSetMaxRestarts(ptr, max_restarts) };
        if status != 0 {
            return Err(LinearSolverError::ConfigurationFailed(status));
        }
        Ok(solver)
    }

    /// Create an SPFGMR solver with left preconditioning.
    ///
    /// # Errors
    /// Returns an error if SUNDIALS cannot allocate the solver.
    /// # Safety
    /// The vector's context must outlive the solver.
    pub unsafe fn spfgmr(y: &NVector) -> Result<Self, LinearSolverError> {
        let ctx = y.context_raw();
        let ptr =
            unsafe { SUNLinSol_SPFGMR(y.as_raw(), PreconditionerType::Left.as_raw(), 0, ctx) };
        Self::from_iterative(ptr, ctx, y.len())
    }

    /// Create an SPBCGS solver with left preconditioning.
    ///
    /// # Errors
    /// Returns an error if SUNDIALS cannot allocate the solver.
    /// # Safety
    /// The vector's context must outlive the solver.
    pub unsafe fn spbcgs(y: &NVector) -> Result<Self, LinearSolverError> {
        let ctx = y.context_raw();
        let ptr =
            unsafe { SUNLinSol_SPBCGS(y.as_raw(), PreconditionerType::Left.as_raw(), 0, ctx) };
        Self::from_iterative(ptr, ctx, y.len())
    }

    /// Create an SPTFQMR solver with left preconditioning.
    ///
    /// # Errors
    /// Returns an error if SUNDIALS cannot allocate the solver.
    /// # Safety
    /// The vector's context must outlive the solver.
    pub unsafe fn sptfqmr(y: &NVector) -> Result<Self, LinearSolverError> {
        let ctx = y.context_raw();
        let ptr =
            unsafe { SUNLinSol_SPTFQMR(y.as_raw(), PreconditionerType::Left.as_raw(), 0, ctx) };
        Self::from_iterative(ptr, ctx, y.len())
    }

    /// Create a dense direct solver, taking ownership of its matrix.
    ///
    /// # Errors
    /// Returns an error if the matrix is not square with the vector's length,
    /// the contexts differ, or SUNDIALS cannot allocate the solver.
    /// # Safety
    /// The vector and matrix context must outlive the solver.
    pub unsafe fn dense(y: &NVector, matrix: DenseMatrix) -> Result<Self, LinearSolverError> {
        if y.context_raw() != matrix.context_raw() {
            return Err(LinearSolverError::ContextMismatch);
        }
        if matrix.rows() != y.len() || matrix.columns() != y.len() {
            return Err(LinearSolverError::MatrixShapeMismatch);
        }
        let ctx = y.context_raw();
        let ptr = unsafe { SUNLinSol_Dense(y.as_raw(), matrix.as_raw(), ctx) };
        if ptr.is_null() {
            return Err(LinearSolverError::CreationFailed);
        }
        Ok(Self {
            ptr,
            ctx,
            matrix: Some(matrix),
            dimension: y.len(),
        })
    }

    fn from_iterative(
        ptr: SUNLinearSolver,
        ctx: SUNContext,
        dimension: usize,
    ) -> Result<Self, LinearSolverError> {
        if ptr.is_null() {
            Err(LinearSolverError::CreationFailed)
        } else {
            Ok(Self {
                ptr,
                ctx,
                matrix: None,
                dimension,
            })
        }
    }

    #[must_use]
    pub fn as_raw(&self) -> SUNLinearSolver {
        self.ptr
    }

    pub(crate) fn matrix_raw(&self) -> sundials_sys::SUNMatrix {
        self.matrix
            .as_ref()
            .map_or(std::ptr::null_mut(), DenseMatrix::as_raw)
    }

    pub(crate) fn context_raw(&self) -> SUNContext {
        self.ctx
    }

    pub(crate) fn dimension(&self) -> usize {
        self.dimension
    }
}

impl Drop for LinearSolver {
    fn drop(&mut self) {
        unsafe { SUNLinSolFree(self.ptr) };
    }
}
