use std::ffi::{CString, c_void};

use sundials_sys::{
    CV_BDF, CV_NORMAL, CV_ONE_STEP, CVEwtFn, CVLsJacFn, CVLsPrecSetupFn, CVLsPrecSolveFn, CVRhsFn,
    CVode, CVodeCreate, CVodeFree, CVodeGetCurrentStep, CVodeGetCurrentTime, CVodeGetLastStep,
    CVodeInit, CVodePrintAllStats, CVodeReInit, CVodeSVtolerances, CVodeSetConstraints,
    CVodeSetEpsLin, CVodeSetEtaConvFail, CVodeSetInitStep, CVodeSetJacEvalFrequency, CVodeSetJacFn,
    CVodeSetLSetupFrequency, CVodeSetLinearSolver, CVodeSetMaxConvFails, CVodeSetMaxErrTestFails,
    CVodeSetMaxHnilWarns, CVodeSetMaxNonlinIters, CVodeSetMaxNumConstraintFails,
    CVodeSetMaxNumSteps, CVodeSetMaxOrd, CVodeSetMaxStep, CVodeSetMinStep, CVodeSetPreconditioner,
    CVodeSetStabLimDet, CVodeSetUserData, CVodeWFtolerances, FILE, SUNContext,
    SUNOutputFormat_SUN_OUTPUTFORMAT_TABLE, fclose, fopen,
};

use crate::{LinearSolver, NVector, SunContext};

pub struct Cvode {
    ptr: *mut c_void,
    ctx: SUNContext,
    linear_solver: Option<LinearSolver>,
    dimension: Option<usize>,
}

impl Cvode {
    /// Create a new CVODE solver in BDF mode with the given context.
    ///
    /// # Safety
    /// `ctx` must outlive the solver and all objects attached to it.
    ///
    /// # Panics
    /// Panics if the solver creation fails (i.e., if the returned pointer is null
    /// which can happen if there is an issue with the context).
    #[must_use]
    pub unsafe fn new_bdf(ctx: &SunContext) -> Self {
        let p = unsafe { CVodeCreate(CV_BDF, ctx.as_raw()) };
        assert!(!p.is_null());
        Self {
            ptr: p,
            ctx: ctx.as_raw(),
            linear_solver: None,
            dimension: None,
        }
    }

    #[must_use]
    #[inline]
    pub fn as_raw(&self) -> *mut c_void {
        self.ptr
    }

    /// Set user data.
    ///
    /// # Safety
    /// The data referenced by `data` must outlive the `Cvode` solver instance.
    /// Modifying or dropping the data while CVODE is still using it will cause undefined behavior.
    pub unsafe fn set_userdata<T>(&mut self, data: &mut T) -> Result<(), CvodeError> {
        CvodeError::from_raw(unsafe {
            CVodeSetUserData(self.ptr, std::ptr::from_mut::<T>(data).cast::<c_void>())
        })
    }

    pub fn init(&mut self, rhs: CVRhsFn, t0: f64, y: &NVector) -> Result<(), CvodeError> {
        if self.dimension.is_some() {
            return Err(CvodeError::AlreadyInitialized);
        }
        self.check_context(y.context_raw())?;
        CvodeError::from_raw(unsafe { CVodeInit(self.ptr, rhs, t0, y.as_raw()) })?;
        self.dimension = Some(y.len());
        Ok(())
    }

    pub fn reinit(&mut self, t0: f64, y: &NVector) -> Result<(), CvodeError> {
        self.check_vector("state", y)?;
        CvodeError::from_raw(unsafe { CVodeReInit(self.ptr, t0, y.as_raw()) })
    }

    pub fn set_tolerances(&mut self, reltol: f64, abstol: &NVector) -> Result<(), CvodeError> {
        self.check_vector("absolute tolerance", abstol)?;
        CvodeError::from_raw(unsafe { CVodeSVtolerances(self.ptr, reltol, abstol.as_raw()) })
    }

    /// Attach a linear solver and take ownership of it and its optional matrix.
    pub fn set_linear_solver(&mut self, linsol: LinearSolver) -> Result<(), CvodeError> {
        if self.linear_solver.is_some() {
            return Err(CvodeError::LinearSolverAlreadyAttached);
        }
        self.check_context(linsol.context_raw())?;
        self.check_dimension("linear solver", linsol.dimension())?;
        let status =
            unsafe { CVodeSetLinearSolver(self.ptr, linsol.as_raw(), linsol.matrix_raw()) };
        if status != 0 {
            return Err(CvodeError::LinearInterface(status));
        }
        self.linear_solver = Some(linsol);
        Ok(())
    }

    /// Register an explicit Jacobian callback for a matrix-based solver.
    pub fn set_jacobian(&mut self, jacobian: CVLsJacFn) -> Result<(), CvodeError> {
        let status = unsafe { CVodeSetJacFn(self.ptr, jacobian) };
        if status == 0 {
            Ok(())
        } else {
            Err(CvodeError::LinearInterface(status))
        }
    }

    /// Integrate the ODE system up to time `tout`, storing the solution in `y` and the actual time reached in `t`.
    ///
    /// # Errors
    /// Returns an error if the integration fails, which can happen for various reasons such as exceeding
    /// the maximum number of steps, convergence failures, or issues with the right-hand side function.
    pub fn integrate(&mut self, tout: f64, y: &mut NVector, t: &mut f64) -> Result<(), CvodeError> {
        self.check_vector("state", y)?;
        let retval = unsafe { CVode(self.ptr, tout, y.as_raw(), t, CV_NORMAL) };
        CvodeError::from_raw(retval)
    }

    pub fn integrate_one_step(
        &mut self,
        tout: f64,
        y: &mut NVector,
        t: &mut f64,
    ) -> Result<(), CvodeError> {
        self.check_vector("state", y)?;
        CvodeError::from_raw(unsafe { CVode(self.ptr, tout, y.as_raw(), t, CV_ONE_STEP) })
    }

    #[must_use]
    #[inline]
    pub fn get_current_time(&self) -> f64 {
        let mut t = 0.0;
        unsafe { CVodeGetCurrentTime(self.ptr, &raw mut t) };
        t
    }

    #[must_use]
    #[inline]
    pub fn get_last_step(&self) -> f64 {
        let mut h = 0.0;
        unsafe { CVodeGetLastStep(self.ptr, &raw mut h) };
        h
    }

    #[must_use]
    #[inline]
    pub fn get_current_step(&self) -> f64 {
        let mut h = 0.0;
        unsafe { CVodeGetCurrentStep(self.ptr, &raw mut h) };
        h
    }

    pub fn set_preconditioner(
        &mut self,
        setup: CVLsPrecSetupFn,
        solve: CVLsPrecSolveFn,
    ) -> Result<(), CvodeError> {
        let status = unsafe { CVodeSetPreconditioner(self.ptr, setup, solve) };
        if status == 0 {
            Ok(())
        } else {
            Err(CvodeError::LinearInterface(status))
        }
    }

    pub fn set_wf_tolerances(&mut self, efun: CVEwtFn) {
        unsafe { CVodeWFtolerances(self.ptr, efun) };
    }

    pub fn set_constraints(&mut self, constraints: &NVector) -> Result<(), CvodeError> {
        self.check_vector("constraints", constraints)?;
        CvodeError::from_raw(unsafe { CVodeSetConstraints(self.ptr, constraints.as_raw()) })
    }

    fn check_context(&self, ctx: SUNContext) -> Result<(), CvodeError> {
        if self.ctx == ctx {
            Ok(())
        } else {
            Err(CvodeError::ContextMismatch)
        }
    }

    fn check_dimension(&self, vector: &'static str, actual: usize) -> Result<(), CvodeError> {
        let expected = self.dimension.ok_or(CvodeError::NotInitialized)?;
        if actual == expected {
            Ok(())
        } else {
            Err(CvodeError::InvalidVectorLength {
                vector,
                expected,
                actual,
            })
        }
    }

    fn check_vector(&self, vector: &'static str, value: &NVector) -> Result<(), CvodeError> {
        self.check_context(value.context_raw())?;
        self.check_dimension(vector, value.len())
    }

    pub fn set_max_nonlin_iters(&mut self, n: i32) {
        unsafe { CVodeSetMaxNonlinIters(self.ptr, n) };
    }

    pub fn set_max_conv_fails(&mut self, n: i32) {
        unsafe { CVodeSetMaxConvFails(self.ptr, n) };
    }

    pub fn set_eta_conv_fail(&mut self, eta: f64) {
        unsafe { CVodeSetEtaConvFail(self.ptr, eta) };
    }

    pub fn set_max_err_test_fails(&mut self, n: i32) {
        unsafe { CVodeSetMaxErrTestFails(self.ptr, n) };
    }

    pub fn set_max_constraints_fails(&mut self, n: i32) {
        unsafe { CVodeSetMaxNumConstraintFails(self.ptr, n) };
    }

    pub fn set_init_step(&mut self, h0: f64) {
        unsafe { CVodeSetInitStep(self.ptr, h0) };
    }

    pub fn set_min_step(&mut self, hmin: f64) {
        unsafe { CVodeSetMinStep(self.ptr, hmin) };
    }

    pub fn set_max_step(&mut self, hmax: f64) {
        unsafe { CVodeSetMaxStep(self.ptr, hmax) };
    }

    pub fn set_max_ord(&mut self, ord: i32) {
        unsafe { CVodeSetMaxOrd(self.ptr, ord) };
    }

    pub fn set_max_num_steps(&mut self, n: i64) {
        unsafe { CVodeSetMaxNumSteps(self.ptr, n) };
    }

    pub fn set_max_hnil_warns(&mut self, n: i32) {
        unsafe { CVodeSetMaxHnilWarns(self.ptr, n) };
    }

    pub fn set_stability_limit_detection(&mut self, enable: i32) {
        unsafe { CVodeSetStabLimDet(self.ptr, enable) };
    }

    pub fn set_jac_eval_frequency(&mut self, freq: i64) {
        unsafe { CVodeSetJacEvalFrequency(self.ptr, freq) };
    }

    pub fn set_linear_solver_setup_frequency(&mut self, freq: i64) {
        unsafe { CVodeSetLSetupFrequency(self.ptr, freq) };
    }

    pub fn set_epslin(&mut self, eps: f64) {
        unsafe { CVodeSetEpsLin(self.ptr, eps) };
    }

    /// Save CVODE statistics to a file named ``cvode_stats.txt`` in a human-readable table format.
    ///
    /// # Panics
    /// Panics if the file cannot be opened for writing, which can happen due to permission
    /// issues or if the filesystem is read-only.
    /// Panics if the CVODE statistics cannot be printed, which can happen if there is an internal error in the CVODE library.
    /// Note that this function will overwrite the file ``cvode_stats.txt`` if it already exists.
    pub fn save_statistics(&self, fname: &str) {
        unsafe {
            let filename = CString::new(fname).unwrap();
            let mode = CString::new("w").unwrap();

            let file: *mut FILE = fopen(filename.as_ptr(), mode.as_ptr());
            assert!(!file.is_null(), "Failed to open file");

            let retval = CVodePrintAllStats(self.ptr, file, SUNOutputFormat_SUN_OUTPUTFORMAT_TABLE);
            if retval != 0 {
                eprintln!("Failed to print CVode statistics");
            }
            fclose(file);
        };
    }
}

impl Drop for Cvode {
    #[inline(always)]
    fn drop(&mut self) {
        unsafe { CVodeFree(&raw mut self.ptr) };
    }
}

#[derive(Debug, PartialEq)]
pub enum CvodeError {
    ContextMismatch,
    NotInitialized,
    AlreadyInitialized,
    LinearSolverAlreadyAttached,
    InvalidVectorLength {
        vector: &'static str,
        expected: usize,
        actual: usize,
    },
    LinearInterface(i32),
    Unknown(i32),
    Warning,
    TStopReturn,
    RootReturn,
    MemNull,
    NoMalloc,
    IllInput,
    TooClose,
    TooMuchWork,
    TooMuchAccuracy,
    ErrFailure,
    ConvergenceFailure,
    LinearInitFailure,
    LinearSetupFailure,
    LinearSolveFailure,
    ConstraintsFailure,
    RHSFuncFailure,
    FirstRHSFuncFailure,
    RepeatedRHSFuncFailure,
    UnrecoverableRHSFuncFailure,
    RootFindingFuncFailure,
}

impl CvodeError {
    fn from_raw(code: i32) -> Result<(), Self> {
        match code {
            0 => Ok(()), // CV_SUCCESS
            1 => Err(CvodeError::TStopReturn),
            2 => Err(CvodeError::RootReturn),
            99 => Err(CvodeError::Warning),
            -1 => Err(CvodeError::TooMuchWork),
            -2 => Err(CvodeError::TooMuchAccuracy),
            -3 => Err(CvodeError::ErrFailure),
            -4 => Err(CvodeError::ConvergenceFailure),
            -5 => Err(CvodeError::LinearInitFailure),
            -6 => Err(CvodeError::LinearSetupFailure),
            -7 => Err(CvodeError::LinearSolveFailure),
            -8 => Err(CvodeError::RHSFuncFailure),
            -9 => Err(CvodeError::FirstRHSFuncFailure),
            -10 => Err(CvodeError::RepeatedRHSFuncFailure),
            -11 => Err(CvodeError::UnrecoverableRHSFuncFailure),
            -12 => Err(CvodeError::RootFindingFuncFailure),
            -15 => Err(CvodeError::ConstraintsFailure),
            -21 => Err(CvodeError::MemNull),
            -22 => Err(CvodeError::IllInput),
            -23 => Err(CvodeError::NoMalloc),
            -27 => Err(CvodeError::TooClose),
            _ => Err(CvodeError::Unknown(code)),
        }
    }
}

#[cfg(test)]
mod tests {
    use sundials_sys::{N_VGetArrayPointer_Serial, N_Vector, SUNMatrix};

    use super::*;
    use crate::{DenseMatrix, DenseMatrixViewMut};

    unsafe extern "C" fn decay(
        _time: f64,
        state: N_Vector,
        derivative: N_Vector,
        _user_data: *mut c_void,
    ) -> i32 {
        let state_ptr = unsafe { N_VGetArrayPointer_Serial(state) };
        let derivative_ptr = unsafe { N_VGetArrayPointer_Serial(derivative) };
        unsafe { derivative_ptr.write(-state_ptr.read()) };
        0
    }

    unsafe extern "C" fn decay_jacobian(
        _time: f64,
        _state: N_Vector,
        _derivative: N_Vector,
        matrix: SUNMatrix,
        _user_data: *mut c_void,
        _tmp1: N_Vector,
        _tmp2: N_Vector,
        _tmp3: N_Vector,
    ) -> i32 {
        let Ok(mut matrix) = (unsafe { DenseMatrixViewMut::from_raw(matrix) }) else {
            return -1;
        };
        *matrix.get_mut(0, 0).unwrap() = -1.0;
        0
    }

    #[test]
    fn dense_solver_owns_matrix() {
        let ctx = SunContext::new();
        let mut state = unsafe { NVector::new_serial(1, &ctx) }.unwrap();
        state.slice_mut()[0] = 1.0;
        let mut absolute_tolerance = unsafe { NVector::new_serial(1, &ctx) }.unwrap();
        absolute_tolerance.slice_mut()[0] = 1e-12;
        let matrix = unsafe { DenseMatrix::new(1, 1, &ctx) }.unwrap();
        let linear = unsafe { LinearSolver::dense(&state, matrix) }.unwrap();
        let mut solver = unsafe { Cvode::new_bdf(&ctx) };

        solver.init(Some(decay), 0.0, &state).unwrap();
        solver.set_tolerances(1e-10, &absolute_tolerance).unwrap();
        solver.set_linear_solver(linear).unwrap();
        let second = unsafe { LinearSolver::spgmr(&state) }.unwrap();
        assert_eq!(
            solver.set_linear_solver(second),
            Err(CvodeError::LinearSolverAlreadyAttached)
        );
        solver.set_jacobian(Some(decay_jacobian)).unwrap();

        let mut time = 0.0;
        solver.integrate(1.0, &mut state, &mut time).unwrap();
        assert!((state.slice()[0] - (-1.0_f64).exp()).abs() < 1e-7);
        assert!((time - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_wrong_vector_length_before_ffi() {
        let ctx = SunContext::new();
        let state = unsafe { NVector::new_serial(2, &ctx) }.unwrap();
        let short = unsafe { NVector::new_serial(1, &ctx) }.unwrap();
        let mut solver = unsafe { Cvode::new_bdf(&ctx) };
        solver.init(Some(decay), 0.0, &state).unwrap();
        assert_eq!(
            solver.set_tolerances(1e-6, &short),
            Err(CvodeError::InvalidVectorLength {
                vector: "absolute tolerance",
                expected: 2,
                actual: 1,
            })
        );
    }
}
