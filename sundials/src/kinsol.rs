use std::ffi::{c_long, c_void};
use std::fmt;

use sundials_sys::{
    KIN_FP, KIN_LINESEARCH, KIN_MEM_FAIL, KIN_NONE, KIN_PICARD, KINCreate, KINFree, KINGetFuncNorm,
    KINGetLastLinFlag, KINGetNumBacktrackOps, KINGetNumBetaCondFails, KINGetNumFuncEvals,
    KINGetNumJtimesEvals, KINGetNumLinConvFails, KINGetNumLinFuncEvals, KINGetNumLinIters,
    KINGetNumNonlinSolvIters, KINGetNumPrecEvals, KINGetNumPrecSolves, KINGetStepLength, KINInit,
    KINLsJacTimesVecFn, KINLsPrecSetupFn, KINLsPrecSolveFn, KINSetConstraints, KINSetFuncNormTol,
    KINSetJacTimesVecFn, KINSetLinearSolver, KINSetMaxBetaFails, KINSetMaxNewtonStep,
    KINSetMaxSetupCalls, KINSetNumMaxIters, KINSetPreconditioner, KINSetRelErrFunc,
    KINSetScaledStepTol, KINSetSysFunc, KINSetUserData, KINSol, KINSysFn, SUNContext,
};

use crate::{LinearSolver, NVector, SunContext};

/// Global strategy used by KINSOL.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum KinsolStrategy {
    Newton,
    #[default]
    LineSearch,
    Picard,
    FixedPoint,
}

impl KinsolStrategy {
    #[inline]
    fn as_raw(self) -> i32 {
        match self {
            Self::Newton => KIN_NONE,
            Self::LineSearch => KIN_LINESEARCH,
            Self::Picard => KIN_PICARD,
            Self::FixedPoint => KIN_FP,
        }
    }
}

/// Successful KINSOL stopping condition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KinsolOutcome {
    Converged,
    InitialGuessOk,
    StepToleranceReached,
}

/// Failure reported by the KINSOL linear-solver interface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KinsolLinearError {
    MemoryNull,
    LinearMemoryNull,
    IllegalInput,
    MemoryFailure,
    PreconditionerMemoryNull,
    JacobianFunctionFailure,
    MatrixFailure,
    LinearSolverFailure,
    Unknown(i32),
}

impl KinsolLinearError {
    fn from_raw(code: i32) -> Self {
        match code {
            -1 => Self::MemoryNull,
            -2 => Self::LinearMemoryNull,
            -3 => Self::IllegalInput,
            -4 => Self::MemoryFailure,
            -5 => Self::PreconditionerMemoryNull,
            -6 => Self::JacobianFunctionFailure,
            -7 => Self::MatrixFailure,
            -8 => Self::LinearSolverFailure,
            _ => Self::Unknown(code),
        }
    }
}

/// Error returned by the safe KINSOL wrapper.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KinsolError {
    CreationFailed,
    ContextMismatch,
    LinearSolverAlreadyAttached,
    NotInitialized,
    AlreadyInitialized,
    InvalidVectorLength {
        vector: &'static str,
        expected: usize,
        actual: usize,
    },
    Warning,
    MemoryNull,
    IllegalInput,
    NotAllocated,
    MemoryFailure,
    LineSearchNonconvergence,
    MaximumIterationsReached,
    RepeatedMaximumNewtonStep,
    LineSearchBetaConditionFailure,
    LinearSolverNoRecovery,
    LinearInitializationFailure,
    LinearSetupFailure,
    LinearSolveFailure,
    SystemFunctionFailure,
    FirstSystemFunctionRecoverableError,
    RepeatedSystemFunctionRecoverableError,
    VectorOperationFailure,
    ContextFailure,
    DampingFunctionFailure,
    DepthFunctionFailure,
    LinearInterface(KinsolLinearError),
    Unknown(i32),
}

impl KinsolError {
    fn from_main_raw(code: i32) -> Self {
        match code {
            99 => Self::Warning,
            -1 => Self::MemoryNull,
            -2 => Self::IllegalInput,
            -3 => Self::NotAllocated,
            -4 => Self::MemoryFailure,
            -5 => Self::LineSearchNonconvergence,
            -6 => Self::MaximumIterationsReached,
            -7 => Self::RepeatedMaximumNewtonStep,
            -8 => Self::LineSearchBetaConditionFailure,
            -9 => Self::LinearSolverNoRecovery,
            -10 => Self::LinearInitializationFailure,
            -11 => Self::LinearSetupFailure,
            -12 => Self::LinearSolveFailure,
            -13 => Self::SystemFunctionFailure,
            -14 => Self::FirstSystemFunctionRecoverableError,
            -15 => Self::RepeatedSystemFunctionRecoverableError,
            -16 => Self::VectorOperationFailure,
            -17 => Self::ContextFailure,
            -18 => Self::DampingFunctionFailure,
            -19 => Self::DepthFunctionFailure,
            _ => Self::Unknown(code),
        }
    }

    fn check_main(code: i32) -> Result<(), Self> {
        if code == 0 {
            Ok(())
        } else {
            Err(Self::from_main_raw(code))
        }
    }

    fn check_linear(code: i32) -> Result<(), Self> {
        if code == 0 {
            Ok(())
        } else {
            Err(Self::LinearInterface(KinsolLinearError::from_raw(code)))
        }
    }
}

impl fmt::Display for KinsolLinearError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl fmt::Display for KinsolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVectorLength {
                vector,
                expected,
                actual,
            } => write!(
                f,
                "KINSOL {vector} vector has length {actual}, expected {expected}"
            ),
            Self::LinearInterface(error) => write!(f, "KINSOL linear interface error: {error}"),
            _ => write!(f, "KINSOL error: {self:?}"),
        }
    }
}

impl std::error::Error for KinsolLinearError {}
impl std::error::Error for KinsolError {}

/// Statistics accumulated by the most recent KINSOL solve.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KinsolStatistics {
    pub nonlinear_iterations: i64,
    pub function_evaluations: i64,
    pub beta_condition_failures: i64,
    pub backtrack_operations: i64,
    pub function_norm: f64,
    pub step_length: f64,
    pub linear_function_evaluations: i64,
    pub preconditioner_evaluations: i64,
    pub preconditioner_solves: i64,
    pub linear_iterations: i64,
    pub linear_convergence_failures: i64,
    pub jacobian_times_evaluations: i64,
    pub last_linear_flag: i64,
}

/// RAII wrapper for a SUNDIALS KINSOL nonlinear solver.
pub struct Kinsol {
    ptr: *mut c_void,
    ctx: SUNContext,
    linear_solver: Option<LinearSolver>,
    dimension: Option<usize>,
}

impl Kinsol {
    /// Create a KINSOL solver associated with `ctx`.
    ///
    /// # Safety
    /// `ctx` must outlive the solver and all objects attached to it.
    ///
    /// # Errors
    /// Returns [`KinsolError::CreationFailed`] when SUNDIALS cannot allocate
    /// the solver.
    pub unsafe fn new(ctx: &SunContext) -> Result<Self, KinsolError> {
        let ptr = unsafe { KINCreate(ctx.as_raw()) };
        if ptr.is_null() {
            Err(KinsolError::CreationFailed)
        } else {
            Ok(Self {
                ptr,
                ctx: ctx.as_raw(),
                linear_solver: None,
                dimension: None,
            })
        }
    }

    #[must_use]
    #[inline]
    pub fn as_raw(&self) -> *mut c_void {
        self.ptr
    }

    /// Initialize the nonlinear system callback and vector template.
    ///
    pub fn init(&mut self, function: KINSysFn, template: &NVector) -> Result<(), KinsolError> {
        if self.dimension.is_some() {
            return Err(KinsolError::AlreadyInitialized);
        }
        if self.ctx != template.context_raw() {
            return Err(KinsolError::ContextMismatch);
        }
        let status = unsafe { KINInit(self.ptr, function, template.as_raw()) };
        // KINInit frees its own memory on vector-allocation failure.
        if status == KIN_MEM_FAIL {
            self.ptr = std::ptr::null_mut();
        }
        KinsolError::check_main(status)?;
        self.dimension = Some(template.len());
        Ok(())
    }

    /// Replace the nonlinear system callback without reallocating the solver.
    pub fn set_system_function(&mut self, function: KINSysFn) -> Result<(), KinsolError> {
        if self.dimension.is_none() {
            return Err(KinsolError::NotInitialized);
        }
        KinsolError::check_main(unsafe { KINSetSysFunc(self.ptr, function) })
    }

    /// Attach application-owned callback data.
    ///
    /// # Safety
    /// `data` must remain at a stable address and outlive this solver or be
    /// detached before it is dropped.
    pub unsafe fn set_user_data<T>(&mut self, data: &mut T) -> Result<(), KinsolError> {
        KinsolError::check_main(unsafe {
            KINSetUserData(self.ptr, std::ptr::from_mut(data).cast::<c_void>())
        })
    }

    /// Attach and own a linear solver and its optional matrix. KINSOL requires
    /// right preconditioning for iterative linear solvers.
    pub fn set_linear_solver(&mut self, linear_solver: LinearSolver) -> Result<(), KinsolError> {
        if self.linear_solver.is_some() {
            return Err(KinsolError::LinearSolverAlreadyAttached);
        }
        if self.ctx != linear_solver.context_raw() {
            return Err(KinsolError::ContextMismatch);
        }
        let expected = self.dimension.ok_or(KinsolError::NotInitialized)?;
        if linear_solver.dimension() != expected {
            return Err(KinsolError::InvalidVectorLength {
                vector: "linear solver",
                expected,
                actual: linear_solver.dimension(),
            });
        }
        KinsolError::check_linear(unsafe {
            KINSetLinearSolver(self.ptr, linear_solver.as_raw(), linear_solver.matrix_raw())
        })?;
        self.linear_solver = Some(linear_solver);
        Ok(())
    }

    pub fn set_preconditioner(
        &mut self,
        setup: KINLsPrecSetupFn,
        solve: KINLsPrecSolveFn,
    ) -> Result<(), KinsolError> {
        KinsolError::check_linear(unsafe { KINSetPreconditioner(self.ptr, setup, solve) })
    }

    /// Override KINSOL's default finite-difference Jacobian-vector product.
    pub fn set_jacobian_times_vector(
        &mut self,
        function: KINLsJacTimesVecFn,
    ) -> Result<(), KinsolError> {
        KinsolError::check_linear(unsafe { KINSetJacTimesVecFn(self.ptr, function) })
    }

    /// Set component-wise inequality constraints. KINSOL copies the vector.
    pub fn set_constraints(&mut self, constraints: &NVector) -> Result<(), KinsolError> {
        self.validate_length("constraints", constraints)?;
        KinsolError::check_main(unsafe { KINSetConstraints(self.ptr, constraints.as_raw()) })
    }

    pub fn set_max_iterations(&mut self, iterations: c_long) -> Result<(), KinsolError> {
        KinsolError::check_main(unsafe { KINSetNumMaxIters(self.ptr, iterations) })
    }

    pub fn set_max_setup_calls(&mut self, calls: c_long) -> Result<(), KinsolError> {
        KinsolError::check_main(unsafe { KINSetMaxSetupCalls(self.ptr, calls) })
    }

    pub fn set_function_norm_tolerance(&mut self, tolerance: f64) -> Result<(), KinsolError> {
        KinsolError::check_main(unsafe { KINSetFuncNormTol(self.ptr, tolerance) })
    }

    pub fn set_scaled_step_tolerance(&mut self, tolerance: f64) -> Result<(), KinsolError> {
        KinsolError::check_main(unsafe { KINSetScaledStepTol(self.ptr, tolerance) })
    }

    pub fn set_max_newton_step(&mut self, length: f64) -> Result<(), KinsolError> {
        KinsolError::check_main(unsafe { KINSetMaxNewtonStep(self.ptr, length) })
    }

    pub fn set_max_beta_failures(&mut self, failures: c_long) -> Result<(), KinsolError> {
        KinsolError::check_main(unsafe { KINSetMaxBetaFails(self.ptr, failures) })
    }

    pub fn set_relative_function_error(&mut self, relative_error: f64) -> Result<(), KinsolError> {
        KinsolError::check_main(unsafe { KINSetRelErrFunc(self.ptr, relative_error) })
    }

    /// Solve the nonlinear system in place.
    ///
    /// A successful return reports why KINSOL stopped. Callers should apply
    /// problem-specific post-validation before accepting `StepToleranceReached`.
    pub fn solve(
        &mut self,
        solution: &mut NVector,
        solution_scale: &NVector,
        function_scale: &NVector,
        strategy: KinsolStrategy,
    ) -> Result<KinsolOutcome, KinsolError> {
        self.validate_length("solution", solution)?;
        self.validate_length("solution_scale", solution_scale)?;
        self.validate_length("function_scale", function_scale)?;

        let status = unsafe {
            KINSol(
                self.ptr,
                solution.as_raw(),
                strategy.as_raw(),
                solution_scale.as_raw(),
                function_scale.as_raw(),
            )
        };
        match status {
            0 => Ok(KinsolOutcome::Converged),
            1 => Ok(KinsolOutcome::InitialGuessOk),
            2 => Ok(KinsolOutcome::StepToleranceReached),
            code => Err(KinsolError::from_main_raw(code)),
        }
    }

    /// Read nonlinear and linear statistics from the most recent solve.
    pub fn statistics(&self) -> Result<KinsolStatistics, KinsolError> {
        let mut nonlinear_iterations: c_long = 0;
        let mut function_evaluations: c_long = 0;
        let mut beta_condition_failures: c_long = 0;
        let mut backtrack_operations: c_long = 0;
        let mut function_norm = 0.0;
        let mut step_length = 0.0;
        let mut linear_function_evaluations: c_long = 0;
        let mut preconditioner_evaluations: c_long = 0;
        let mut preconditioner_solves: c_long = 0;
        let mut linear_iterations: c_long = 0;
        let mut linear_convergence_failures: c_long = 0;
        let mut jacobian_times_evaluations: c_long = 0;
        let mut last_linear_flag: c_long = 0;

        macro_rules! main_stat {
            ($function:ident, $value:ident) => {
                KinsolError::check_main(unsafe { $function(self.ptr, &raw mut $value) })?;
            };
        }
        macro_rules! linear_stat {
            ($function:ident, $value:ident) => {
                KinsolError::check_linear(unsafe { $function(self.ptr, &raw mut $value) })?;
            };
        }

        main_stat!(KINGetNumNonlinSolvIters, nonlinear_iterations);
        main_stat!(KINGetNumFuncEvals, function_evaluations);
        main_stat!(KINGetNumBetaCondFails, beta_condition_failures);
        main_stat!(KINGetNumBacktrackOps, backtrack_operations);
        main_stat!(KINGetFuncNorm, function_norm);
        main_stat!(KINGetStepLength, step_length);
        linear_stat!(KINGetNumLinFuncEvals, linear_function_evaluations);
        linear_stat!(KINGetNumPrecEvals, preconditioner_evaluations);
        linear_stat!(KINGetNumPrecSolves, preconditioner_solves);
        linear_stat!(KINGetNumLinIters, linear_iterations);
        linear_stat!(KINGetNumLinConvFails, linear_convergence_failures);
        linear_stat!(KINGetNumJtimesEvals, jacobian_times_evaluations);
        linear_stat!(KINGetLastLinFlag, last_linear_flag);

        Ok(KinsolStatistics {
            nonlinear_iterations: nonlinear_iterations as i64,
            function_evaluations: function_evaluations as i64,
            beta_condition_failures: beta_condition_failures as i64,
            backtrack_operations: backtrack_operations as i64,
            function_norm,
            step_length,
            linear_function_evaluations: linear_function_evaluations as i64,
            preconditioner_evaluations: preconditioner_evaluations as i64,
            preconditioner_solves: preconditioner_solves as i64,
            linear_iterations: linear_iterations as i64,
            linear_convergence_failures: linear_convergence_failures as i64,
            jacobian_times_evaluations: jacobian_times_evaluations as i64,
            last_linear_flag: last_linear_flag as i64,
        })
    }

    fn validate_length(&self, vector: &'static str, value: &NVector) -> Result<(), KinsolError> {
        if self.ctx != value.context_raw() {
            return Err(KinsolError::ContextMismatch);
        }
        let expected = self.dimension.ok_or(KinsolError::NotInitialized)?;
        let actual = value.len();
        if actual == expected {
            Ok(())
        } else {
            Err(KinsolError::InvalidVectorLength {
                vector,
                expected,
                actual,
            })
        }
    }
}

impl Drop for Kinsol {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { KINFree(&raw mut self.ptr) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GramSchmidtType, LinearSolverError, PreconditionerType, SpgmrOptions};
    use sundials_sys::{N_VGetArrayPointer_Serial, N_Vector};

    #[derive(Default)]
    struct UserData {
        diagonal: [f64; 2],
        system_calls: usize,
        preconditioner_setups: usize,
        preconditioner_solves: usize,
    }

    unsafe extern "C" fn system(
        solution: N_Vector,
        residual: N_Vector,
        user_data: *mut c_void,
    ) -> i32 {
        let solution = unsafe { N_VGetArrayPointer_Serial(solution) };
        let residual = unsafe { N_VGetArrayPointer_Serial(residual) };
        let data = unsafe { &mut *user_data.cast::<UserData>() };

        unsafe {
            *residual = solution.read().powi(2) - 4.0;
            residual.add(1).write(solution.add(1).read() - 3.0);
        }
        data.system_calls += 1;
        0
    }

    unsafe extern "C" fn setup_preconditioner(
        solution: N_Vector,
        _solution_scale: N_Vector,
        _residual: N_Vector,
        _residual_scale: N_Vector,
        user_data: *mut c_void,
    ) -> i32 {
        let solution = unsafe { N_VGetArrayPointer_Serial(solution) };
        let data = unsafe { &mut *user_data.cast::<UserData>() };
        data.diagonal = [2.0 * unsafe { solution.read() }, 1.0];
        data.preconditioner_setups += 1;
        0
    }

    unsafe extern "C" fn solve_preconditioner(
        _solution: N_Vector,
        _solution_scale: N_Vector,
        _residual: N_Vector,
        _residual_scale: N_Vector,
        vector: N_Vector,
        user_data: *mut c_void,
    ) -> i32 {
        let vector = unsafe { N_VGetArrayPointer_Serial(vector) };
        let data = unsafe { &mut *user_data.cast::<UserData>() };
        unsafe {
            *vector /= data.diagonal[0];
            *vector.add(1) /= data.diagonal[1];
        }
        data.preconditioner_solves += 1;
        0
    }

    #[test]
    fn solves_constrained_system_with_right_preconditioning() {
        let context = SunContext::new();
        let mut solution = unsafe { NVector::new_serial(2, &context) }.unwrap();
        solution.slice_mut().copy_from_slice(&[1.0, 1.0]);
        let mut scale = unsafe { NVector::new_serial(2, &context) }.unwrap();
        scale.slice_mut().fill(1.0);
        let mut constraints = unsafe { NVector::new_serial(2, &context) }.unwrap();
        constraints.slice_mut().fill(2.0);

        let linear_solver = unsafe {
            LinearSolver::spgmr_with_options(
                &solution,
                SpgmrOptions {
                    preconditioner: PreconditionerType::Right,
                    max_krylov_dimension: 10,
                    max_restarts: 2,
                    gram_schmidt: GramSchmidtType::Modified,
                },
            )
        }
        .unwrap();
        let mut data = Box::<UserData>::default();
        let mut solver = unsafe { Kinsol::new(&context) }.unwrap();
        solver.init(Some(system), &solution).unwrap();
        unsafe { solver.set_user_data(data.as_mut()) }.unwrap();
        solver.set_linear_solver(linear_solver).unwrap();
        let second = unsafe { LinearSolver::spgmr(&solution) }.unwrap();
        assert_eq!(
            solver.set_linear_solver(second),
            Err(KinsolError::LinearSolverAlreadyAttached)
        );
        solver
            .set_preconditioner(Some(setup_preconditioner), Some(solve_preconditioner))
            .unwrap();
        solver.set_constraints(&constraints).unwrap();
        drop(constraints); // KINSOL owns a private copy after set_constraints.
        solver.set_max_iterations(50).unwrap();
        solver.set_max_setup_calls(1).unwrap();
        solver.set_function_norm_tolerance(1e-12).unwrap();
        solver.set_scaled_step_tolerance(1e-12).unwrap();

        let outcome = solver
            .solve(&mut solution, &scale, &scale, KinsolStrategy::LineSearch)
            .unwrap();
        assert_eq!(outcome, KinsolOutcome::Converged);
        assert!((solution.slice()[0] - 2.0).abs() < 1e-10);
        assert!((solution.slice()[1] - 3.0).abs() < 1e-10);

        let statistics = solver.statistics().unwrap();
        assert!(statistics.nonlinear_iterations > 0);
        assert!(statistics.function_evaluations > 0);
        assert!(statistics.linear_iterations > 0);
        assert!(statistics.preconditioner_evaluations > 0);
        assert!(statistics.preconditioner_solves > 0);
        assert!(statistics.function_norm <= 1e-12);
        assert!(data.system_calls > 0);
        assert!(data.preconditioner_setups > 0);
        assert!(data.preconditioner_solves > 0);

        solution.slice_mut().copy_from_slice(&[2.0, 3.0]);
        let outcome = solver
            .solve(&mut solution, &scale, &scale, KinsolStrategy::LineSearch)
            .unwrap();
        assert_eq!(outcome, KinsolOutcome::InitialGuessOk);
    }

    #[test]
    fn rejects_mismatched_vector_lengths_before_ffi() {
        let context = SunContext::new();
        let mut solution = unsafe { NVector::new_serial(2, &context) }.unwrap();
        let scale = unsafe { NVector::new_serial(1, &context) }.unwrap();
        let linear_result = unsafe {
            LinearSolver::spgmr_with_options(
                &solution,
                SpgmrOptions {
                    max_krylov_dimension: usize::MAX,
                    ..SpgmrOptions::default()
                },
            )
        };
        assert!(matches!(
            linear_result,
            Err(LinearSolverError::ValueOutOfRange("max_krylov_dimension"))
        ));

        let mut solver = unsafe { Kinsol::new(&context) }.unwrap();
        solver.init(Some(system), &solution).unwrap();
        assert_eq!(
            solver.init(Some(system), &solution),
            Err(KinsolError::AlreadyInitialized)
        );

        assert_eq!(
            solver.solve(&mut solution, &scale, &scale, KinsolStrategy::LineSearch),
            Err(KinsolError::InvalidVectorLength {
                vector: "solution_scale",
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn maps_kinsol_and_linear_interface_errors() {
        assert_eq!(
            KinsolError::from_main_raw(-6),
            KinsolError::MaximumIterationsReached
        );
        assert_eq!(
            KinsolLinearError::from_raw(-5),
            KinsolLinearError::PreconditionerMemoryNull
        );
    }
}
