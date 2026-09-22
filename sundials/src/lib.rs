#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::inline_always)]

mod context;
mod cvode;
mod dense_matrix;
mod kinsol;
mod linear_solver;
mod nvector;
mod vector_view;

pub use context::SunContext;
pub use cvode::{Cvode, CvodeError};
pub use dense_matrix::{DenseMatrix, DenseMatrixError, DenseMatrixViewError, DenseMatrixViewMut};
pub use kinsol::{
    Kinsol, KinsolError, KinsolLinearError, KinsolOutcome, KinsolStatistics, KinsolStrategy,
};
pub use linear_solver::{
    GramSchmidtType, LinearSolver, LinearSolverError, PreconditionerType, SpgmrOptions,
};
pub use nvector::{NVector, NVectorError};
pub use vector_view::{SerialVectorView, SerialVectorViewError, SerialVectorViewMut};
