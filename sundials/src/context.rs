use sundials_sys::{SUN_COMM_NULL, SUNContext, SUNContext_Create, SUNContext_Free};

pub struct SunContext {
    ptr: SUNContext,
}

impl Default for SunContext {
    fn default() -> Self {
        Self::new()
    }
}

impl SunContext {
    /// Create a new [`SunContext`].
    ///
    /// # Panics
    /// Panics if the context creation fails (i.e., if the returned pointer is null
    /// which should not happen under normal circumstances).
    #[must_use]
    pub fn new() -> Self {
        let mut ctx = std::ptr::null_mut();
        let status = unsafe { SUNContext_Create(SUN_COMM_NULL, &raw mut ctx) };
        assert_eq!(status, 0, "SUNDIALS failed to create a context");
        assert!(!ctx.is_null(), "SUNDIALS returned a null context");
        Self { ptr: ctx }
    }

    #[must_use]
    #[inline]
    pub fn as_raw(&self) -> SUNContext {
        self.ptr
    }
}

impl Drop for SunContext {
    #[inline]
    fn drop(&mut self) {
        unsafe { SUNContext_Free(&raw mut self.ptr) };
    }
}
