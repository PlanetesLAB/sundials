use bindgen::{BindgenError, Bindings};
use std::{
    collections::HashSet,
    env,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

// SUNDIALS has a few non-negative constants that need to be parsed as an i32.
// This is an attempt at doing so generally.
#[derive(Debug)]
struct ParseSignedConstants;

impl bindgen::callbacks::ParseCallbacks for ParseSignedConstants {
    fn int_macro(&self, name: &str, _value: i64) -> Option<bindgen::callbacks::IntKind> {
        let prefix: String = name.chars().take_while(|c| *c != '_').collect();
        match prefix.as_ref() {
            "CV" | "IDA" | "KIN" | "SUN" => Some(bindgen::callbacks::IntKind::Int),
            _ => None,
        }
    }
}

// Ignore some macros (based on https://github.com/rust-lang/rust-bindgen/issues/687#issuecomment-1312298570)
#[derive(Debug)]
struct IgnoreMacros(HashSet<&'static str>);

impl bindgen::callbacks::ParseCallbacks for IgnoreMacros {
    fn will_parse_macro(&self, name: &str) -> bindgen::callbacks::MacroParsingBehavior {
        use bindgen::callbacks::MacroParsingBehavior;
        if self.0.contains(name) {
            MacroParsingBehavior::Ignore
        } else {
            MacroParsingBehavior::Default
        }
    }
}

impl IgnoreMacros {
    const IGNORE_CONSTANTS: [&'static str; 19] = [
        "FE_DIVBYZERO",
        "FE_DOWNWARD",
        "FE_INEXACT",
        "FE_INVALID",
        "FE_OVERFLOW",
        "FE_TONEAREST",
        "FE_TOWARDZERO",
        "FE_UNDERFLOW",
        "FE_UPWARD",
        "FP_INFINITE",
        "FP_INT_DOWNWARD",
        "FP_INT_TONEAREST",
        "FP_INT_TONEARESTFROMZERO",
        "FP_INT_TOWARDZERO",
        "FP_INT_UPWARD",
        "FP_NAN",
        "FP_NORMAL",
        "FP_SUBNORMAL",
        "FP_ZERO",
    ];

    fn new() -> Self {
        Self(Self::IGNORE_CONSTANTS.iter().copied().collect())
    }
}

#[derive(Debug)]
struct Library {
    /// Location if the include files.
    inc: Option<String>,
    /// Location of the library.
    lib: Option<String>,
}

/// Build the Sundials code vendor with sundials-sys.
fn build_vendor_sundials(klu: &Library) -> (Library, &'static str) {
    // helper: feature → ON/OFF
    macro_rules! feature {
        ($s:tt) => {
            if cfg!(feature = $s) { "ON" } else { "OFF" }
        };
    }

    // static vs shared
    let static_libraries = feature!("static_libraries");
    let (shared_libraries, library_type) = match static_libraries {
        "ON" => ("OFF", "static"),
        "OFF" => ("ON", "dylib"),
        _ => unreachable!(),
    };

    // detect platform + feature flags
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let openmp_enabled = std::env::var("CARGO_FEATURE_NVECOPENMP").is_ok();

    // OpenMP logic
    let openmp = if openmp_enabled && target_os != "macos" {
        "ON"
    } else {
        "OFF"
    };

    // optional: fail loudly if user explicitly asked for OpenMP on macOS
    if openmp_enabled && target_os == "macos" {
        println!("cargo:warning=OpenMP requested but not available on macOS, disabling.");
    }

    let mut config = cmake::Config::new("vendor/sundials");

    config
        // install dirs
        .define("CMAKE_INSTALL_BINDIR", "lib")
        .define("CMAKE_INSTALL_LIBDIR", "lib")
        // library type
        .define("BUILD_SHARED_LIBS", shared_libraries)
        // core toggles
        .define("BUILD_TESTING", "OFF")
        .define("SUNDIALS_ENABLE_EXAMPLES", "OFF")
        // modules (modern flags)
        .define("SUNDIALS_ENABLE_ARKODE", feature!("arkode"))
        .define("SUNDIALS_ENABLE_CVODE", feature!("cvode"))
        .define("SUNDIALS_ENABLE_CVODES", feature!("cvodes"))
        .define("SUNDIALS_ENABLE_IDA", feature!("ida"))
        .define("SUNDIALS_ENABLE_IDAS", feature!("idas"))
        .define("SUNDIALS_ENABLE_KINSOL", feature!("kinsol"))
        // TPLs
        .define("SUNDIALS_ENABLE_KLU", feature!("klu"))
        .define("SUNDIALS_ENABLE_PTHREAD", feature!("nvecpthreads"))
        .define("SUNDIALS_ENABLE_OPENMP", openmp);

    // optional KLU paths
    if let Some(inc) = &klu.inc {
        config.define("KLU_INCLUDE_DIR", inc);
    }
    if let Some(lib) = &klu.lib {
        config.define("KLU_LIBRARY_DIR", lib);
    }

    // build
    let dst = config.build();

    let dst_disp = dst.display();
    let lib_loc = Some(format!("{dst_disp}/lib"));
    let inc_dir = Some(format!("{dst_disp}/include"));

    (
        Library {
            inc: inc_dir,
            lib: lib_loc,
        },
        library_type,
    )
}

fn generate_bindings(
    inc_dirs: &[Option<String>],
    use_openmp: bool,
) -> Result<Bindings, BindgenError> {
    macro_rules! define {
        ($a:tt, $b:tt) => {
            format!(
                "-DUSE_{}={}",
                stringify!($b),
                if cfg!(feature = $a) { 1 } else { 0 }
            )
        };
    }

    let use_openmp_val = i32::from(use_openmp);

    let mut builder = bindgen::Builder::default().header("wrapper.h");
    for dir in inc_dirs.iter().flatten() {
        builder = builder.clang_arg(format!("-I{dir}"));
    }
    builder
        .clang_args(&[
            define!("arkode", ARKODE),
            define!("cvode", CVODE),
            define!("cvodes", CVODES),
            define!("ida", IDA),
            define!("idas", IDAS),
            define!("kinsol", KINSOL),
            define!("klu", KLU),
            format!("-DUSE_OPENMP={use_openmp_val}"),
            define!("nvecpthreads", PTHREADS),
        ])
        .parse_callbacks(Box::new(ParseSignedConstants))
        .parse_callbacks(Box::new(IgnoreMacros::new()))
        .blocklist_function("malloc")
        .blocklist_function("realloc")
        .blocklist_function("calloc")
        .blocklist_function("free")
        .generate()
}

fn get_sundials_version_major(bindings: impl AsRef<Path>) -> Option<u32> {
    let b = File::open(bindings).expect("Couldn't read file bindings.rs!");
    let mut b = BufReader::new(b).bytes();
    'version: while b.find(|c| c.as_ref().is_ok_and(|&c| c == b'S')).is_some() {
        for c0 in "UNDIALS_VERSION_MAJOR".bytes() {
            match b.next() {
                Some(Ok(c)) => {
                    if c != c0 {
                        continue 'version;
                    }
                }
                Some(Err(_)) | None => return None,
            }
        }
        // Match " : u32 = 6"
        if b.find(|c| c.as_ref().is_ok_and(|&c| c == b'=')).is_some() {
            let is_not_digit = |c: &u8| !c.is_ascii_digit();
            let b = b.skip_while(|c| c.as_ref().is_ok_and(is_not_digit));
            let v: Vec<_> = b.map_while(|c| c.ok().filter(u8::is_ascii_digit)).collect();
            match String::from_utf8(v) {
                Ok(v) => return v.parse().ok(),
                Err(_) => return None,
            }
        }
        return None;
    }
    None
}

#[allow(clippy::too_many_lines)]
fn main() {
    // get klu dirs
    let klu_inc = env::var("DEP_SUITESPARSE_SUITESPARSE_INCLUDE").ok();
    let klu_lib = env::var("DEP_SUITESPARSE_SUITESPARSE_LIB").ok();

    // First, we build the SUNDIALS library, with requested modules with CMake
    let klu = Library {
        inc: klu_inc,
        lib: klu_lib,
    };
    let mut sundials = Library {
        inc: None,
        lib: None,
    };
    let mut library_type = "dylib";
    if cfg!(any(feature = "build_libraries", target_family = "wasm")) {
        (sundials, library_type) = build_vendor_sundials(&klu);
    } else {
        sundials.inc = env::var("SUNDIALS_INCLUDE_DIR").ok();
        sundials.lib = env::var("SUNDIALS_LIBRARY_DIR").ok();
    }

    if sundials.lib.is_none() && sundials.inc.is_none() {
        #[cfg(target_family = "windows")]
        {
            let vcpkg = vcpkg::Config::new()
                .emit_includes(true)
                .find_package("sundials");
            if vcpkg.is_err() {
                (sundials, library_type) = build_vendor_sundials(&klu);
            }
        }
    }

    // Second, we use bindgen to generate the Rust types

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let use_openmp = cfg!(feature = "nvecopenmp") && target_os != "macos";

    let bindings_rs = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    let mut build_vendor = true;
    let mut sundials_version_major = 0;
    if let Ok(bindings) = generate_bindings(&[sundials.inc, klu.inc.clone()], use_openmp) {
        bindings
            .write_to_file(&bindings_rs)
            .expect("Couldn't write file bindings.rs!");
        if let Some(v) = get_sundials_version_major(&bindings_rs) {
            if v >= 6 {
                build_vendor = false;
                sundials_version_major = v;
            } else {
                println!(
                    "cargo:warning=System sundials version = \
                          {v} < 6, will use the vendor version"
                );
            }
        }
    }
    if build_vendor {
        (sundials, library_type) = build_vendor_sundials(&klu);
        if let Ok(bindings) = generate_bindings(&[sundials.inc, klu.inc.clone()], use_openmp) {
            bindings
                .write_to_file(&bindings_rs)
                .expect("Couldn't write file bindings.rs!");
            sundials_version_major = get_sundials_version_major(&bindings_rs)
                .expect("Cannot determine vendor sundials version!");
        } else {
            panic!("Unable to generate bindings of the vendor sundials!");
        }
    }
    println!(
        "cargo::rustc-check-cfg=cfg(sundials_version_major, \
        values(\"6\", \"7\"))"
    );
    println!("cargo::rustc-check-cfg=cfg(sundials_use_openmp)");
    println!("cargo:rustc-cfg=sundials_version_major=\"{sundials_version_major}\"");
    if use_openmp {
        println!("cargo:rustc-cfg=sundials_use_openmp");
    }

    // Third, we let Cargo know about the library files

    if let Some(dir) = sundials.lib {
        println!("cargo:rustc-link-search=native={dir}");
    }

    let mut lib_names = vec![];
    if sundials_version_major >= 7 {
        lib_names.push("core");
    }
    lib_names.append(&mut vec![
        "nvecserial",
        "sunlinsolband",
        "sunlinsoldense",
        "sunlinsolpcg",
        "sunlinsolspbcgs",
        "sunlinsolspfgmr",
        "sunlinsolspgmr",
        "sunlinsolsptfqmr",
        "sunmatrixband",
        "sunmatrixdense",
        "sunmatrixsparse",
        "sunnonlinsolfixedpoint",
        "sunnonlinsolnewton",
    ]);
    if cfg!(feature = "klu") {
        lib_names.push("sunlinsolklu");
    }
    macro_rules! link { ($($s:tt),*) => {
        $(if cfg!(feature = $s) { lib_names.push($s) })*
    }}
    link!(
        "arkode",
        "cvode",
        "cvodes",
        "ida",
        "idas",
        "kinsol",
        "nvecpthreads"
    );
    if use_openmp {
        lib_names.push("nvecopenmp");
    }

    for lib_name in &lib_names {
        println!("cargo:rustc-link-lib={library_type}=sundials_{lib_name}");
    }
    // And that's all.
}
