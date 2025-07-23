#![feature(once_cell_try)]

use camino::Utf8PathBuf;
use clap::{Parser, ValueEnum};
use fastembed::ExecutionProviderDispatch;
use model::{BATCH_SIZE, Model, ModelKind, get_embedding_model_info};
use ort::environment::GlobalThreadPoolOptions;
#[cfg(windows)]
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, DirectMLExecutionProvider, ExecutionProvider,
    OpenVINOExecutionProvider, XNNPACKExecutionProvider,
};
#[cfg(target_os = "macos")]
use ort::execution_providers::{
    CPUExecutionProvider, CoreMLExecutionProvider, ExecutionProvider, XNNPACKExecutionProvider,
};
use serde_json::json;
use strum::EnumString;

use std::panic::set_hook;

mod model;
mod pool;
#[rustfmt::skip]
mod chunks;

use chunks::CHUNKS;

#[cfg(not(feature = "static"))]
fn find_onnxruntime_lib() -> Utf8PathBuf {
    use std::str::FromStr;

    let exe_dir = Utf8PathBuf::from_path_buf(std::env::current_exe().unwrap())
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    #[cfg(target_os = "linux")]
    const LIB_FILENAME: &str = "libonnxruntime.so";
    #[cfg(target_os = "windows")]
    const LIB_FILENAME: &str = "onnxruntime.dll";
    let mut dirs = vec![
        exe_dir.clone(),
        exe_dir.join("onnx"),
        exe_dir.join("../onnx"),
        exe_dir.join("../../onnx"),
    ];

    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(';') {
            dirs.push(Utf8PathBuf::from_str(dir.trim()).unwrap());
        }
    }

    search_dir(&dirs, LIB_FILENAME).expect("Failed to find onnxruntime library")
}

#[cfg(any(windows, target_os = "linux"))]
fn find_cuda_lib_path(file: &str) -> Utf8PathBuf {
    use std::str::FromStr;
    let exe_dir = Utf8PathBuf::from_path_buf(std::env::current_exe().unwrap())
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let mut dirs = vec![
        exe_dir.clone(),
        exe_dir.join("cuda"),
        exe_dir.join("../../cuda"),
    ];

    if let Ok(dir) = std::env::var("CUDA_HOME") {
        dirs.push(Utf8PathBuf::from_str(&dir).unwrap());
    }

    if let Ok(dir) = std::env::var("CUDNN_ROOT") {
        dirs.push(Utf8PathBuf::from_str(&dir).unwrap());
    }

    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(';') {
            dirs.push(Utf8PathBuf::from_str(dir.trim()).unwrap());
        }
    }

    Utf8PathBuf::from_path_buf(
        dunce::canonicalize(
            search_dir(&dirs, file)
                .expect("Failed to find cuda dir")
                .parent()
                .unwrap(),
        )
        .expect("Failed to canonicalize cuda dir"),
    )
    .unwrap()
}

#[cfg(any(windows, target_os = "linux"))]
fn preload_cuda() -> anyhow::Result<()> {
    use ort::execution_providers::cuda::{CUDA_DYLIBS, CUDNN_DYLIBS};
    use ort::util::preload_dylib;
    for lib in CUDA_DYLIBS {
        let path = find_cuda_lib_path(lib).join(lib);
        eprintln!("Preloading CUDA dylib: {}", path);
        preload_dylib(path.as_os_str())?;
    }
    for lib in CUDNN_DYLIBS {
        let path = find_cuda_lib_path(lib).join(lib);
        eprintln!("Preloading CUDNN dylib: {}", path);
        preload_dylib(path.as_os_str())?;
    }
    Ok(())
}

#[cfg(not(feature = "static"))]
fn setup_onnxruntime(shoud_preload_cuda: bool) {
    eprintln!("Initializing onnxruntime");

    #[cfg(not(target_os = "macos"))]
    if shoud_preload_cuda {
        if let Err(e) = preload_cuda() {
            let win_err = windows::core::Error::from_win32();
            panic!("Failed to preload CUDA dylibs: {}; win_err: {}", e, win_err);
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = ort::init_from(find_onnxruntime_lib())
        .with_global_thread_pool(
            GlobalThreadPoolOptions::default()
                .with_inter_threads(2)
                .expect("Failed to set inter threads")
                .with_intra_threads(2)
                .expect("Failed to set intra threads")
                .with_thread_manager(pool::StdThreadManager::default())
                .expect("Failed to set thread manager"),
        )
        .commit()
        .expect("Failed to initialize onnxruntime");
    #[cfg(target_os = "macos")]
    let _ = ort::init()
        .with_global_thread_pool(
            GlobalThreadPoolOptions::default()
                .with_inter_threads(2)
                .expect("Failed to set inter threads")
                .with_intra_threads(2)
                .expect("Failed to set intra threads")
                .with_thread_manager(pool::StdThreadManager::default())
                .expect("Failed to set thread manager"),
        )
        .commit()
        .expect("Failed to initialize onnxruntime");
}

fn setup_execution_providers(mode: ExecutionMode) -> Vec<ExecutionProviderDispatch> {
    let mut providers = Vec::new();

    #[cfg(not(target_os = "macos"))]
    if matches!(mode, ExecutionMode::Fallback | ExecutionMode::Cuda) {
        let cuda = CUDAExecutionProvider::default()
            // .with_cuda_graph(true)
            .with_tf32(true)
            .with_conv_algorithm_search(
                ort::execution_providers::cuda::CuDNNConvAlgorithmSearch::Default,
            );
        if cuda.is_available().unwrap() {
            eprintln!("CUDA is available");
            let mut dispatcher = cuda.build();
            if mode == ExecutionMode::Cuda {
                dispatcher = dispatcher.error_on_failure();
            }
            providers.push(dispatcher);
        } else if mode == ExecutionMode::Cuda {
            panic!("CUDA is not available");
        }
    }

    #[cfg(target_os = "macos")]
    if matches!(mode, ExecutionMode::Fallback | ExecutionMode::CoreML) {
        // TODO: failed to detect CoreML:
        // Note that even though ONNX Runtime was compiled with CoreML, registration could still fail!
        eprintln!("CoreML is available");
        let mut dispatcher = CoreMLExecutionProvider::default().build();
        if mode == ExecutionMode::CoreML {
            dispatcher = dispatcher.error_on_failure();
        }
        providers.push(dispatcher);
    }

    #[cfg(not(target_os = "macos"))]
    if matches!(mode, ExecutionMode::Fallback | ExecutionMode::OpenVINO) {
        let openvino = OpenVINOExecutionProvider::default();
        if openvino.is_available().unwrap() {
            eprintln!("OpenVINO is available");
            let mut dispatcher = openvino.build();
            if mode == ExecutionMode::OpenVINO {
                dispatcher = dispatcher.error_on_failure();
            }
            providers.push(dispatcher);
        } else if mode == ExecutionMode::OpenVINO {
            panic!("OpenVINO is not available");
        }
    }

    #[cfg(target_os = "windows")]
    if matches!(mode, ExecutionMode::Fallback | ExecutionMode::DirectML) {
        let directml = DirectMLExecutionProvider::default();
        if directml.is_available().unwrap() {
            eprintln!("DirectML is available");
            providers.push(directml.build());
        } else if mode == ExecutionMode::DirectML {
            panic!("DirectML is not available");
        }
    }

    if matches!(mode, ExecutionMode::Fallback | ExecutionMode::XNNPack) {
        let xnnpack = XNNPACKExecutionProvider::default();
        if xnnpack.is_available().unwrap() {
            eprintln!("XNNPack is available");
            providers.push(xnnpack.build());
        } else if mode == ExecutionMode::XNNPack {
            panic!("XNNPack is not available");
        }
    }

    if matches!(mode, ExecutionMode::Fallback | ExecutionMode::Cpu) {
        eprintln!("CPU is available");
        providers.push(CPUExecutionProvider::default().build());
    }

    providers
}

#[derive(Debug, Default, EnumString, ValueEnum, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
enum ExecutionMode {
    #[default]
    Fallback,
    Cuda,
    #[value(name = "directml")]
    #[serde(rename = "directml")]
    DirectML,
    #[value(name = "xnnpack")]
    #[serde(rename = "xnnpack")]
    XNNPack,
    #[value(name = "coreml")]
    #[serde(rename = "coreml")]
    CoreML,
    #[value(name = "openvino")]
    #[serde(rename = "openvino")]
    OpenVINO,
    Cpu,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "fallback")]
    execution_mode: ExecutionMode,
    #[arg(long)]
    chunks_file: Option<Utf8PathBuf>,
    #[arg(long, default_value = "cache")]
    cache_dir: Utf8PathBuf,
    #[arg(short, long, default_value = "int8")]
    model_kind: ModelKind,
    #[arg(long)]
    enable_profiling: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Parser, Debug, PartialEq, Eq)]
enum Command {
    DumpChunks,
}

const TEST_DOC: &str = include_str!("../resources/《从前有座寻仙山》（全本+番外插入）.txt");

const CHUNKS_LIMIT: usize = BATCH_SIZE * 1000;

fn search_dir(dirs: &[Utf8PathBuf], name: &str) -> Option<Utf8PathBuf> {
    for dir in dirs {
        let path = dir.join(name);
        eprintln!("Searching for {} at {}", name, path);
        if path.exists() {
            eprintln!("Found {} at {}", name, path);
            return Some(path);
        }
    }
    None
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    set_hook(Box::new(|panic_info| {
        let payload_str = panic_info
            .payload()
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| {
                panic_info
                    .payload()
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
            })
            .unwrap_or("unknown error".to_string());
        eprintln!("Panic payload: {}", payload_str);
        eprintln!("Panic location: {:#?}", panic_info.location());
        if let Some(location) = panic_info.location() {
            if location.file().contains("thread") && payload_str.contains("unwrap") {
                return;
            }
        }
        std::process::exit(1);
    }));

    let args = Args::parse();
    let chunks = args
        .chunks_file
        .map(|path| {
            let file = std::fs::File::open(path).unwrap();
            let reader = std::io::BufReader::new(file);
            let chunks: Vec<String> = serde_json::from_reader(reader).unwrap();
            chunks
        })
        .unwrap_or(CHUNKS.iter().map(|s| s.to_string()).collect());
    #[cfg(not(feature = "static"))]
    setup_onnxruntime(matches!(
        args.execution_mode,
        ExecutionMode::Cuda | ExecutionMode::Fallback
    ));
    let current_exe =
        Utf8PathBuf::from_path_buf(std::env::current_exe().expect("Failed to get current exe"))
            .expect("Failed to convert current exe to Utf8PathBuf");
    let mut current_dir = current_exe
        .parent()
        .expect("Failed to get current exe parent")
        .to_path_buf();
    if current_dir
        .parent()
        .is_some_and(|p| p.file_name().is_some_and(|name| name == "target"))
    {
        current_dir.pop();
        current_dir.pop();
    }
    let cache_dir = camino::absolute_utf8(current_dir.join(args.cache_dir))
        .expect("Failed to absolute cache dir");
    let output_dir = current_dir.join("output");
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir).expect("Failed to create output dir");
    }
    let profiling_path = current_dir.join("model_profiling.json");
    eprintln!("Cache dir: {}", cache_dir);
    eprintln!("Model kind: {:?}", args.model_kind);
    let model = Model::new(
        get_embedding_model_info(args.model_kind),
        cache_dir.as_std_path(),
        if args.enable_profiling {
            eprintln!("enable profiling, output: {:?}", profiling_path);
            Some(profiling_path.as_std_path())
        } else {
            None
        },
    );
    let s: String = TEST_DOC.chars().take(20).collect();
    eprintln!("First 20 chars: {}", s);
    eprintln!(
        "Splitting {} chunks\n First 20 chunks: {:#?}",
        chunks.len(),
        chunks[..20].join("\n")
    );
    if let Some(command) = args.command {
        match command {
            Command::DumpChunks => {
                let mut file = tokio::fs::File::create(current_dir.join("chunks.json"))
                    .await
                    .unwrap()
                    .into_std()
                    .await;
                serde_json::to_writer_pretty(&mut file, &chunks.to_vec()).unwrap();
                eprintln!(
                    "Dumped {} chunks to {}",
                    chunks.len(),
                    cache_dir.join("chunks.json")
                );
                return Ok(());
            }
        }
    }

    let start = std::time::Instant::now();
    eprintln!("Embedding {} chunks", CHUNKS_LIMIT);
    let _ = model
        .embed_chunks(
            chunks[..CHUNKS_LIMIT.min(chunks.len())]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            args.execution_mode,
        )
        .await;
    let log = json!({
        "time": start.elapsed().as_secs_f32(),
        "chunks": CHUNKS_LIMIT,
        "execution_mode": args.execution_mode,
    });
    println!("{}", serde_json::to_string(&log).unwrap());
    #[cfg(windows)]
    std::process::exit(0);
    Ok(())
}
