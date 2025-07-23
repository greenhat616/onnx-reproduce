use unicode_segmentation::UnicodeSegmentation;

const TEST_DOC: &str = include_str!("resources/《从前有座寻仙山》（全本+番外插入）.txt");

fn get_test_chunks() -> Vec<String> {
    TEST_DOC
        .split(&[' ', '\n', '\r', '\t'])
        .filter(|line| !line.trim().is_empty())
        .flat_map(|line| line.unicode_words())
        .map(|word| word.to_string())
        .collect()
}

fn main() {
    let chunks = get_test_chunks();

    // 生成 Rust 源码
    let out = format!(
        "pub static CHUNKS: [&str; {}] = [{}];",
        chunks.len(),
        chunks
            .iter()
            .map(|s| format!("r#\"{}\"#", s))
            .collect::<Vec<_>>()
            .join(", ")
    );
    std::fs::write("src/chunks.rs", out).unwrap();

    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos"))]
    println!("cargo:rustc-link-arg=-fapple-link-rtlib");
}
