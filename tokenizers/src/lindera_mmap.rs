//! INSTR-4902 fix: mmap-backed dict storage.
//!
//! Background: the original prewarm patch (#4902) loaded six Lindera
//! dictionaries into postmaster's heap. That state was inherited by every
//! forked Postgres process — leader and parallel workers alike — and slowed
//! down `launch_parallel_process!` because each worker forked from a 381 MB
//! anonymous heap and paid PTE-duplication / arena-fragmentation costs.
//!
//! This module replaces that with a one-time on-disk materialization.
//! Postmaster loads each embedded dict ONCE, writes the components to
//! disk in lindera's `FSDictionaryLoader` format, drops the Dictionary,
//! and calls `malloc_trim` to release the heap pages back to the OS.
//! Per-tokenizer Lazy statics then `Dictionary::load_from_path_with_options(
//! use_mmap = true)`, which returns `Data::Map(Arc<Mmap>)` — file-backed
//! pages shared via the OS page cache across every Postgres process.
//!
//! Net effect:
//!   - Postmaster's anon heap stays at baseline.
//!   - fork() PTE-duplication cost stays at baseline.
//!   - Workers' glibc arenas stay clean.
//!   - CJK queries still get zero cold-start because the mmap'd file is
//!     warm in page cache after the first access.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use lindera::dictionary::Dictionary;

/// The three Lindera dictionaries paradedb ships.
const DICTS: &[(&str, &str)] = &[
    ("cc-cedict", "embedded://cc-cedict"),
    ("ipadic", "embedded://ipadic"),
    ("ko-dic", "embedded://ko-dic"),
];

/// Names of the files lindera's FSDictionaryLoader expects to find inside
/// each dict directory. We materialize each from the corresponding Data
/// field of the in-memory Dictionary.
const DA_FILE: &str = "dict.da";
const VALS_FILE: &str = "dict.vals";
const WORDS_IDX_FILE: &str = "dict.wordsidx";
const WORDS_FILE: &str = "dict.words";
const MATRIX_FILE: &str = "matrix.mtx";
const CHAR_DEF_FILE: &str = "char_def.bin";
const UNK_FILE: &str = "unk.bin";
const METADATA_FILE: &str = "metadata.json";

/// Marker file written after a dict has been fully materialized.
/// We treat its existence as "dict on disk is complete and consistent."
const READY_MARKER: &str = ".materialized";

/// Default location for materialized dicts. Override with the
/// `PARADEDB_LINDERA_DICT_ROOT` env var (for tests).
pub fn default_dict_root() -> PathBuf {
    if let Ok(p) = std::env::var("PARADEDB_LINDERA_DICT_ROOT") {
        return PathBuf::from(p);
    }
    PathBuf::from("/var/lib/postgresql/lindera-dicts")
}

fn write_atomic(dest: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = dest.with_extension("tmp");
    {
        let mut f = fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        f.write_all(bytes)
            .with_context(|| format!("write {}", tmp.display()))?;
        f.sync_data().ok();
    }
    fs::rename(&tmp, dest)
        .with_context(|| format!("rename {} -> {}", tmp.display(), dest.display()))?;
    Ok(())
}

fn dict_is_ready(dir: &Path) -> bool {
    dir.join(READY_MARKER).exists()
        && dir.join(DA_FILE).exists()
        && dir.join(METADATA_FILE).exists()
}

/// Load `uri` via lindera's embedded loader, extract bytes from each
/// component, write them to `out_dir` in the layout `FSDictionaryLoader`
/// expects. Drops the in-memory Dictionary before returning so the only
/// remaining state is on disk.
fn materialize_one(uri: &str, out_dir: &Path) -> Result<()> {
    fs::create_dir_all(out_dir).with_context(|| format!("mkdir {}", out_dir.display()))?;

    let dict = lindera::dictionary::load_dictionary(uri)
        .map_err(|e| anyhow::anyhow!("load_dictionary({uri}) failed: {e}"))?;

    // PrefixDictionary components: raw byte slices (via Data: Deref<[u8]>).
    write_atomic(&out_dir.join(DA_FILE), &dict.prefix_dictionary.da.0)?;
    write_atomic(&out_dir.join(VALS_FILE), &dict.prefix_dictionary.vals_data)?;
    write_atomic(
        &out_dir.join(WORDS_IDX_FILE),
        &dict.prefix_dictionary.words_idx_data,
    )?;
    write_atomic(
        &out_dir.join(WORDS_FILE),
        &dict.prefix_dictionary.words_data,
    )?;

    // ConnectionCostMatrix: raw byte slice.
    write_atomic(
        &out_dir.join(MATRIX_FILE),
        &dict.connection_cost_matrix.costs_data,
    )?;

    // CharacterDefinition and UnknownDictionary are typed Rust structs;
    // lindera persists them as rkyv blobs.
    let char_def_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&dict.character_definition)
        .with_context(|| "rkyv serialize CharacterDefinition")?;
    write_atomic(&out_dir.join(CHAR_DEF_FILE), &char_def_bytes)?;

    let unk_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&dict.unknown_dictionary)
        .with_context(|| "rkyv serialize UnknownDictionary")?;
    write_atomic(&out_dir.join(UNK_FILE), &unk_bytes)?;

    let metadata_json =
        serde_json::to_string(&dict.metadata).with_context(|| "serde_json serialize Metadata")?;
    write_atomic(&out_dir.join(METADATA_FILE), metadata_json.as_bytes())?;

    // Ready marker: indicates dict is fully written.
    write_atomic(&out_dir.join(READY_MARKER), b"ok\n")?;

    // Explicitly drop the loaded dictionary so its heap allocations are
    // freed before we move on to the next language.
    drop(dict);

    Ok(())
}

/// Idempotent: materializes any missing dicts under `root`.
/// After all materialization, calls `malloc_trim` (Linux only) to encourage
/// glibc to release freed pages back to the OS.
pub fn ensure_materialized(root: &Path) -> Result<()> {
    fs::create_dir_all(root).with_context(|| format!("mkdir {}", root.display()))?;

    let mut materialized_any = false;
    for (name, uri) in DICTS {
        let dir = root.join(name);
        if dict_is_ready(&dir) {
            continue;
        }
        materialize_one(uri, &dir)?;
        materialized_any = true;
    }

    if materialized_any {
        // Release freed heap pages back to the kernel. On Linux glibc,
        // malloc_trim consolidates the heap and gives back what it can
        // via sbrk/munmap.
        #[cfg(target_os = "linux")]
        unsafe {
            extern "C" {
                fn malloc_trim(pad: libc::size_t) -> libc::c_int;
            }
            malloc_trim(0);
        }
    }

    Ok(())
}

/// mmap-load one of the materialized dicts. Returns a Dictionary whose
/// internal Data variants are `Data::Map(Arc<Mmap>)` — zero anon-heap cost.
pub fn load_mmap(root: &Path, lang: &str) -> Result<Arc<Dictionary>> {
    use lindera_dictionary::dictionary::Dictionary as DictType;
    let dir = root.join(lang);
    if !dict_is_ready(&dir) {
        anyhow::bail!(
            "lindera dict for {} not materialized; expected at {}",
            lang,
            dir.display()
        );
    }
    let dict = DictType::load_from_path_with_options(&dir, true)
        .map_err(|e| anyhow::anyhow!("load_from_path_with_options({}): {}", dir.display(), e))?;
    Ok(Arc::new(dict))
}
