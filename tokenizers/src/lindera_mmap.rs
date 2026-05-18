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
use lindera_dictionary::decompress::{Algorithm, CompressedData};

/// Wrap raw bytes in lindera's `CompressedData` envelope with `Algorithm::Raw`
/// so the file loader (compiled with the `compress` feature) accepts them
/// directly without trying to decompress. The wrapper is a small rkyv blob;
/// no actual compression happens here.
fn wrap_raw(bytes: &[u8]) -> Result<rkyv::util::AlignedVec> {
    let cd = CompressedData::new(Algorithm::Raw, bytes.to_vec());
    let aligned = rkyv::to_bytes::<rkyv::rancor::Error>(&cd)
        .with_context(|| "rkyv serialize CompressedData{Algorithm::Raw}")?;
    Ok(aligned)
}

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

    // PrefixDictionary components: raw byte slices wrapped in CompressedData{Raw}
    // so lindera's `compress`-aware loader can consume them on read.
    write_atomic(
        &out_dir.join(DA_FILE),
        &wrap_raw(&dict.prefix_dictionary.da.0)?,
    )?;
    write_atomic(
        &out_dir.join(VALS_FILE),
        &wrap_raw(&dict.prefix_dictionary.vals_data)?,
    )?;
    write_atomic(
        &out_dir.join(WORDS_IDX_FILE),
        &wrap_raw(&dict.prefix_dictionary.words_idx_data)?,
    )?;
    write_atomic(
        &out_dir.join(WORDS_FILE),
        &wrap_raw(&dict.prefix_dictionary.words_data)?,
    )?;

    // ConnectionCostMatrix: same Raw-wrapped envelope.
    write_atomic(
        &out_dir.join(MATRIX_FILE),
        &wrap_raw(&dict.connection_cost_matrix.costs_data)?,
    )?;

    // CharacterDefinition and UnknownDictionary: rkyv-serialize the struct,
    // then wrap the resulting bytes in CompressedData{Raw}. The loader (with
    // compress feature) will unwrap to get our rkyv bytes, then deserialize
    // them with the existing rkyv-of-struct path.
    let char_def_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&dict.character_definition)
        .with_context(|| "rkyv serialize CharacterDefinition")?;
    write_atomic(&out_dir.join(CHAR_DEF_FILE), &wrap_raw(&char_def_bytes)?)?;

    let unk_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&dict.unknown_dictionary)
        .with_context(|| "rkyv serialize UnknownDictionary")?;
    write_atomic(&out_dir.join(UNK_FILE), &wrap_raw(&unk_bytes)?)?;

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

/// Materialize dicts in a forked subprocess so the parent's heap stays
/// pristine. The child runs `ensure_materialized` and exits via `_exit`
/// (skipping atexit handlers). All the dict-load heap state is freed back
/// to the OS when the child exits, so the parent (postmaster) is never
/// perturbed.
///
/// Idempotent: short-circuits if all dicts are already on disk; no fork.
#[cfg(target_os = "linux")]
pub fn ensure_materialized_via_subprocess(root: &Path) -> Result<()> {
    // Fast path — all dicts already materialized, no fork required.
    let all_ready = DICTS
        .iter()
        .all(|(name, _)| dict_is_ready(&root.join(name)));
    if all_ready {
        return Ok(());
    }

    // Make the root dir up front so the child doesn't need to mkdir
    // (avoids race if the child's first allocation is the path string).
    fs::create_dir_all(root).with_context(|| format!("mkdir {}", root.display()))?;

    let pid = unsafe { libc::fork() };
    if pid < 0 {
        anyhow::bail!(
            "fork() failed for lindera dict materialization: {}",
            std::io::Error::last_os_error()
        );
    }
    if pid == 0 {
        // Child: run the materializer, then exit via _exit so we skip
        // atexit handlers that the parent (postmaster) may have registered.
        // We use eprintln to surface errors to the PG log — the child shares
        // the parent's stderr fd at this point.
        let code = match ensure_materialized(root) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("[pg_search INSTR-4902] dict materialization (child) failed: {e}");
                1
            }
        };
        unsafe {
            libc::_exit(code);
        }
    }

    // Parent: wait for the child to complete. The child should be quick
    // (~1-2 sec total for all three dicts on modern hardware), so a blocking
    // wait is fine — _PG_init is a one-time cost at postmaster startup.
    let mut status: libc::c_int = 0;
    let waited = unsafe { libc::waitpid(pid, &mut status, 0) };
    if waited < 0 {
        anyhow::bail!(
            "waitpid for dict materialization subprocess failed: {}",
            std::io::Error::last_os_error()
        );
    }
    if unsafe { libc::WIFSIGNALED(status) } {
        anyhow::bail!(
            "dict materialization subprocess killed by signal {}",
            unsafe { libc::WTERMSIG(status) }
        );
    }
    if !unsafe { libc::WIFEXITED(status) } {
        anyhow::bail!("dict materialization subprocess terminated abnormally");
    }
    let exit_code = unsafe { libc::WEXITSTATUS(status) };
    if exit_code != 0 {
        anyhow::bail!("dict materialization subprocess exited with code {exit_code}");
    }
    Ok(())
}

/// Fallback for non-Linux: just call the in-process materializer.
/// (macOS doesn't have the same fork+postmaster pattern in our test;
/// production is Linux-only for paradedb CI.)
#[cfg(not(target_os = "linux"))]
pub fn ensure_materialized_via_subprocess(root: &Path) -> Result<()> {
    ensure_materialized(root)
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
