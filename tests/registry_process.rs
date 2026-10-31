// Cargo also discovers this file as a standalone integration-test crate. Define
// the tests here, but instantiate them only inside registry's unit-test module
// so they retain access to private transaction and deletion-preparation APIs.
#[allow(unused_macros)]
macro_rules! registry_process_tests {
    () => {
        mod process_tests {
            use super::*;

            fn run_lock_helper(root: &Path, mode: &str) {
                use std::{
                    process::Command,
                    thread,
                    time::{Duration, Instant},
                };
                let mut child = Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "models::registry::tests::process_tests::cross_process_lock_helper",
                        "--ignored",
                        "--nocapture",
                    ])
                    .env("LMA_REGISTRY_LOCK_TEST_ROOT", root)
                    .env("LMA_REGISTRY_LOCK_TEST_MODE", mode)
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .unwrap();
                let deadline = Instant::now() + Duration::from_secs(10);
                loop {
                    if child.try_wait().unwrap().is_some() {
                        break;
                    }
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let output = child.wait_with_output().unwrap();
                        panic!("registry lock helper timed out: {output:?}");
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                let output = child.wait_with_output().unwrap();
                assert!(
                    output.status.success(),
                    "registry lock helper failed: {output:?}"
                );
            }

            #[test]
            #[ignore = "invoked by cross_process_lock_covers_read_modify_publish_and_remove_preparation"]
            fn cross_process_lock_helper() {
                let root = PathBuf::from(std::env::var_os("LMA_REGISTRY_LOCK_TEST_ROOT").unwrap());
                let entry = Entry {
                    path: fs::canonicalize(root.parent().unwrap().join("weights.gguf")).unwrap(),
                    projector: None,
                };
                let definition = crate::models::modelfile::parse("FROM fixture\nSYSTEM child").unwrap();
                if std::env::var("LMA_REGISTRY_LOCK_TEST_MODE").unwrap() == "publish" {
                    put_definition(&root, "child", entry, definition).unwrap();
                    return;
                }
                let before = fs::read(location(&root)).unwrap();
                assert_eq!(read(&root).unwrap()["defined:latest"], entry);
                assert!(super::definition(&root, "defined").unwrap().is_some());
                let results = [
                    put(&root, "child", entry.clone()).map(|_| ()),
                    put_definition(&root, "child", entry.clone(), definition).map(|_| ()),
                    copy(&root, "defined", "child").map(|_| ()),
                    delete(&root, "defined"),
                    remove_with(&root, "defined", &entry, |_| {
                        panic!("a busy registry must not prepare file deletion")
                    }),
                    change_full(&root, |_| panic!("a busy registry must not run updates")),
                ];
                for result in results {
                    assert!(result.unwrap_err().contains("registry busy"));
                }
                assert_eq!(fs::read(location(&root)).unwrap(), before);
                assert_eq!(fs::read_dir(&root).unwrap().count(), 2);
            }

            #[test]
            fn cross_process_lock_covers_read_modify_publish_and_remove_preparation() {
                let (_dir, root, entry, definition) = fixture();
                put_definition(&root, "defined", entry.clone(), definition).unwrap();
                change_full(&root, |entries| {
                    entries.insert(
                        "pending:latest".into(),
                        StoredEntry::new(entry.clone(), None),
                    );
                    // The child sees the old atomic document, but cannot publish over
                    // this in-flight update or start any deletion preparation.
                    run_lock_helper(&root, "busy");
                    Ok(())
                })
                .unwrap();
                remove_with(&root, "defined", &entry, |entries| {
                    assert!(entries.contains_key("pending:latest"));
                    run_lock_helper(&root, "busy");
                    Ok(())
                })
                .unwrap();
                run_lock_helper(&root, "publish");
                let entries = read(&root).unwrap();
                assert_eq!(entries.len(), 2);
                assert_eq!(entries["pending:latest"], entry);
                assert_eq!(entries["child:latest"], entry);
                assert!(super::definition(&root, "child").unwrap().is_some());
            }
        }
    };
}

#[allow(unused_imports)]
pub(crate) use registry_process_tests;
