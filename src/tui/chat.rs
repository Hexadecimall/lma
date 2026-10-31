use super::{
    state::{Action, App, Focus},
    stream,
    terminal::Screen,
    view,
};
use crate::{config::ServeConfig, desktop::Session};
use crossterm::event::{self, Event, KeyEventKind};
use std::{
    future::{Future, poll_fn},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    task::Poll,
    time::Duration,
};
use tokio::{sync::mpsc, task::JoinHandle};

#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Quit,
    Models,
}

#[derive(Default)]
struct RequestTask {
    handle: Option<JoinHandle<Result<serde_json::Value, String>>>,
    chunks: Option<mpsc::Receiver<String>>,
}
impl Drop for RequestTask {
    fn drop(&mut self) {
        if let Some(handle) = &self.handle {
            handle.abort();
        }
    }
}
impl RequestTask {
    async fn stop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
            let _ = handle.await;
        }
    }
    fn drain(&mut self, app: &mut App) {
        if let Some(chunks) = &mut self.chunks {
            // Bound work per frame so a fast producer cannot starve input/cancel.
            for _ in 0..64 {
                let Ok(text) = chunks.try_recv() else { break };
                app.partial.push_str(&text);
            }
        }
    }
    fn finished(&self) -> bool {
        self.handle.as_ref().is_some_and(JoinHandle::is_finished)
    }
}

/// All error paths return through the same cancellation and worker-teardown barrier.
pub async fn run(session: Session, config: ServeConfig, vision: bool) -> Result<Outcome, String> {
    let session = Arc::new(session);
    let mut task = RequestTask::default();
    let mut app = App::new(config, vision);
    let result = catch_ui(interact(&session, &mut task, &mut app)).await;
    finish(session, &mut task, result).await
}

async fn catch_ui<T>(future: impl Future<Output = Result<T, String>>) -> Result<T, String> {
    let mut future = std::pin::pin!(future);
    poll_fn(
        |cx| match catch_unwind(AssertUnwindSafe(|| future.as_mut().poll(cx))) {
            Ok(poll) => poll,
            Err(_) => Poll::Ready(Err(
                "terminal workspace panicked; native session is shutting down".into(),
            )),
        },
    )
    .await
}

async fn finish(
    session: Arc<Session>,
    task: &mut RequestTask,
    result: Result<Outcome, String>,
) -> Result<Outcome, String> {
    task.stop().await;
    let session = Arc::try_unwrap(session)
        .map_err(|_| "request still owns the native session after cancellation".to_string())?;
    let shutdown = session.shutdown().await;
    match (result, shutdown) {
        (Err(error), Err(shutdown)) => Err(format!("{error}; shutdown: {shutdown}")),
        (Err(error), _) | (_, Err(error)) => Err(error),
        (Ok(outcome), Ok(())) => Ok(outcome),
    }
}

async fn interact(
    session: &Arc<Session>,
    task: &mut RequestTask,
    app: &mut App,
) -> Result<Outcome, String> {
    let mut screen = Screen::enter()?;
    loop {
        task.drain(app);
        if task.finished() {
            let result = task
                .handle
                .take()
                .unwrap()
                .await
                .map_err(|e| format!("generation task failed: {e}"))
                .and_then(|result| result);
            task.drain(app);
            task.chunks = None;
            match result {
                Ok(usage) => {
                    let text = std::mem::take(&mut app.partial);
                    app.complete(text, &usage);
                }
                Err(error) => app.fail(&error),
            }
        }
        let action = step(&mut screen, app)?;
        match action {
            Action::Quit => return Ok(Outcome::Quit),
            Action::Models => return Ok(Outcome::Models),
            Action::Cancel => {
                task.stop().await;
                task.drain(app);
                task.chunks = None;
                app.cancel();
            }
            Action::Send => match app.request() {
                Ok((path, body)) => {
                    let session = Arc::clone(session);
                    let context_limit = app.tokens == 0;
                    let (send, receive) = mpsc::channel(32);
                    task.chunks = Some(receive);
                    task.handle = Some(tokio::spawn(async move {
                        stream::generate(&session, path, body, send, context_limit).await
                    }));
                    app.begin();
                }
                Err(error) => app.notice = error,
            },
            Action::None => {}
        }
    }
}

fn step(screen: &mut Screen, app: &mut App) -> Result<Action, String> {
    screen
        .draw(|frame| view::draw(frame, app))
        .map_err(|e| format!("terminal draw failed: {e}"))?;
    if !event::poll(Duration::from_millis(40)).map_err(|e| format!("terminal input failed: {e}"))? {
        return Ok(Action::None);
    }
    match event::read().map_err(|e| format!("terminal input failed: {e}"))? {
        Event::Key(key) if key.kind != KeyEventKind::Release => Ok(app.key(key)),
        Event::Paste(text)
            if app.busy.is_none() && app.focus == Focus::Composer && app.panel.is_none() =>
        {
            app.editor.insert(&text);
            Ok(Action::None)
        }
        Event::Mouse(mouse) => {
            app.mouse(mouse.kind);
            Ok(Action::None)
        }
        _ => Ok(Action::None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn error_and_panic_paths_await_worker_shutdown() {
        use std::sync::atomic::{AtomicBool, Ordering};
        for panic in [false, true] {
            let stopped = Arc::new(AtomicBool::new(false));
            let flag = stopped.clone();
            let (sender, mut receiver) = tokio::sync::mpsc::channel::<()>(1);
            let router = axum::Router::new().route(
                "/health",
                axum::routing::get(move || {
                    let _sender = sender.clone();
                    async move {
                        drop(_sender);
                        "ok"
                    }
                }),
            );
            let worker = tokio::spawn(async move {
                while receiver.recv().await.is_some() {}
                flag.store(true, Ordering::SeqCst);
            });
            let session = Arc::new(Session::new(router, worker));
            let borrowed = session.clone();
            let mut task = RequestTask {
                handle: Some(tokio::spawn(async move {
                    let _session = borrowed;
                    std::future::pending::<()>().await;
                    unreachable!()
                })),
                chunks: None,
            };
            let result = catch_ui(async {
                if panic {
                    panic!("simulated drawing panic");
                }
                Err::<Outcome, _>("simulated input error".into())
            })
            .await;
            assert!(finish(session, &mut task, result).await.is_err());
            assert!(stopped.load(Ordering::SeqCst));
            assert!(task.handle.is_none());
        }
    }

    #[tokio::test]
    async fn abort_joins_and_drops_request_resources() {
        struct Dropped(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let resource = Dropped(dropped.clone());
        let mut task = RequestTask {
            handle: Some(tokio::spawn(async move {
                let _resource = resource;
                std::future::pending::<()>().await;
                unreachable!()
            })),
            chunks: None,
        };
        tokio::task::yield_now().await;
        task.stop().await;
        assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
        assert!(task.handle.is_none());
    }
}
