use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
};

use ort::environment::{ThreadManager, ThreadWorker};

struct ThreadStats {
    active_threads: AtomicUsize,
}

impl Default for ThreadStats {
    fn default() -> Self {
        Self {
            active_threads: AtomicUsize::new(0),
        }
    }
}

pub struct StdThread {
    stats: Arc<ThreadStats>,
    join_handle: JoinHandle<()>,
}

impl StdThread {
    pub fn spawn(worker: ThreadWorker, stats: &Arc<ThreadStats>) -> Self {
        let join_handle = thread::spawn(move || worker.work());
        stats.active_threads.fetch_add(1, Ordering::AcqRel);
        Self {
            stats: Arc::clone(stats),
            join_handle,
        }
    }

    pub fn join(self) {
        let _ = self.join_handle.join();
        self.stats.active_threads.fetch_sub(1, Ordering::AcqRel);
    }
}

pub struct StdThreadManager {
    stats: Arc<ThreadStats>,
}

impl Default for StdThreadManager {
    fn default() -> Self {
        Self {
            stats: Arc::new(ThreadStats::default()),
        }
    }
}

impl ThreadManager for StdThreadManager {
    type Thread = StdThread;

    fn create(&mut self, worker: ThreadWorker) -> ort::Result<Self::Thread> {
        Ok(StdThread::spawn(worker, &self.stats))
    }

    fn join(thread: Self::Thread) -> ort::Result<()> {
        thread.join();
        Ok(())
    }
}
