#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>

namespace adbc_nif {

// Runs ADBC and Arrow releases on threads of its own, never on a BEAM
// scheduler.
//
// A resource destructor runs on whichever scheduler drops the last
// reference to the resource, a normal scheduler included. A driver may
// block in a release: DuckDB takes a connection's lock to release one of
// its statements, so a release that lands while a query runs on that
// connection waits for the query to end. On a normal scheduler that stalls
// every process and timer the scheduler holds. Destructors hand their
// release to `Releaser::run` instead.
//
// A worker thread starts whenever a job arrives and no worker is free, up
// to `max_workers`; past that, jobs wait in the queue for the next free
// worker. So a release stuck behind a long query does not hold up another
// one unless that many are stuck at once. A worker left idle for a while
// exits. What a release depends on travels with its job: a statement's
// job holds its connection, so the job that lets go of the last holder
// of a chain releases all of it, in order (see adbc_nif.cpp).
//
// If a job cannot be queued or no worker can start, the release runs
// inline, on the calling scheduler, as every release did before: a
// destructor must not throw. A job is moved, never copied, into the queue,
// so the caller is never left holding the last reference to what it
// releases once a worker has run it.
//
// A release that is queued or running when the VM halts would otherwise be
// cut short by exit(): a database stopped just before halt would not finish
// closing, and its release would run on while exit() destroys the driver's
// static objects. The NIF library's on-halt callback calls `drain`, which
// waits for the queue to empty, up to `halt_timeout`, before the VM exits.
//
// Workers are not joined, so unloading the NIF library while one is alive
// would unmap code it still runs. The library is never unloaded in a
// release, which loads it once per VM.
class Releaser {
public:
  static constexpr std::chrono::seconds halt_timeout{30};

  template <typename F> static void run(F release) {
    std::function<void()> job;

    try {
      job = std::function<void()>(std::move(release));
    } catch (...) {
      run_inline(release);
      return;
    }

    instance().push(job);
  }

  // Waits until no release is queued or running, or `timeout` passes.
  // Returns whether every release finished.
  static bool drain(std::chrono::milliseconds timeout) {
    auto &releaser = instance();
    std::unique_lock<std::mutex> lock(releaser.mutex);

    return releaser.idle.wait_for(lock, timeout, [&releaser] {
      return releaser.jobs.empty() && releaser.running == 0;
    });
  }

private:
  // Never destroyed: a worker may still be running when the VM exits and
  // static destructors run.
  static Releaser &instance() {
    static Releaser *releaser = new Releaser();
    return *releaser;
  }

  template <typename F> static void run_inline(F &release) {
    try {
      release();
    } catch (...) {
    }
  }

  void push(std::function<void()> &job) {
    std::unique_lock<std::mutex> lock(mutex);

    try {
      jobs.push_back(std::move(job));
    } catch (...) {
      lock.unlock();
      run_inline(job);
      return;
    }

    if (waiting >= jobs.size()) {
      ready.notify_one();
      return;
    }

    if (workers < max_workers) {
      try {
        std::thread([this] { work(); }).detach();
        workers++;
        return;
      } catch (...) {
      }
    }

    if (workers == 0) {
      auto own = std::move(jobs.back());
      jobs.pop_back();
      lock.unlock();
      run_inline(own);
    } else if (waiting > 0) {
      ready.notify_one();
    }
  }

  void work() {
    std::unique_lock<std::mutex> lock(mutex);

    for (;;) {
      while (jobs.empty()) {
        waiting++;
        auto woken = ready.wait_for(lock, idle_timeout,
                                    [this] { return !jobs.empty(); });
        waiting--;

        if (!woken) {
          workers--;
          return;
        }
      }

      auto job = std::move(jobs.front());
      jobs.pop_front();
      running++;
      lock.unlock();

      try {
        job();
      } catch (...) {
      }

      job = nullptr;
      lock.lock();
      running--;

      if (jobs.empty() && running == 0) {
        idle.notify_all();
      }
    }
  }

  static constexpr size_t max_workers = 16;
  static constexpr std::chrono::seconds idle_timeout{30};

  std::mutex mutex;
  std::condition_variable ready;
  std::condition_variable idle;
  std::deque<std::function<void()>> jobs;
  size_t waiting = 0;
  size_t workers = 0;
  size_t running = 0;
};

} // namespace adbc_nif
