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
// A worker thread starts whenever a job arrives and no worker is free, so
// a release stuck behind a long query never holds up another one. A worker
// left idle for a while exits. A job keeps the resources its release
// depends on referenced until it has run, which keeps a statement released
// before its connection, and a connection before its database.
class Releaser {
public:
  static void run(std::function<void()> job) {
    instance().push(std::move(job));
  }

private:
  // Never destroyed: a worker may still be running when the VM exits and
  // static destructors run.
  static Releaser &instance() {
    static Releaser *releaser = new Releaser();
    return *releaser;
  }

  // If no worker can be started, the job runs here, as a release did
  // before: a destructor must not throw.
  void push(std::function<void()> job) {
    std::unique_lock<std::mutex> lock(mutex);

    if (waiting > jobs.size()) {
      jobs.push_back(std::move(job));
      ready.notify_one();
      return;
    }

    try {
      jobs.push_back(std::move(job));
      std::thread([this] { work(); }).detach();
    } catch (...) {
      if (jobs.empty()) {
        return;
      }

      auto inline_job = std::move(jobs.back());
      jobs.pop_back();
      lock.unlock();

      try {
        inline_job();
      } catch (...) {
      }
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
          return;
        }
      }

      auto job = std::move(jobs.front());
      jobs.pop_front();
      lock.unlock();

      try {
        job();
      } catch (...) {
      }

      job = nullptr;
      lock.lock();
    }
  }

  static constexpr std::chrono::seconds idle_timeout{30};

  std::mutex mutex;
  std::condition_variable ready;
  std::deque<std::function<void()>> jobs;
  size_t waiting = 0;
};

} // namespace adbc_nif
