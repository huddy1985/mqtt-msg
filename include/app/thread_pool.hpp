#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace app {

class ThreadPool {
public:
    explicit ThreadPool(std::size_t n)
        : stop_(false) {
        if (n == 0) n = 1;
        workers_.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            workers_.emplace_back([this] {
                for (;;) {
                    Task task;
                    {
                        std::unique_lock<std::mutex> lk(m_);
                        cv_.wait(lk, [this] { return stop_ || !q_.empty(); });
                        if (stop_ && q_.empty()) return;
                        task = std::move(q_.front());
                        q_.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto &t : workers_) if (t.joinable()) t.join();
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using R = std::invoke_result_t<F, Args...>;
        auto p = std::make_shared<std::packaged_task<R()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        {
            std::lock_guard<std::mutex> lk(m_);
            if (stop_) throw std::runtime_error("ThreadPool stopped");
            q_.emplace([p]{ (*p)(); });
        }
        cv_.notify_one();
        return p->get_future();
    }

private:
    using Task = std::function<void()>;
    std::vector<std::thread> workers_;
    std::queue<Task> q_;
    std::mutex m_;
    std::condition_variable cv_;
    bool stop_;
};

} // namespace app
