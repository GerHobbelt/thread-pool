#pragma once

/**
 * @file BS_thread_pool.hpp
 * @author Barak Shoshany (baraksh@gmail.com) (http://baraksh.com)
 * @version 3.3.0
 * @date 2022-08-03
 * @copyright Copyright (c) 2022 Barak Shoshany. Licensed under the MIT license. If you found this project useful, please consider starring it on GitHub! If you use this library in software of any kind, please provide a link to the GitHub repository https://github.com/bshoshany/thread-pool in the source code and documentation. If you use this library in published research, please cite it as follows: Barak Shoshany, "A C++17 Thread Pool for High-Performance Scientific Computing", doi:10.5281/zenodo.4742687, arXiv:2105.00613 (May 2021)
 *
 * @brief BS::thread_pool: a fast, lightweight, and easy-to-use C++17 thread pool library. This header file contains the entire library, including the main BS::thread_pool class and the helper classes BS::multi_future, BS::blocks, BS:synced_stream, and BS::timer.
 */

#define BS_THREAD_POOL_VERSION "v3.3.0 (2022-08-03)"

#include <atomic>             // std::atomic
#include <chrono>             // std::chrono
#include <condition_variable> // std::condition_variable
#include <exception>          // std::current_exception
#include <functional>         // std::bind, std::function, std::invoke
#include <future>             // std::future, std::promise
#include <iostream>           // std::cout, std::endl, std::flush, std::ostream
#include <memory>             // std::make_shared, std::make_unique, std::shared_ptr, std::unique_ptr
#include <mutex>              // std::mutex, std::scoped_lock, std::unique_lock
#include <queue>              // std::queue
#include <thread>             // std::thread
#include <type_traits>        // std::common_type_t, std::conditional_t, std::decay_t, std::invoke_result_t, std::is_void_v
#include <utility>            // std::forward, std::move, std::swap
#include <vector>             // std::vector
#include <stdexcept>
#if defined(_MSC_VER)
// see also https://docs.microsoft.com/en-us/cpp/cpp/try-except-statement?view=msvc-170 et al.
#include <winnt.h>
#include <excpt.h>
#pragma warning(push)
#pragma warning(disable: 4005) // warning C4005: macro redefinition
#include <ntstatus.h> // STATUS_POSSIBLE_DEADLOCK
#pragma warning(pop)
#endif

namespace BS
{
/**
 * @brief A convenient shorthand for the type of std::thread::hardware_concurrency(). Should evaluate to unsigned int.
 */
using concurrency_t = std::invoke_result_t<decltype(std::thread::hardware_concurrency)>;

// ============================================================================================= //
//                                    Begin class multi_future                                   //

/**
 * @brief A helper class to facilitate waiting for and/or getting the results of multiple futures at once.
 *
 * @tparam T The return type of the futures.
 */
template <typename T>
class [[nodiscard]] multi_future
{
public:
    /**
     * @brief Construct a multi_future object with the given number of futures.
     *
     * @param num_futures_ The desired number of futures to store.
     */
    multi_future(const size_t num_futures_ = 0) : futures(num_futures_) {}

    /**
     * @brief Get the results from all the futures stored in this multi_future object, rethrowing any stored exceptions.
     *
     * @return If the futures return void, this function returns void as well. Otherwise, it returns a vector containing the results.
     */
    [[nodiscard]] std::conditional_t<std::is_void_v<T>, void, std::vector<T>> get()
    {
        if constexpr (std::is_void_v<T>)
        {
            for (size_t i = 0; i < futures.size(); ++i)
                futures[i].get();
            return;
        }
        else
        {
            std::vector<T> results(futures.size());
            for (size_t i = 0; i < futures.size(); ++i)
                results[i] = futures[i].get();
            return results;
        }
    }

    /**
     * @brief Get a reference to one of the futures stored in this multi_future object.
     *
     * @param i The index of the desired future.
     * @return The future.
     */
    [[nodiscard]] std::future<T>& operator[](const size_t i)
    {
        return futures[i];
    }

    /**
     * @brief Append a future to this multi_future object.
     *
     * @param future The future to append.
     */
    void push_back(std::future<T> future)
    {
        futures.push_back(std::move(future));
    }

    /**
     * @brief Get the number of futures stored in this multi_future object.
     *
     * @return The number of futures.
     */
    [[nodiscard]] size_t size() const
    {
        return futures.size();
    }

    /**
     * @brief Wait for all the futures stored in this multi_future object.
     */
    void wait() const
    {
        for (size_t i = 0; i < futures.size(); ++i)
            futures[i].wait();
    }

private:
    /**
     * @brief A vector to store the futures.
     */
    std::vector<std::future<T>> futures;
};

//                                     End class multi_future                                    //
// ============================================================================================= //

// ============================================================================================= //
//                                       Begin class blocks                                      //

/**
 * @brief A helper class to divide a range into blocks. Used by parallelize_loop() and push_loop().
 *
 * @tparam T1 The type of the first index in the range. Should be a signed or unsigned integer.
 * @tparam T2 The type of the index after the last index in the range. Should be a signed or unsigned integer. If T1 is not the same as T2, a common type will be automatically inferred.
 * @tparam T The common type of T1 and T2.
 */
template <typename T1, typename T2, typename T = std::common_type_t<T1, T2>>
class [[nodiscard]] blocks
{
public:
    /**
     * @brief Construct a blocks object with the given specifications.
     *
     * @param first_index_ The first index in the range.
     * @param index_after_last_ The index after the last index in the range.
     * @param num_blocks_ The desired number of blocks to divide the range into.
     */
    blocks(const T1 first_index_, const T2 index_after_last_, const size_t num_blocks_) : first_index(static_cast<T>(first_index_)), index_after_last(static_cast<T>(index_after_last_)), num_blocks(num_blocks_)
    {
        if (index_after_last < first_index)
            std::swap(index_after_last, first_index);
        total_size = static_cast<size_t>(index_after_last - first_index);
        block_size = static_cast<size_t>(total_size / num_blocks);
        if (block_size == 0)
        {
            block_size = 1;
            num_blocks = (total_size > 1) ? total_size : 1;
        }
    }

    /**
     * @brief Get the first index of a block.
     *
     * @param i The block number.
     * @return The first index.
     */
    [[nodiscard]] T start(const size_t i) const
    {
        return static_cast<T>(i * block_size) + first_index;
    }

    /**
     * @brief Get the index after the last index of a block.
     *
     * @param i The block number.
     * @return The index after the last index.
     */
    [[nodiscard]] T end(const size_t i) const
    {
        return (i == num_blocks - 1) ? index_after_last : (static_cast<T>((i + 1) * block_size) + first_index);
    }

    /**
     * @brief Get the number of blocks. Note that this may be different than the desired number of blocks that was passed to the constructor.
     *
     * @return The number of blocks.
     */
    [[nodiscard]] size_t get_num_blocks() const
    {
        return num_blocks;
    }

    /**
     * @brief Get the total number of indices in the range.
     *
     * @return The total number of indices.
     */
    [[nodiscard]] size_t get_total_size() const
    {
        return total_size;
    }

private:
    /**
     * @brief The size of each block (except possibly the last block).
     */
    size_t block_size = 0;

    /**
     * @brief The first index in the range.
     */
    T first_index = 0;

    /**
     * @brief The index after the last index in the range.
     */
    T index_after_last = 0;

    /**
     * @brief The number of blocks.
     */
    size_t num_blocks = 0;

    /**
     * @brief The total number of indices in the range.
     */
    size_t total_size = 0;
};

//                                        End class blocks                                       //
// ============================================================================================= //

// ============================================================================================= //
//                                    Begin class thread_pool                                    //

/**
 * @brief A fast, lightweight, and easy-to-use C++17 thread pool class.
 */
class [[nodiscard]] thread_pool
{
public:
    // ============================
    // Constructors and destructors
    // ============================

    /**
     * @brief Construct a new thread pool.
     *
     * @param thread_count_ The number of threads to use. The default value is the total number of hardware threads available, as reported by the implementation. This is usually determined by the number of cores in the CPU. If a core is hyperthreaded, it will count as two threads.
     */
    thread_pool(const concurrency_t thread_count_ = 0) : thread_count(determine_thread_count(thread_count_)), threads(std::make_unique<std::thread[]>(determine_thread_count(thread_count_)))
    {
        create_threads();
    }

    /**
     * @brief Destruct the thread pool. Waits for all tasks to complete, then destroys all threads. Note that if the pool is paused, then any tasks still in the queue will never be executed.
     */
    ~thread_pool()
    {
        wait_for_tasks();
        destroy_threads();
    }

    // =======================
    // Public member functions
    // =======================

    /**
     * @brief Get the number of tasks currently waiting in the queue to be executed by the threads.
     *
     * @return The number of queued tasks.
     */
    [[nodiscard]] size_t get_tasks_queued() const
    {
        const std::scoped_lock tasks_lock(tasks_mutex);
        return tasks.size();
    }

    /**
     * @brief Get the number of tasks currently being executed by the threads.
     *
     * @return The number of running tasks.
     */
    [[nodiscard]] size_t get_tasks_running() const
    {
        const std::scoped_lock tasks_lock(tasks_mutex);
        return tasks_total - tasks.size();
    }

    /**
     * @brief Get the total number of unfinished tasks: either still in the queue, or running in a thread. Note that get_tasks_total() == get_tasks_queued() + get_tasks_running().
     *
     * @return The total number of tasks.
     */
    [[nodiscard]] size_t get_tasks_total() const
    {
        return tasks_total;
    }

    /**
     * @brief Get the number of threads in the pool.
     *
     * @return The number of threads.
     */
    [[nodiscard]] concurrency_t get_thread_count() const
    {
        return thread_count;
    }

    /**
     * @brief Check whether the pool is currently paused.
     *
     * @return true if the pool is paused, false if it is not paused.
     */
    [[nodiscard]] bool is_paused() const
    {
        return paused;
    }

    /**
     * @brief Parallelize a loop by automatically splitting it into blocks and submitting each block separately to the queue. Returns a multi_future object that contains the futures for all of the blocks.
     *
     * @tparam F The type of the function to loop through.
     * @tparam T1 The type of the first index in the loop. Should be a signed or unsigned integer.
     * @tparam T2 The type of the index after the last index in the loop. Should be a signed or unsigned integer. If T1 is not the same as T2, a common type will be automatically inferred.
     * @tparam T The common type of T1 and T2.
     * @tparam R The return value of the loop function F (can be void).
     * @param first_index The first index in the loop.
     * @param index_after_last The index after the last index in the loop. The loop will iterate from first_index to (index_after_last - 1) inclusive. In other words, it will be equivalent to "for (T i = first_index; i < index_after_last; ++i)". Note that if index_after_last == first_index, no blocks will be submitted.
     * @param loop The function to loop through. Will be called once per block. Should take exactly two arguments: the first index in the block and the index after the last index in the block. loop(start, end) should typically involve a loop of the form "for (T i = start; i < end; ++i)".
     * @param num_blocks The maximum number of blocks to split the loop into. The default is to use the number of threads in the pool.
     * @return A multi_future object that can be used to wait for all the blocks to finish. If the loop function returns a value, the multi_future object can also be used to obtain the values returned by each block.
     */
    template <typename F, typename T1, typename T2, typename T = std::common_type_t<T1, T2>, typename R = std::invoke_result_t<std::decay_t<F>, T, T>>
    [[nodiscard]] multi_future<R> parallelize_loop(const T1 first_index, const T2 index_after_last, F&& loop, const size_t num_blocks = 0)
    {
        blocks blks(first_index, index_after_last, num_blocks ? num_blocks : thread_count);
        if (blks.get_total_size() > 0)
        {
            multi_future<R> mf(blks.get_num_blocks());
            for (size_t i = 0; i < blks.get_num_blocks(); ++i)
                mf[i] = submit(std::forward<F>(loop), blks.start(i), blks.end(i));
            return mf;
        }
        else
        {
            return multi_future<R>();
        }
    }

    /**
     * @brief Parallelize a loop by automatically splitting it into blocks and submitting each block separately to the queue. Returns a multi_future object that contains the futures for all of the blocks. This overload is used for the special case where the first index is 0.
     *
     * @tparam F The type of the function to loop through.
     * @tparam T The type of the loop indices. Should be a signed or unsigned integer.
     * @tparam R The return value of the loop function F (can be void).
     * @param index_after_last The index after the last index in the loop. The loop will iterate from 0 to (index_after_last - 1) inclusive. In other words, it will be equivalent to "for (T i = 0; i < index_after_last; ++i)". Note that if index_after_last == 0, no blocks will be submitted.
     * @param loop The function to loop through. Will be called once per block. Should take exactly two arguments: the first index in the block and the index after the last index in the block. loop(start, end) should typically involve a loop of the form "for (T i = start; i < end; ++i)".
     * @param num_blocks The maximum number of blocks to split the loop into. The default is to use the number of threads in the pool.
     * @return A multi_future object that can be used to wait for all the blocks to finish. If the loop function returns a value, the multi_future object can also be used to obtain the values returned by each block.
     */
    template <typename F, typename T, typename R = std::invoke_result_t<std::decay_t<F>, T, T>>
    [[nodiscard]] multi_future<R> parallelize_loop(const T index_after_last, F&& loop, const size_t num_blocks = 0)
    {
        return parallelize_loop(0, index_after_last, std::forward<F>(loop), num_blocks);
    }

    /**
     * @brief Pause the pool. The workers will temporarily stop retrieving new tasks out of the queue, although any tasks already executed will keep running until they are finished.
     */
    void pause()
    {
        paused = true;
    }

    /**
     * @brief Parallelize a loop by automatically splitting it into blocks and submitting each block separately to the queue. Does not return a multi_future, so the user must use wait_for_tasks() or some other method to ensure that the loop finishes executing, otherwise bad things will happen.
     *
     * @tparam F The type of the function to loop through.
     * @tparam T1 The type of the first index in the loop. Should be a signed or unsigned integer.
     * @tparam T2 The type of the index after the last index in the loop. Should be a signed or unsigned integer. If T1 is not the same as T2, a common type will be automatically inferred.
     * @tparam T The common type of T1 and T2.
     * @param first_index The first index in the loop.
     * @param index_after_last The index after the last index in the loop. The loop will iterate from first_index to (index_after_last - 1) inclusive. In other words, it will be equivalent to "for (T i = first_index; i < index_after_last; ++i)". Note that if index_after_last == first_index, no blocks will be submitted.
     * @param loop The function to loop through. Will be called once per block. Should take exactly two arguments: the first index in the block and the index after the last index in the block. loop(start, end) should typically involve a loop of the form "for (T i = start; i < end; ++i)".
     * @param num_blocks The maximum number of blocks to split the loop into. The default is to use the number of threads in the pool.
     */
    template <typename F, typename T1, typename T2, typename T = std::common_type_t<T1, T2>>
    void push_loop(const T1 first_index, const T2 index_after_last, F&& loop, const size_t num_blocks = 0)
    {
        blocks blks(first_index, index_after_last, num_blocks ? num_blocks : thread_count);
        if (blks.get_total_size() > 0)
        {
            for (size_t i = 0; i < blks.get_num_blocks(); ++i)
                push_task(std::forward<F>(loop), blks.start(i), blks.end(i));
        }
    }

    /**
     * @brief Parallelize a loop by automatically splitting it into blocks and submitting each block separately to the queue. Does not return a multi_future, so the user must use wait_for_tasks() or some other method to ensure that the loop finishes executing, otherwise bad things will happen. This overload is used for the special case where the first index is 0.
     *
     * @tparam F The type of the function to loop through.
     * @tparam T The type of the loop indices. Should be a signed or unsigned integer.
     * @param index_after_last The index after the last index in the loop. The loop will iterate from 0 to (index_after_last - 1) inclusive. In other words, it will be equivalent to "for (T i = 0; i < index_after_last; ++i)". Note that if index_after_last == 0, no blocks will be submitted.
     * @param loop The function to loop through. Will be called once per block. Should take exactly two arguments: the first index in the block and the index after the last index in the block. loop(start, end) should typically involve a loop of the form "for (T i = start; i < end; ++i)".
     * @param num_blocks The maximum number of blocks to split the loop into. The default is to use the number of threads in the pool.
     */
    template <typename F, typename T>
    void push_loop(const T index_after_last, F&& loop, const size_t num_blocks = 0)
    {
        push_loop(0, index_after_last, std::forward<F>(loop), num_blocks);
    }

    /**
     * @brief Push a function with zero or more arguments, but no return value, into the task queue. Does not return a future, so the user must use wait_for_tasks() or some other method to ensure that the task finishes executing, otherwise bad things will happen.
     *
     * @tparam F The type of the function.
     * @tparam A The types of the arguments.
     * @param task The function to push.
     * @param args The zero or more arguments to pass to the function. Note that if the task is a class member function, the first argument must be a pointer to the object, i.e. &object (or this), followed by the actual arguments.
     */
    template <typename F, typename... A>
    void push_task(F&& task, A&&... args)
    {
        std::function<void()> task_function = std::bind(std::forward<F>(task), std::forward<A>(args)...);
        {
            const std::scoped_lock tasks_lock(tasks_mutex);
            tasks.push(task_function);
        }
        ++tasks_total;
        task_available_cv.notify_one();
    }

    /**
     * @brief Reset the number of threads in the pool. Waits for all currently running tasks to be completed, then destroys all threads in the pool and creates a new thread pool with the new number of threads. Any tasks that were waiting in the queue before the pool was reset will then be executed by the new threads. If the pool was paused before resetting it, the new pool will be paused as well.
     *
     * @param thread_count_ The number of threads to use. The default value is the total number of hardware threads available, as reported by the implementation. This is usually determined by the number of cores in the CPU. If a core is hyperthreaded, it will count as two threads.
     */
    void reset(const concurrency_t thread_count_ = 0)
    {
        const bool was_paused = paused;
        paused = true;
        wait_for_tasks();
        destroy_threads();
        thread_count = determine_thread_count(thread_count_);
        threads = std::make_unique<std::thread[]>(thread_count);
        paused = was_paused;
        create_threads();
    }

    /**
     * @brief Submit a function with zero or more arguments into the task queue. If the function has a return value, get a future for the eventual returned value. If the function has no return value, get an std::future<void> which can be used to wait until the task finishes.
     *
     * @tparam F The type of the function.
     * @tparam A The types of the zero or more arguments to pass to the function.
     * @tparam R The return type of the function (can be void).
     * @param task The function to submit.
     * @param args The zero or more arguments to pass to the function. Note that if the task is a class member function, the first argument must be a pointer to the object, i.e. &object (or this), followed by the actual arguments.
     * @return A future to be used later to wait for the function to finish executing and/or obtain its returned value if it has one.
     */
    template <typename F, typename... A, typename R = std::invoke_result_t<std::decay_t<F>, std::decay_t<A>...>>
    [[nodiscard]] std::future<R> submit(F&& task, A&&... args)
    {
        std::function<R()> task_function = std::bind(std::forward<F>(task), std::forward<A>(args)...);
        std::shared_ptr<std::promise<R>> task_promise = std::make_shared<std::promise<R>>();
        push_task(
            [task_function, task_promise]
            {
                try
                {
                    if constexpr (std::is_void_v<R>)
                    {
                        std::invoke(task_function);
                        task_promise->set_value();
                    }
                    else
                    {
                        task_promise->set_value(std::invoke(task_function));
                    }
                }
                catch (...)
                {
                    try
                    {
                        task_promise->set_exception(std::current_exception());
                    }
                    catch (...)
                    {
                    }
                }
            });
        return task_promise->get_future();
    }

    /**
     * @brief Unpause the pool. The workers will resume retrieving new tasks out of the queue.
     */
    void unpause()
    {
        paused = false;
    }

    /**
     * @brief Wait for tasks to be completed. Normally, this function waits for all tasks, both those that are currently running in the threads and those that are still waiting in the queue. However, if the pool is paused, this function only waits for the currently running tasks (otherwise it would wait forever). Note: To wait for just one specific task, use submit() instead, and call the wait() member function of the generated future.
     */
    void wait_for_tasks()
    {
		int sleep_factor = 1;
        cv.notify_all();
        while (true)
        {
			if (alive_threads_total == 0)
				break;

			// don't check the task queue when we've already shut down the pool. Just terminate those threads as these numbers
			// won't be changing anymore anyway.
			if (running)
			{
				if (!paused)
				{
					if (tasks_total == 0)
						break;
				} else
				{
					if (get_tasks_running() == 0)
						break;
				}
			}
			else
			{
				bool go = true;
				for (ui32 i = 0; i < thread_count; i++)
				{
					// This is the real check that also detects when a thread has been swiftly terminated
					// WITHOUT AN EXCEPTION OR ERROR when the application has invoked `exit()` to
					// terminate the current run.
					//
					// https://stackoverflow.com/questions/33943601/check-if-stdthread-is-still-running
					//
					// While some say thread.joinable() is not dependable when used once, we don't mind
					// as we'll be looping through here anyway until all threads check out as such.
					//
					// Which is also why we keep firing those `cv.notify_all()` notifications further below:
					// together they guarantee the threads will be cleaned up properly, assuming none of
					// them is stuck forever in a task() they just happen to be executing...
					bool terminated = threads[i].joinable();
					go &= terminated;
				}
				if (go)
					break;

				if (sleep_factor < 1000)
				{
					sleep_factor++;
				}
			}
			
			int alive_count = alive_threads_total;
			int a = get_tasks_running();
			int b = tasks_total;

			// just keep screaming...
			// Without this, in practice it turns out sometimes a thread (or more) remains stuck for a while...
			//
			// See also:
			// - https://en.cppreference.com/w/cpp/thread/condition_variable/notify_all
			// - https://stackoverflow.com/questions/38184549/not-all-threads-notified-of-condition-variable-notify-all
			// where one of the answers says: "Finally, cv.notify_all() only notified currently waiting threads. If a thread shows up later, no dice."
			// which is corroborated by the docs above: "Unblocks all threads currently waiting for *this." and then later:
			// "This makes it impossible for notify_one() to, for example, be delayed and unblock a thread that started waiting just after the call to notify_one() was made."
			// Ditto for notify_all() on that one, of course.
			cv.notify_all();

			tesseract::tprintf("threads pending: {}, tasks running: {}, tasks total: {}\n", alive_count, a, b);

			sleep_or_yield(sleep_factor);
        }
    }

private:
    // ========================
    // Private member functions
    // ========================

    /**
     * @brief Create the threads in the pool and assign a worker to each thread.
     */
    void create_threads()
    {
        running = true;
        for (concurrency_t i = 0; i < thread_count; ++i)
        {
            threads[i] = std::thread(&thread_pool::worker, this);
        }
    }

    /**
     * @brief Destroy the threads in the pool.
     */
    void destroy_threads()
    {
        running = false;
        task_available_cv.notify_all();
        wait_for_tasks();
        for (concurrency_t i = 0; i < thread_count; ++i)
        {
            threads[i].join();
        }
    }

    /**
     * @brief Determine how many threads the pool should have, based on the parameter passed to the constructor or reset().
     *
     * @param thread_count_ The parameter passed to the constructor or reset(). If the parameter is a positive number, then the pool will be created with this number of threads. If the parameter is non-positive, or a parameter was not supplied (in which case it will have the default value of 0), then the pool will be created with the total number of hardware threads available, as obtained from std::thread::hardware_concurrency(). If the latter returns a non-positive number for some reason, then the pool will be created with just one thread.
     * @return The number of threads to use for constructing the pool.
     */
    [[nodiscard]] concurrency_t determine_thread_count(const concurrency_t thread_count_)
    {
        if (thread_count_ > 0)
            return thread_count_;
        else
        {
            if (std::thread::hardware_concurrency() > 0)
                return std::thread::hardware_concurrency();
            else
                return 1;
        }
    }

    /**
     * @brief A worker function to be assigned to each thread in the pool. Waits until it is notified by push_task() that a task is available, and then retrieves the task from the queue and executes it. Once the task finishes, the worker notifies wait_for_tasks() in case it is waiting.
     */
    void worker()
    {
        while (running)
        {
            std::function<void()> task;
            std::unique_lock<std::mutex> tasks_lock(tasks_mutex);
            task_available_cv.wait(tasks_lock, [this] { return !tasks.empty() || !running; });
            if (running && !paused)
            {
                task = std::move(tasks.front());
                tasks.pop();
                tasks_lock.unlock();
                task();
                tasks_lock.lock();
                --tasks_total;
                if (waiting)
                    task_done_cv.notify_one();
            }
        }
    }

    /**
     * @brief Sleep for sleep_duration microseconds. If that variable is set to zero, yield instead.
     *
     */
    void sleep_or_yield(int factor = 1)
    {
		auto t = sleep_duration * factor;
        if (t > 0)
            std::this_thread::sleep_for(std::chrono::microseconds(t));
        else
            std::this_thread::yield();
    }

    /**
     * @brief A worker function to be assigned to each thread in the pool. Continuously pops tasks out of the queue and executes them, as long as the atomic variable running is set to true.
     */
	void __worker()
	{
		try
		{
			while (running)
			{
				std::function<void()> task;
				if (!paused && pop_task(task))
				{
					task();
					tasks_total--;
				} else
				{
					std::unique_lock<std::mutex> lock(cv_m);
					cv.wait(lock);
				}
			}
		}
		catch (...)
		{
			// don't care that much no more. Still, try to log it.
			std::exception_ptr p = std::current_exception();
			try {
				if (p) {
					std::rethrow_exception(p);
				}
			}
			catch (const std::exception& e) {
				tesseract::tprintf("ERROR: thread::worker caught unhandled exception: {}.\nWARNING: The thread will terminate/abort now!\n", e.what());
			}
		}
	}

// MSVC supports hardware SEH:
#if defined(_MSC_VER)

	struct _EXCEPTION_POINTERS __ex_info = {0};

	int ex_filter(unsigned long code, struct _EXCEPTION_POINTERS *info)
	{
		if (info != NULL)
		{
			__ex_info = *info;
		}
		return EXCEPTION_EXECUTE_HANDLER;
	}

#endif

	void worker()
	{
		alive_threads_total++;

// MSVC supports hardware SEH:
#if defined(_MSC_VER)

		__try
		{
			__try
			{
				__worker();
			}
			__finally
			{
				alive_threads_total--;
				tesseract::tprintf("%s: thread::worker unwinding; termination is %s\n", AbnormalTermination() ? "ERROR" : "INFO", AbnormalTermination() ? "ABNORMAL" : "normal");
			}
		}
		__except (ex_filter(GetExceptionCode(), GetExceptionInformation()))
		{
			struct _EXCEPTION_POINTERS ex_info = __ex_info;
			auto code = GetExceptionCode();
#if 0
			std::exception_ptr p = std::current_exception();
#else
			void *p = ex_info.ExceptionRecord->ExceptionAddress;
#endif

#define select_and_report(x)																	\
	case x:																						\
		tesseract::tprintf("ERROR: thread::worker unwinding; termination code is %s, current_exception_ptr = %p\n", #x, p);

			switch (code)
			{
			default:
				tesseract::tprintf("ERROR: thread::worker unwinding; termination code is %s\n", "**UNKNOWN**");
				break;

				select_and_report(STILL_ACTIVE)
					break;
				select_and_report(EXCEPTION_ACCESS_VIOLATION)
					break;
				select_and_report(EXCEPTION_DATATYPE_MISALIGNMENT)
					break;
				select_and_report(EXCEPTION_BREAKPOINT)
					break;
				select_and_report(EXCEPTION_SINGLE_STEP)
					break;
				select_and_report(EXCEPTION_ARRAY_BOUNDS_EXCEEDED)
					break;
				select_and_report(EXCEPTION_FLT_DENORMAL_OPERAND)
					break;
				select_and_report(EXCEPTION_FLT_DIVIDE_BY_ZERO)
					break;
				select_and_report(EXCEPTION_FLT_INEXACT_RESULT)
					break;
				select_and_report(EXCEPTION_FLT_INVALID_OPERATION)
					break;
				select_and_report(EXCEPTION_FLT_OVERFLOW)
					break;
				select_and_report(EXCEPTION_FLT_STACK_CHECK)
					break;
				select_and_report(EXCEPTION_FLT_UNDERFLOW)
					break;
				select_and_report(EXCEPTION_INT_DIVIDE_BY_ZERO)
					break;
				select_and_report(EXCEPTION_INT_OVERFLOW)
					break;
				select_and_report(EXCEPTION_PRIV_INSTRUCTION)
					break;
				select_and_report(EXCEPTION_IN_PAGE_ERROR)
					break;
				select_and_report(EXCEPTION_ILLEGAL_INSTRUCTION)
					break;
				select_and_report(EXCEPTION_NONCONTINUABLE_EXCEPTION)
					break;
				select_and_report(EXCEPTION_STACK_OVERFLOW)
					break;
				select_and_report(EXCEPTION_INVALID_DISPOSITION)
					break;
				select_and_report(EXCEPTION_GUARD_PAGE)
					break;
				select_and_report(EXCEPTION_INVALID_HANDLE)
					break;
				select_and_report(EXCEPTION_POSSIBLE_DEADLOCK)
					break;
				select_and_report(CONTROL_C_EXIT)
					break;
			}

#if 0
			if (p)
			{
				std::rethrow_exception(p);
			}
#endif

			// TODO:
			// - RaiseException https://docs.microsoft.com/en-us/windows/win32/api/errhandlingapi/nf-errhandlingapi-raiseexception
			// - SetThreadErrorMode https://docs.microsoft.com/en-us/windows/win32/api/errhandlingapi/nf-errhandlingapi-setthreaderrormode
		}

#else

		__worker();
		alive_threads_total--;

#endif

	}

    // ============
    // Private data
    // ============

    /**
     * @brief An atomic variable indicating whether the workers should pause. When set to true, the workers temporarily stop retrieving new tasks out of the queue, although any tasks already executed will keep running until they are finished. When set to false again, the workers resume retrieving tasks.
     */
    std::atomic<bool> paused = false;

    /**
     * @brief An atomic variable indicating to the workers to keep running. When set to false, the workers permanently stop working.
     */
    std::atomic<bool> running = false;

    /**
     * @brief The duration, in microseconds, that the worker function should sleep for when it cannot find any tasks in the queue. If set to 0, then instead of sleeping, the worker function will execute std::this_thread::yield() if there are no tasks in the queue. The default value is 1000.
     */
    ui32 sleep_duration = 1000;

	std::condition_variable cv;
    std::mutex cv_m;

    /**
     * @brief A condition variable used to notify worker() that a new task has become available.
     */
    std::condition_variable task_available_cv = {};

    /**
     * @brief A condition variable used to notify wait_for_tasks() that a tasks is done.
     */
    std::condition_variable task_done_cv = {};

    /**
     * @brief A queue of tasks to be executed by the threads.
     */
    std::queue<std::function<void()>> tasks = {};

    /**
     * @brief An atomic variable to keep track of the total number of unfinished tasks - either still in the queue, or running in a thread.
     */
    std::atomic<size_t> tasks_total = 0;

    /**
     * @brief A mutex to synchronize access to the task queue by different threads.
     */
    mutable std::mutex tasks_mutex = {};

    /**
     * @brief The number of threads in the pool.
     */
    concurrency_t thread_count = 0;

    /**
     * @brief A smart pointer to manage the memory allocated for the threads.
     */
    std::unique_ptr<std::thread[]> threads = nullptr;

    /**
     * @brief An atomic variable indicating that wait_for_tasks() is active and expects to be notified whenever a task is done.
     */
    std::atomic<bool> waiting = false;

    /**
     * @brief An atomic variable to keep track of the total number of unfinished tasks - either still in the queue, or running in a thread.
     */
    std::atomic<ui64> tasks_total = 0;

	/**
	 * @brief An atomic variable to keep track of the total number of activated threads - each is either waiting for a task or executing a task. That number is tracked by `tasks_total`.
	 */
	std::atomic<ui32> alive_threads_total = 0;
};

//                                     End class thread_pool                                     //
// ============================================================================================= //

// ============================================================================================= //
//                                   Begin class synced_stream                                   //

/**
 * @brief A helper class to synchronize printing to an output stream by different threads.
 */
class [[nodiscard]] synced_stream
{
public:
    /**
     * @brief Construct a new synced stream.
     *
     * @param out_stream_ The output stream to print to. The default value is std::cout.
     */
    synced_stream(std::ostream& out_stream_ = std::cout) : out_stream(out_stream_) {}

    /**
     * @brief Print any number of items into the output stream. Ensures that no other threads print to this stream simultaneously, as long as they all exclusively use the same synced_stream object to print.
     *
     * @tparam T The types of the items
     * @param items The items to print.
     */
    template <typename... T>
    void print(T&&... items)
    {
        const std::scoped_lock lock(stream_mutex);
        (out_stream << ... << std::forward<T>(items));
    }

    /**
     * @brief Print any number of items into the output stream, followed by a newline character. Ensures that no other threads print to this stream simultaneously, as long as they all exclusively use the same synced_stream object to print.
     *
     * @tparam T The types of the items
     * @param items The items to print.
     */
    template <typename... T>
    void println(T&&... items)
    {
        print(std::forward<T>(items)..., '\n');
    }

    /**
     * @brief A stream manipulator to pass to a synced_stream (an explicit cast of std::endl). Prints a newline character to the stream, and then flushes it. Should only be used if flushing is desired, otherwise '\n' should be used instead.
     */
    inline static std::ostream& (&endl)(std::ostream&) = static_cast<std::ostream& (&)(std::ostream&)>(std::endl);

    /**
     * @brief A stream manipulator to pass to a synced_stream (an explicit cast of std::flush). Used to flush the stream.
     */
    inline static std::ostream& (&flush)(std::ostream&) = static_cast<std::ostream& (&)(std::ostream&)>(std::flush);

private:
    /**
     * @brief The output stream to print to.
     */
    std::ostream& out_stream;

    /**
     * @brief A mutex to synchronize printing.
     */
    mutable std::mutex stream_mutex = {};
};

//                                    End class synced_stream                                    //
// ============================================================================================= //

// ============================================================================================= //
//                                       Begin class timer                                       //

/**
 * @brief A helper class to measure execution time for benchmarking purposes.
 */
class [[nodiscard]] timer
{
public:
    /**
     * @brief Start (or restart) measuring time.
     */
    void start()
    {
        start_time = std::chrono::steady_clock::now();
    }

    /**
     * @brief Stop measuring time and store the elapsed time since start().
     */
    void stop()
    {
        elapsed_time = std::chrono::steady_clock::now() - start_time;
    }

    /**
     * @brief Get the number of milliseconds that have elapsed between start() and stop().
     *
     * @return The number of milliseconds.
     */
    [[nodiscard]] std::chrono::milliseconds::rep ms() const
    {
        return (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time)).count();
    }

private:
    /**
     * @brief The time point when measuring started.
     */
    std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::steady_clock::now();

    /**
     * @brief The duration that has elapsed between start() and stop().
     */
    std::chrono::duration<double> elapsed_time = std::chrono::duration<double>::zero();
};

//                                        End class timer                                        //
// ============================================================================================= //

} // namespace BS
