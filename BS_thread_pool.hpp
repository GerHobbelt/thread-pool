#pragma once

/**
 * @file thread_pool.hpp
 * @author Barak Shoshany (baraksh@gmail.com) (http://baraksh.com)
 * @version 2.0.0
 * @date 2021-08-14
 * @copyright Copyright (c) 2021 Barak Shoshany. Licensed under the MIT license. If you use this library in published research, please cite it as follows:
 *  - Barak Shoshany, "A C++17 Thread Pool for High-Performance Scientific Computing", doi:10.5281/zenodo.4742687, arXiv:2105.00613 (May 2021)
 *
 * @brief A C++17 thread pool for high-performance scientific computing.
 * @details A modern C++17-compatible thread pool implementation, built from scratch with high-performance scientific computing in mind. The thread pool is implemented as a single lightweight and self-contained class, and does not have any dependencies other than the C++17 standard library, thus allowing a great degree of portability. In particular, this implementation does not utilize OpenMP or any other high-level multithreading APIs, and thus gives the programmer precise low-level control over the details of the parallelization, which permits more robust optimizations. The thread pool was extensively tested on both AMD and Intel CPUs with up to 40 cores and 80 threads. Other features include automatic generation of futures and easy parallelization of loops. Two helper classes enable synchronizing printing to an output stream by different threads and measuring execution time for benchmarking purposes. Please visit the GitHub repository at https://github.com/bshoshany/thread-pool for documentation and updates, or to submit feature requests and bug reports.
 */

#define THREAD_POOL_VERSION "v2.0.0 (2021-08-14)"

#include <atomic>      // std::atomic
#include <chrono>      // std::chrono
#include <condition_variable> // std::condition_variable
#include <cstdint>     // std::int_fast64_t, std::uint_fast32_t
#include <functional>  // std::function
#include <future>      // std::future, std::promise
#include <iostream>    // std::cout, std::ostream
#include <memory>      // std::shared_ptr, std::unique_ptr
#include <mutex>       // std::mutex, std::scoped_lock
#include <queue>       // std::queue
#include <thread>      // std::this_thread, std::thread
#include <type_traits> // std::common_type_t, std::decay_t, std::enable_if_t, std::is_void_v, std::invoke_result_t
#include <utility>     // std::move
#if defined(_MSC_VER)
// see also https://docs.microsoft.com/en-us/cpp/cpp/try-except-statement?view=msvc-170 et al.
#include <winnt.h>
#include <excpt.h>
#include <exception>
#include <stdexcept>
#pragma warning(push)
#pragma warning(disable: 4005) // warning C4005: macro redefinition
#include <ntstatus.h> // STATUS_POSSIBLE_DEADLOCK
#pragma warning(pop)
#endif

// ============================================================================================= //
//                                    Begin class thread_pool                                    //

/**
 * @brief A C++17 thread pool class. The user submits tasks to be executed into a queue. Whenever a thread becomes available, it pops a task from the queue and executes it. Each task is automatically assigned a future, which can be used to wait for the task to finish executing and/or obtain its eventual return value.
 */
class thread_pool
{
    typedef std::uint_fast32_t ui32;
    typedef std::uint_fast64_t ui64;

public:
    // ============================
    // Constructors and destructors
    // ============================

    /**
     * @brief Construct a new thread pool.
     *
     * @param _thread_count The number of threads to use. The default value is the total number of hardware threads available, as reported by the implementation. With a hyperthreaded CPU, this will be twice the number of CPU cores. If the argument is zero, the default value will be used instead.
     */
    thread_pool(const ui32 &_thread_count = std::thread::hardware_concurrency())
        : thread_count(_thread_count ? _thread_count : std::thread::hardware_concurrency()), threads(new std::thread[_thread_count ? _thread_count : std::thread::hardware_concurrency()])
    {
        create_threads();
    }

    /**
     * @brief Destruct the thread pool. Waits for all tasks to complete, then destroys all threads. Note that if the variable paused is set to true, then any tasks still in the queue will never be executed.
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
    ui64 get_tasks_queued() const
    {
        const std::scoped_lock lock(queue_mutex);
        return tasks.size();
    }

    /**
     * @brief Get the number of tasks currently being executed by the threads.
     *
     * @return The number of running tasks.
     */
	ui64 get_tasks_running() const
    {
        return (tasks_total - get_tasks_queued());
    }

    /**
     * @brief Get the total number of unfinished tasks - either still in the queue, or running in a thread.
     *
     * @return The total number of tasks.
     */
	ui64 get_tasks_total() const
    {
        return tasks_total;
    }

    /**
     * @brief Get the number of threads in the pool.
     *
     * @return The number of threads.
     */
    ui32 get_thread_count() const
    {
        return thread_count;
    }

    /**
     * @brief Parallelize a loop by splitting it into blocks, submitting each block separately to the thread pool, and waiting for all blocks to finish executing. The user supplies a loop function, which will be called once per block and should iterate over the block's range.
     *
     * @tparam T1 The type of the first index in the loop. Should be a signed or unsigned integer.
     * @tparam T2 The type of the index after the last index in the loop. Should be a signed or unsigned integer. If T1 is not the same as T2, a common type will be automatically inferred.
     * @tparam F The type of the function to loop through.
     * @param first_index The first index in the loop.
     * @param index_after_last The index after the last index in the loop. The loop will iterate from first_index to (index_after_last - 1) inclusive. In other words, it will be equivalent to "for (T i = first_index; i < index_after_last; i++)". Note that if first_index == index_after_last, the function will terminate without doing anything.
     * @param loop The function to loop through. Will be called once per block. Should take exactly two arguments: the first index in the block and the index after the last index in the block. loop(start, end) should typically involve a loop of the form "for (T i = start; i < end; i++)".
     * @param num_blocks The maximum number of blocks to split the loop into. The default is to use the number of threads in the pool.
     */
    template <typename T1, typename T2, typename F>
    void parallelize_loop(const T1 &first_index, const T2 &index_after_last, const F &loop, ui32 num_blocks = 0)
    {
        typedef std::common_type_t<T1, T2> T;
        T the_first_index = (T)first_index;
        T last_index = (T)index_after_last;
        if (the_first_index == last_index)
            return;
        if (last_index < the_first_index)
        {
            T temp = last_index;
            last_index = the_first_index;
            the_first_index = temp;
        }
        last_index--;
        if (num_blocks == 0)
            num_blocks = thread_count;
        ui64 total_size = (ui64)(last_index - the_first_index + 1);
        ui64 block_size = (ui64)(total_size / num_blocks);
        if (block_size == 0)
        {
            block_size = 1;
            num_blocks = (ui32)total_size > 1 ? (ui32)total_size : 1;
        }
        std::atomic<ui32> blocks_running = 0;
        for (ui32 t = 0; t < num_blocks; t++)
        {
            T start = ((T)(t * block_size) + the_first_index);
            T end = (t == num_blocks - 1) ? last_index + 1 : ((T)((t + 1) * block_size) + the_first_index);
            blocks_running++;
            push_task([start, end, &loop, &blocks_running]
                      {
                          loop(start, end);
                          blocks_running--;
                      });
        }
        while (blocks_running != 0)
        {
            std::this_thread::yield();
        }
    }

    /**
     * @brief Push a function with no arguments or return value into the task queue.
     *
     * @tparam F The type of the function.
     * @param task The function to push.
     */
    template <typename F>
    void push_task(const F &task)
    {
        tasks_total++;
        {
            const std::scoped_lock lock(queue_mutex);
            tasks.push(std::function<void()>(task));
            cv.notify_one();
        }
    }

    /**
     * @brief Push a function with arguments, but no return value, into the task queue.
     * @details The function is wrapped inside a lambda in order to hide the arguments, as the tasks in the queue must be of type std::function<void()>, so they cannot have any arguments or return value. If no arguments are provided, the other overload will be used, in order to avoid the (slight) overhead of using a lambda.
     *
     * @tparam F The type of the function.
     * @tparam A The types of the arguments.
     * @param task The function to push.
     * @param args The arguments to pass to the function.
     */
    template <typename F, typename... A>
    void push_task(const F &task, const A &...args)
    {
        push_task([task, args...]
                  { task(args...); });
    }

    /**
     * @brief Reset the number of threads in the pool. Waits for all currently running tasks to be completed, then destroys all threads in the pool and creates a new thread pool with the new number of threads. Any tasks that were waiting in the queue before the pool was reset will then be executed by the new threads. If the pool was paused before resetting it, the new pool will be paused as well.
     *
     * @param _thread_count The number of threads to use. The default value is the total number of hardware threads available, as reported by the implementation. With a hyperthreaded CPU, this will be twice the number of CPU cores. If the argument is zero, the default value will be used instead.
     */
    void reset(const ui32 &_thread_count = std::thread::hardware_concurrency())
    {
        bool was_paused = paused;
        paused = true;
        wait_for_tasks();
        destroy_threads();
        thread_count = _thread_count ? _thread_count : std::thread::hardware_concurrency();
        threads.reset(new std::thread[thread_count]);
        paused = was_paused;
        running = true;
        create_threads();
    }

    /**
     * @brief Submit a function with zero or more arguments and no return value into the task queue, and get an std::future<bool> that will be set to true upon completion of the task.
     *
     * @tparam F The type of the function.
     * @tparam A The types of the zero or more arguments to pass to the function.
     * @param task The function to submit.
     * @param args The zero or more arguments to pass to the function.
     * @return A future to be used later to check if the function has finished its execution.
     */
    template <typename F, typename... A, typename = std::enable_if_t<std::is_void_v<std::invoke_result_t<std::decay_t<F>, std::decay_t<A>...>>>>
    std::future<bool> submit(const F &task, const A &...args)
    {
        std::shared_ptr<std::promise<bool>> task_promise(new std::promise<bool>);
        std::future<bool> future = task_promise->get_future();
        push_task([task, args..., task_promise]
                  {
                      try
                      {
                          task(args...);
                          task_promise->set_value(true);
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
        return future;
    }

    /**
     * @brief Submit a function with zero or more arguments and a return value into the task queue, and get a future for its eventual returned value.
     *
     * @tparam F The type of the function.
     * @tparam A The types of the zero or more arguments to pass to the function.
     * @tparam R The return type of the function.
     * @param task The function to submit.
     * @param args The zero or more arguments to pass to the function.
     * @return A future to be used later to obtain the function's returned value, waiting for it to finish its execution if needed.
     */
    template <typename F, typename... A, typename R = std::invoke_result_t<std::decay_t<F>, std::decay_t<A>...>, typename = std::enable_if_t<!std::is_void_v<R>>>
    std::future<R> submit(const F &task, const A &...args)
    {
        std::shared_ptr<std::promise<R>> task_promise(new std::promise<R>);
        std::future<R> future = task_promise->get_future();
        push_task([task, args..., task_promise]
                  {
                      try
                      {
                          task_promise->set_value(task(args...));
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
        return future;
    }

    /**
     * @brief Wait for tasks to be completed. Normally, this function waits for all tasks, both those that are currently running in the threads and those that are still waiting in the queue. However, if the variable paused is set to true, this function only waits for the currently running tasks (otherwise it would wait forever). To wait for a specific task, use submit() instead, and call the wait() member function of the generated future.
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

    // ===========
    // Public data
    // ===========

    /**
     * @brief An atomic variable indicating to the workers to pause. When set to true, the workers temporarily stop popping new tasks out of the queue, although any tasks already executed will keep running until they are done. Set to false again to resume popping tasks.
     */
    std::atomic<bool> paused = false;

    /**
     * @brief The duration, in microseconds, that the worker function should sleep for when it cannot find any tasks in the queue. If set to 0, then instead of sleeping, the worker function will execute std::this_thread::yield() if there are no tasks in the queue. The default value is 1000.
     */
    ui32 sleep_duration = 1000;

	std::condition_variable cv;
    std::mutex cv_m;

private:
    // ========================
    // Private member functions
    // ========================

    /**
     * @brief Create the threads in the pool and assign a worker to each thread.
     */
    void create_threads()
    {
        for (ui32 i = 0; i < thread_count; i++)
        {
            threads[i] = std::thread(&thread_pool::worker, this);
        }
    }

    /**
     * @brief Destroy the threads in the pool by joining them.
     */
    void destroy_threads()
    {
        running = false;
        wait_for_tasks();
        for (ui32 i = 0; i < thread_count; i++)
        {
            threads[i].join();
        }
    }

    /**
     * @brief Try to pop a new task out of the queue.
     *
     * @param task A reference to the task. Will be populated with a function if the queue is not empty.
     * @return true if a task was found, false if the queue is empty.
     */
    bool pop_task(std::function<void()> &task)
    {
        const std::scoped_lock lock(queue_mutex);
        if (tasks.empty())
            return false;
        else
        {
            task = std::move(tasks.front());
            tasks.pop();
            return true;
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
     * @brief A mutex to synchronize access to the task queue by different threads.
     */
    mutable std::mutex queue_mutex = {};

    /**
     * @brief An atomic variable indicating to the workers to keep running. When set to false, the workers permanently stop working.
     */
    std::atomic<bool> running = true;

    /**
     * @brief A queue of tasks to be executed by the threads.
     */
    std::queue<std::function<void()>> tasks = {};

    /**
     * @brief The number of threads in the pool.
     */
    ui32 thread_count;

    /**
     * @brief A smart pointer to manage the memory allocated for the threads.
     */
    std::unique_ptr<std::thread[]> threads;

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
class synced_stream
{
public:
    /**
     * @brief Construct a new synced stream.
     *
     * @param _out_stream The output stream to print to. The default value is std::cout.
     */
    synced_stream(std::ostream &_out_stream = std::cout)
        : out_stream(_out_stream){};

    /**
     * @brief Print any number of items into the output stream. Ensures that no other threads print to this stream simultaneously, as long as they all exclusively use this synced_stream object to print.
     *
     * @tparam T The types of the items
     * @param items The items to print.
     */
    template <typename... T>
    void print(const T &...items)
    {
        const std::scoped_lock lock(stream_mutex);
        (out_stream << ... << items);
    }

    /**
     * @brief Print any number of items into the output stream, followed by a newline character. Ensures that no other threads print to this stream simultaneously, as long as they all exclusively use this synced_stream object to print.
     *
     * @tparam T The types of the items
     * @param items The items to print.
     */
    template <typename... T>
    void println(const T &...items)
    {
        print(items..., '\n');
    }

private:
    /**
     * @brief A mutex to synchronize printing.
     */
    mutable std::mutex stream_mutex = {};

    /**
     * @brief The output stream to print to.
     */
    std::ostream &out_stream;
};

//                                    End class synced_stream                                    //
// ============================================================================================= //

// ============================================================================================= //
//                                       Begin class timer                                       //

/**
 * @brief A helper class to measure execution time for benchmarking purposes.
 */
class timer
{
    typedef std::int_fast64_t i64;

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
    i64 ms() const
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
