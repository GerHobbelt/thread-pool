#include "BS_thread_pool.hpp"

#include <thread>

BS::synced_stream sync_cout(std::cout);

void work(int i)
{
    std::this_thread::sleep_for(std::chrono::seconds(i));
    sync_cout.println("Finished work item ", i);
}

int main()
{
    BS::thread_pool pool(2);

    BS::multi_future<void> result;
    for (int i = 0; i < 10; i++)
    {
        result.push_back(pool.submit(work, i));
    }

    int i = 0;

    std::future_status status;
    while ((status = result.wait_for(std::chrono::milliseconds(300))) != std::future_status::ready)
    {
        sync_cout.println("Waiting... ");

        if (++i == 5) {
            pool.pause();
            pool.wait_for_tasks();

            sync_cout.println("Queued ", pool.get_tasks_queued());
            sync_cout.println("Running ", pool.get_tasks_running());
            sync_cout.println("Total ", pool.get_tasks_total());
            pool.cancel_tasks();

            sync_cout.println("Cancelled ");
            return 0;
        }
    }
}