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
    BS::thread_pool pool;

    BS::multi_future<void> result;
    for (int i = 0; i < 10; i++)
    {
        result.push_back(pool.submit(work, i));
    }

    std::future_status status;
    while ((status = result.wait_for(std::chrono::milliseconds(300))) != std::future_status::ready)
    {
        sync_cout.println("Waiting... ");
    }
}