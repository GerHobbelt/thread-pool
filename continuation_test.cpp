#include "BS_thread_pool.hpp"

#include <thread>
#include <iostream>

BS::synced_stream sync_cout(std::cout);

int init()
{
    return 1;
}

int multiply(std::shared_future<int> temporary_result, int multiplier)
{
    return temporary_result.get() * multiplier;
}

int main()
{
    BS::thread_pool pool(1);

    auto tmp = pool.submit(init);
    auto result = pool.submit(multiply, tmp.share(), 2);

    std::cout << result.get() << std::endl;
}