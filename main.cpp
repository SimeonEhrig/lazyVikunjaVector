#include "device.hpp"
#include "vector.hpp"

#include <alpaka/alpaka.hpp>
#include <iostream>
//#include <type_traits>

class InitKernel
{
public:
    template<typename TAcc, typename TData>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, TData* const v, std::size_t const& numElements) const
    {
        std::size_t const gridThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        std::size_t const threadElemExtent(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        std::size_t const threadFirstElemIdx(gridThreadIdx * threadElemExtent);

        if(threadFirstElemIdx < numElements)
        {
            std::size_t const threadLastElemIdx(threadFirstElemIdx + threadElemExtent);
            std::size_t const threadLastElemIdxClipped(
                (numElements > threadLastElemIdx) ? threadLastElemIdx : numElements);

            for(std::size_t i(threadFirstElemIdx); i < threadLastElemIdxClipped; ++i)
            {
                v[i] = static_cast<TData>(i);
            }
        }
    }
};

int main()
{
    // ######################
    // # prepare
    // ######################

    constexpr std::size_t vector_size = 5;
    using Dim = alpaka::DimInt<1u>;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    using DevAcc = alpaka::AccGpuCudaRt<Dim, std::size_t>;
#else
    using DevAcc = alpaka::AccCpuSerial<Dim, std::size_t>;
#endif

    lazyVec::AlpakaRunner<DevAcc, alpaka::Blocking> devRunner(0u);
    devRunner.createQueue();

    // only a notice, how to get the device type from the object
    /*
      lazyVec::AlpakaDevice<DevAcc> const devAcc(0u);
      using accType = std::decay_t<decltype(devAcc.device)>;
      accType const & d = devAcc.device;
    */

    using MappingType = DevAcc;

    alpaka::Vec<Dim, std::size_t> const extent(vector_size);
    alpaka::WorkDivMembers<Dim, std::size_t> const workdiv{
        vector_size,
        static_cast<std::size_t>(1),
        static_cast<std::size_t>(1)};

    lazyVec::Vector<int, vector_size, lazyVec::AlpakaDevice<DevAcc>> a(devRunner.device);
    lazyVec::Vector<int, vector_size, lazyVec::AlpakaDevice<DevAcc>> b(devRunner.device);

    // ######################
    // # init data
    // ######################

    InitKernel const initKernel;
    alpaka::exec<MappingType>(devRunner.queues.front(), workdiv, initKernel, a.begin(), a.size);
    alpaka::exec<MappingType>(devRunner.queues.front(), workdiv, initKernel, b.begin(), b.size);

    // ######################
    // # run calculation
    // ######################

    // should work again, if lazy evaluation is implemented
    // auto c = a + b;
    auto c = lazyVec::sum(a, b, devRunner.queues.front());

    // ######################
    // # print result
    // ######################

    using HostAcc = alpaka::AccCpuSerial<Dim, std::size_t>;
    lazyVec::AlpakaRunner<HostAcc, alpaka::Blocking> hostRunner(0u);
    hostRunner.createQueue();

    // use vector to run safe std::cout -> if the Acc of a,b or c is CUDA, std::cout should not work
    lazyVec::Vector<int, vector_size, lazyVec::AlpakaDevice<HostAcc>> output(hostRunner.device);
    output.copy(a, devRunner.queues.front());
    alpaka::wait(devRunner.queues.front());
    std::cout << "a: " << output << std::endl;

    output.copy(b, devRunner.queues.front());
    alpaka::wait(devRunner.queues.front());
    std::cout << "b: " << output << std::endl;

    output.copy(c, devRunner.queues.front());
    alpaka::wait(devRunner.queues.front());
    std::cout << "c: " << output << std::endl;

    return 0;
}
