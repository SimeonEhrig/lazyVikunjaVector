#include "vector.hpp"

#include <alpaka/alpaka.hpp>
#include <iostream>

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
    using DevQueue = alpaka::Queue<DevAcc, alpaka::Blocking>;

    auto const devAcc = alpaka::getDevByIdx<DevAcc>(0u);
    DevQueue devQueue(devAcc);

    alpaka::Vec<Dim, std::size_t> const extent(vector_size);
    alpaka::WorkDivMembers<Dim, std::size_t> const workdiv{
        vector_size,
        static_cast<std::size_t>(1),
        static_cast<std::size_t>(1)};

    lazyVec::Vector<int, vector_size, DevAcc, decltype(devAcc), DevQueue> a(devAcc, devQueue);
    lazyVec::Vector<int, vector_size, DevAcc, decltype(devAcc), DevQueue> b(devAcc, devQueue);

    // ######################
    // # init data
    // ######################

    InitKernel const initKernel;
    alpaka::exec<DevAcc>(devQueue, workdiv, initKernel, a.begin(), a.size);
    alpaka::exec<DevAcc>(devQueue, workdiv, initKernel, b.begin(), b.size);

    // ######################
    // # run calculation
    // ######################

    auto c = a + b;

    // ######################
    // # print result
    // ######################

    using HostAcc = alpaka::AccCpuSerial<Dim, std::size_t>;
    using HostQueue = alpaka::Queue<HostAcc, alpaka::Blocking>;
    auto const hostAcc = alpaka::getDevByIdx<HostAcc>(0u);
    HostQueue hostQueue(hostAcc);

    // use vector to run safe std::cout -> if the Acc of a,b or c is CUDA, std::cout should not work
    lazyVec::Vector<int, vector_size, HostAcc, decltype(hostAcc), HostQueue> output(hostAcc, hostQueue);
    output.copy(a, devQueue);
    alpaka::wait(devQueue);
    std::cout << "a: " << output << std::endl;

    output.copy(b, devQueue);
    alpaka::wait(devQueue);
    std::cout << "b: " << output << std::endl;

    output.copy(c, devQueue);
    alpaka::wait(devQueue);
    std::cout << "c: " << output << std::endl;

    return 0;
}
