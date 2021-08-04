#pragma once

#include <alpaka/alpaka.hpp>
#include <vector>

namespace lazyVec
{
    // contains all information, which are required to determine,
    // who is the own of a chunk of memory
    template<typename TAcc>
    struct AlpakaDevice
    {
        // Alpaka mapping strategy
        using MappingType = TAcc;
        // type of the actual device
        using AccType = decltype(alpaka::getDevByIdx<TAcc>(0u));

        const unsigned int id;
        const AccType device;

        AlpakaDevice(unsigned int id) : id(id), device(alpaka::getDevByIdx<TAcc>(id))
        {
        }
    };

    // helper struct to bundle device with queues
    template<typename TAcc, typename TQueueKind>
    struct AlpakaRunner
    {
        using AlpakaDeviceType = AlpakaDevice<TAcc>;
        using QueueKind = TQueueKind;
        using QueueType = alpaka::Queue<TAcc, TQueueKind>;

        AlpakaDeviceType const device;
        std::vector<QueueType> queues;

        AlpakaRunner(AlpakaDeviceType const device) : device(device)
        {
        }
        AlpakaRunner(unsigned int id) : device(AlpakaDeviceType(id))
        {
        }

        void createQueue()
        {
            queues.push_back(QueueType(device.device));
        }
    };
} // namespace lazyVec
