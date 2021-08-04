#pragma once

#include <alpaka/alpaka.hpp>
#include <cstdlib>
#include <ostream>
#include <vikunja/transform/transform.hpp>

namespace lazyVec
{
    struct Sum
    {
        int ALPAKA_FN_HOST_ACC operator()(/*TAcc const&, */ int a, int b) const
        {
            return a + b;
        }
    };

    template<typename TType, std::size_t TSize, typename TAlpakaDevice>
    class Vector
    {
    public:
        using value_type = TType;
        std::size_t const size = TSize;
        using AlpakaDeviceType = TAlpakaDevice;
        using MappingType = typename AlpakaDeviceType::MappingType;
        using BufType = alpaka::Buf<MappingType, TType, alpaka::DimInt<1u>, std::size_t>;

    private:
        AlpakaDeviceType const devAcc;
        using Vec = alpaka::Vec<alpaka::DimInt<1u>, std::size_t>;
        BufType buffer;
        TType* mem_ptr;

    public:
        Vector(TAlpakaDevice const acc)
            : devAcc(acc)
            , buffer(alpaka::allocBuf<TType, std::size_t>(devAcc.device, Vec::all(TSize)))
            , mem_ptr(alpaka::getPtrNative(buffer))
        {
        }

        TType* begin()
        {
            return mem_ptr;
        }
        TType* end()
        {
            return mem_ptr + TSize;
        }

        TType& operator[](std::size_t index)
        {
            return mem_ptr[index];
        }

        TType& operator[](std::size_t index) const
        {
            return mem_ptr[index];
        }

        AlpakaDeviceType const& getDevAcc() const
        {
            return devAcc;
        }

        BufType const& get_buffer() const
        {
            return buffer;
        }


        template<typename TOType, std::size_t TOSize, typename TOAlpakaDevice, typename TQueue>
        // copies data from other to own buffer
        void copy(Vector<TOType, TOSize, TOAlpakaDevice> const& other, TQueue& queue)
        {
            alpaka::Vec<alpaka::DimInt<1u>, std::size_t> const extent(TSize);
            alpaka::memcpy(queue, buffer, other.get_buffer(), extent);
        }

        //*** I/O operators ***
        friend std::ostream& operator<<(std::ostream& co, Vector<TType, TSize, TAlpakaDevice> const& v)
        {
            co << "{";
            co << v[0];
            for(std::size_t i = 1; i < TSize; ++i)
                co << ", " << v[i];
            co << "}";
            return co;
        }
    };


    template<typename TType, std::size_t TSize, typename TAlpakaDevice, typename TQueue>
    Vector<TType, TSize, TAlpakaDevice>
    // operator+(
    sum(Vector<TType, TSize, TAlpakaDevice> /* const */& a,
        Vector<TType, TSize, TAlpakaDevice> /* const */& b,
        TQueue& queue)

    {
        Vector<TType, TSize, TAlpakaDevice> c(a.getDevAcc());

        using MappingType = typename TAlpakaDevice::MappingType;

        Sum const sum;
        vikunja::transform::deviceTransform<MappingType>(
            a.getDevAcc(),
            queue,
            TSize,
            a.begin(),
            b.begin(),
            c.begin(),
            sum);
        return c;
    }

} // namespace lazyVec
