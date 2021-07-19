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

    template<typename TType, std::size_t TSize, typename TAcc, typename TDevAcc, typename TDevQueue>
    class Vector
    {
    public:
        using value_type = TType;
        std::size_t const size = TSize;
        using AccType = TAcc;
        using BufType = alpaka::Buf<TAcc, TType, alpaka::DimInt<1u>, std::size_t>;

    private:
        TDevAcc const& devAcc;
        TDevQueue& devQueue;
        using Vec = alpaka::Vec<alpaka::DimInt<1u>, std::size_t>;
        BufType buffer;
        TType* mem_ptr;

    public:
        Vector(TDevAcc const& acc, TDevQueue& queue)
            : devAcc(acc)
            , devQueue(queue)
            , buffer(alpaka::allocBuf<TType, std::size_t>(devAcc, Vec::all(TSize)))
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

        TDevAcc& getDevAcc() const
        {
            return devAcc;
        }

        TDevQueue& getDevQueue()
        {
            return devQueue;
        }

        BufType const& get_buffer() const
        {
            return buffer;
        }


        template<
            typename TOType,
            std::size_t TOSize,
            typename TOAcc,
            typename TODevAcc,
            typename TODevQueue,
            typename TQueue>
        // copies data from other to own buffer
        void copy(Vector<TOType, TOSize, TOAcc, TODevAcc, TODevQueue> const& other, TQueue& queue)
        {
            alpaka::Vec<alpaka::DimInt<1u>, std::size_t> const extent(TSize);
            alpaka::memcpy(queue, buffer, other.get_buffer(), extent);
        }

        //*** I/O operators ***
        friend std::ostream& operator<<(std::ostream& co, Vector<TType, TSize, TAcc, TDevAcc, TDevQueue> const& v)
        {
            co << "{";
            co << v[0];
            for(std::size_t i = 1; i < TSize; ++i)
                co << ", " << v[i];
            co << "}";
            return co;
        }
    };

    template<typename TType, std::size_t TSize, typename TAcc, typename TDevAcc, typename TDevQueue>
    Vector<TType, TSize, TAcc, TDevAcc, TDevQueue> operator+(
        Vector<TType, TSize, TAcc, TDevAcc, TDevQueue> /* const */& a,
        Vector<TType, TSize, TAcc, TDevAcc, TDevQueue> /* const */& b)
    {
        Vector<TType, TSize, TAcc, TDevAcc, TDevQueue> c(a.getDevAcc(), a.getDevQueue());

        Sum const sum;
        vikunja::transform::deviceTransform<TAcc>(
            a.getDevAcc(),
            a.getDevQueue(),
            TSize,
            a.begin(),
            b.begin(),
            c.begin(),
            sum);
        return c;
    }

} // namespace lazyVec
