#pragma once

// template <typename Real>
// class MHDCore;

template <typename Real, typename MHDCoreType, typename GridType>
class BoundaryConditionBase {
    public:
        virtual void apply(MHDCoreType& qq) = 0;
        virtual ~BoundaryConditionBase() = default;
};