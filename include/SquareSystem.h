#ifndef SQUARE_SYSTEM_H
#define SQUARE_SYSTEM_H

#include <vector>
#include <iostream>

template<typename Real>
class SquareSystem
{
public:
    SquareSystem(uint32_t numUnknowns) 
        : unknownAppearance(numUnknowns, std::vector<uint32_t>()), eqIdx(0) {}

    void addTerm(uint32_t unknownId, Real coeff)
    {
        unknownAppearance[unknownId].push_back({unknownId, coeff});
    }

    void addTerms(std::initializer_list<std::tuple<uint32_t, Real>> listcoeff)
    {
        for(auto& term : listcoeff)
        {
            addTerm(std::get<0>(term), std::get<1>(term));
        }
    }

    void addConstantTerm(Real coeff)
    {
        if(constantTerms.size() > eqIdx)
        {
            std::cerr << "equation constant already added" << std::endl;
            return;
        }
        constantTerms.push_back(coeff);
    }

    void endEquation() { eqIdx++; }

    template<typename Solver>
    void solve()
    {
        const uint32_t numEquations = eqIdx;
        Solver solver(numEquations, unknownAppearance.size());
        
    }
private:
    std::vector<std::vector<std::tuple<uint32_t, Real>>> unknownAppearance;
    std::vector<Real> constantTerms;
    uint32_t eqIdx;
};

#endif