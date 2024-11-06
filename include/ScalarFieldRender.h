#ifndef SCALAR_FIELD_RENDER_H
#define SCALAR_FIELD_RENDER_H

#include "NodeTree.h"
#include "MyImage.h"
#include "ScalarField.h"

namespace ScalarFieldRender
{
    void renderNodeTreeGrid(const NodeTree<2>& nodeTree, Image& outImage, int thickness, float opacity)
    {
        glm::vec2 minCoord = nodeTree.getMinCoord();
        glm::vec2 size = nodeTree.getMaxCoord() - nodeTree.getMinCoord();
        glm::vec2 invSize = 1.0f / size;
        for(uint32_t i=0; i < outImage.width(); i++)
        {
            for(uint32_t j=0; j < outImage.height(); j++)
            {
                glm::vec2 pos = glm::vec2((static_cast<float>(i)+0.5f) / static_cast<float>(outImage.width()), 
									  (static_cast<float>(j)+0.5f) / static_cast<float>(outImage.height()));
			    pos = minCoord + pos * size;

                std::optional<NodeTree<2>::Node> node;
                nodeTree.getNode(pos, node);
                if(!node) continue;
                glm::vec2 nsize = (node->maxCoord - node->minCoord);
                glm::vec2 np = (pos - node->minCoord) / nsize;
                np = (glm::vec2(0.5f) - glm::abs(np - glm::vec2(0.5f))) * nsize * glm::vec2(outImage.width(), outImage.height()) * invSize;
                
                float gridColorWeight = glm::clamp(1.0 - glm::pow(glm::min(np.x, np.y) / (0.5f * thickness), 12), 0.0, 1.0);

                outImage(i, j) = glm::mix(outImage(i, j), glm::vec3(0.0f), opacity * gridColorWeight);
            }
        }
    }

    void renderScalarField(ScalarField<2>& scalarField, Image& outImage, std::function<float(float)> colorValueTransform, const std::vector<glm::vec3>& colorPalette, 
                           float mainIsolineThickness, float mainIsolineValue, float mainIsolineOpacity,
                           float isolinesThickness, float isolinesSpacing, float isolinesOpacity)
    {
        glm::vec2 minCoord = scalarField.getMinCoord();
        glm::vec2 size = scalarField.getMaxCoord() - scalarField.getMinCoord();
        glm::vec2 invSize = 1.0f / size;
        float invSizeMag = glm::length(glm::vec2(outImage.width(), outImage.height()) * invSize);
        float winSizeMag = glm::length(glm::vec2(outImage.width(), outImage.height()));
        for(uint32_t i=0; i < outImage.width(); i++)
        {
            for(uint32_t j=0; j < outImage.height(); j++)
            {
                glm::vec2 pos = glm::vec2((static_cast<float>(i)+0.5f) / static_cast<float>(outImage.width()), 
									  (static_cast<float>(j)+0.5f) / static_cast<float>(outImage.height()));
			    pos = minCoord + pos * size;

                // Field color
                const float fval = scalarField.eval(pos);
                float val = colorValueTransform(fval);
                float nVal = static_cast<float>(colorPalette.size() - 1) * glm::clamp(val, 0.0f, 0.999999f); 
                uint32_t cid = glm::floor(nVal);
                glm::vec3 bgColor = glm::mix(colorPalette[cid], colorPalette[cid+1], glm::fract(nVal));

                // Main isoline
                float mainIsolineWeight = 0.0f;
                if(mainIsolineThickness > 0.0f)
                {
                    const float isoDist = glm::abs(fval - mainIsolineValue) * invSizeMag;
                    const float nDist = isoDist / (0.5f * glm::sqrt(2.0f) * mainIsolineThickness);
                    mainIsolineWeight = glm::clamp(1.0 - glm::pow(nDist, 12), 0.0, 1.0);
                }

                // Isolines
                float isolinesWeight = 0.0f;
                if(isolinesThickness > 0.0f)
                {
                    const float isoDist = (0.5f - glm::abs(glm::fract(fval * glm::min(invSize.x, invSize.y) / isolinesSpacing) - 0.5)) * isolinesSpacing * winSizeMag;
                    const float nDist = isoDist / (0.5f * glm::sqrt(2.0f) * isolinesThickness);
                    isolinesWeight = glm::clamp(1.0 - glm::pow(nDist, 12), 0.0, 1.0);
                }

                float isoWeight = glm::max(mainIsolineWeight * mainIsolineOpacity, isolinesWeight * isolinesOpacity);
                outImage(i, j) = glm::mix(bgColor, glm::vec3(0.0f), isoWeight);
            }
        }
    }

    void renderColorField(ScalarField<2>& scalarField, std::function<float(glm::vec2)> colorField, Image& outImage, const std::vector<glm::vec3>& colorPalette)
    {
        glm::vec2 minCoord = scalarField.getMinCoord();
        glm::vec2 size = scalarField.getMaxCoord() - scalarField.getMinCoord();
        glm::vec2 invSize = 1.0f / size;
        float invSizeMag = glm::length(glm::vec2(outImage.width(), outImage.height()) * invSize);
        float winSizeMag = glm::length(glm::vec2(outImage.width(), outImage.height()));
        for(uint32_t i=0; i < outImage.width(); i++)
        {
            for(uint32_t j=0; j < outImage.height(); j++)
            {
                glm::vec2 pos = glm::vec2((static_cast<float>(i)+0.5f) / static_cast<float>(outImage.width()), 
									  (static_cast<float>(j)+0.5f) / static_cast<float>(outImage.height()));
			    pos = minCoord + pos * size;

                // Field color
                float val = colorField(pos);
                float nVal = static_cast<float>(colorPalette.size() - 1) * glm::clamp(val, 0.0f, 0.999999f); 
                uint32_t cid = glm::floor(nVal);
                glm::vec3 bgColor = glm::mix(colorPalette[cid], colorPalette[cid+1], glm::fract(nVal));
                outImage(i, j) = bgColor;
            }
        }
    }
}

#endif