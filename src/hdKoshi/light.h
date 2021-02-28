#pragma once

#include "pxr/pxr.h"
#include "pxr/imaging/hd/light.h"

#include <koshi/geometry/GeometryEnvironment.h>
#include <koshi/Scene.h>

PXR_NAMESPACE_OPEN_SCOPE

// TODO: Unify this as one light.
class HdKoshiLight final : public HdLight 
{
public:
    HdKoshiLight(const SdfPath& id, const TfToken& hd_light_type);
    HdDirtyBits GetInitialDirtyBitsMask() const override;

    void Sync(HdSceneDelegate * sceneDelegate, HdRenderParam * renderParam, HdDirtyBits * dirtyBits) override;
    virtual void Finalize(HdRenderParam * renderParam) override;

protected:

    const TfToken hd_light_type;

    std::shared_ptr<Koshi::GeometryEnvironment> geometry;

    // This class does not support copying.
    HdKoshiLight(const HdKoshiLight&) = delete;
    HdKoshiLight &operator =(const HdKoshiLight&) = delete;
};

PXR_NAMESPACE_CLOSE_SCOPE