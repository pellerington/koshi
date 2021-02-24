#pragma once

#include "pxr/pxr.h"
#include "pxr/imaging/hd/light.h"

#include <koshi/geometry/GeometryEnvironment.h>
#include <koshi/Scene.h>

PXR_NAMESPACE_OPEN_SCOPE

// TODO: Unify this as one light.
class HdKoshiDomeLight final : public HdLight 
{
public:
    HdKoshiDomeLight(const SdfPath& id);
    HdDirtyBits GetInitialDirtyBitsMask() const override;

    void Sync(HdSceneDelegate * sceneDelegate, HdRenderParam * renderParam, HdDirtyBits * dirtyBits) override;
    virtual void Finalize(HdRenderParam * renderParam) override;

protected:

    std::shared_ptr<Koshi::GeometryEnvironment> geometry;

    // This class does not support copying.
    HdKoshiDomeLight(const HdKoshiDomeLight&) = delete;
    HdKoshiDomeLight &operator =(const HdKoshiDomeLight&) = delete;
};

PXR_NAMESPACE_CLOSE_SCOPE