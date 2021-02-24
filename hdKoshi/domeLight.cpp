#include "domeLight.h"

#include <iostream>
#include "pxr/base/gf/matrix4f.h"
#include "pxr/imaging/hd/sceneDelegate.h"
#include "pxr/usd/sdf/types.h"
#include "renderParam.h"


PXR_NAMESPACE_OPEN_SCOPE

HdKoshiDomeLight::HdKoshiDomeLight(const SdfPath& id)
: HdLight(id), geometry(nullptr)
{
}

void HdKoshiDomeLight::Finalize(HdRenderParam *renderParam)
{
    static_cast<HdKoshiRenderParam*>(renderParam)->getScene()->removeGeometry(GetId().GetString());
}

HdDirtyBits HdKoshiDomeLight::GetInitialDirtyBitsMask() const
{
    int mask = HdChangeTracker::Clean
        | HdChangeTracker::InitRepr
        | HdChangeTracker::DirtyPoints
        | HdChangeTracker::DirtyTopology
        | HdChangeTracker::DirtyTransform
        | HdChangeTracker::DirtyVisibility
        | HdChangeTracker::DirtyCullStyle
        | HdChangeTracker::DirtyDoubleSided
        | HdChangeTracker::DirtyDisplayStyle
        | HdChangeTracker::DirtySubdivTags
        | HdChangeTracker::DirtyPrimvar
        | HdChangeTracker::DirtyNormals
        | HdChangeTracker::DirtyInstancer;
    return (HdDirtyBits)mask;
}

void HdKoshiDomeLight::Sync(HdSceneDelegate * sceneDelegate, HdRenderParam * renderParam, HdDirtyBits * dirtyBits)
{
    const SdfPath& id = GetId();

    // If we don't have a geometry yet create our geometry.
    if(!geometry)
    {
        geometry = std::make_shared<Koshi::GeometryEnvironment>();

        GfMatrix4f transform = GfMatrix4f(sceneDelegate->GetTransform(id));
        geometry->setTransform(Koshi::Transform::fromColumnFirstData(transform.data()));

        // // intensity
        // VtValue intensity = sceneDelegate->GetLightParamValue(id, HdLightTokens->intensity);
        // if (intensity.IsHolding<float>())
        //     std::cout << "Intensity: " << intensity.Get<float>() << std::endl;

        // // exposure
        // VtValue exposure =
        //     sceneDelegate->GetLightParamValue(id, HdLightTokens->exposure);
        // if (exposure.IsHolding<float>()) {
        //     lightNode.params.SetFloat(us_exposure, exposure.UncheckedGet<float>());
        // }

        // // color -> lightColor
        // VtValue lightColor =
        //     sceneDelegate->GetLightParamValue(id, HdLightTokens->color);
        // if (lightColor.IsHolding<GfVec3f>()) {
        //     GfVec3f v = lightColor.UncheckedGet<GfVec3f>();
        //     lightNode.params.SetColor(us_lightColor, RtColorRGB(v[0], v[1], v[2]));
        // }

        VtValue textureFile = sceneDelegate->GetLightParamValue(id, HdLightTokens->textureFile);        
        if(textureFile.IsHolding<SdfAssetPath>())
        {
            std::string filename = textureFile.UncheckedGet<SdfAssetPath>().GetResolvedPath();
            if (filename.empty())
                filename = textureFile.UncheckedGet<SdfAssetPath>().GetAssetPath();
            geometry->createTexture(filename);
        }

        static_cast<HdKoshiRenderParam*>(renderParam)->getScene()->addGeometry(GetId().GetString(), geometry.get());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
