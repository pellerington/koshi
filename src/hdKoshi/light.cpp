#include "light.h"

#include <iostream>
#include "pxr/base/gf/matrix4f.h"
#include "pxr/imaging/hd/sceneDelegate.h"
#include "pxr/usd/sdf/types.h"
#include "pxr/usd/usdLux/blackbody.h"
#include "renderParam.h"

#include <koshi/geometry/GeometryQuad.h>
#include <koshi/geometry/GeometryEnvironment.h>

PXR_NAMESPACE_OPEN_SCOPE

HdKoshiLight::HdKoshiLight(const SdfPath& id, const TfToken& hd_light_type)
: HdLight(id), hd_light_type(hd_light_type), geometry(nullptr)
{
}

void HdKoshiLight::Finalize(HdRenderParam *renderParam)
{
    static_cast<HdKoshiRenderParam*>(renderParam)->getScene()->removeGeometry(GetId().GetString());
}

HdDirtyBits HdKoshiLight::GetInitialDirtyBitsMask() const
{
    return HdChangeTracker::AllDirty;
}

void HdKoshiLight::Sync(HdSceneDelegate * sceneDelegate, HdRenderParam * renderParam, HdDirtyBits * dirtyBits)
{
    const SdfPath& id = GetId();

    Koshi::Vec3f light(1.f);

    VtValue color = sceneDelegate->GetLightParamValue(id, HdLightTokens->color);
    if(color.IsHolding<GfVec3f>()) {
        GfVec3f v = color.Get<GfVec3f>();
        light = Koshi::Vec3f(v[0], v[1], v[2]);
    } 

    // TODO: Should also support enable color temp attribute.
    VtValue temp = sceneDelegate->GetLightParamValue(id, HdLightTokens->colorTemperature);
    if(temp.IsHolding<float>()) {
        GfVec3f v = UsdLuxBlackbodyTemperatureAsRgb(temp.Get<float>());
        light *= Koshi::Vec3f(v[0], v[1], v[2]);
    }

    VtValue intensity = sceneDelegate->GetLightParamValue(id, HdLightTokens->intensity);
    if(intensity.IsHolding<float>())
        light *= intensity.Get<float>();

    VtValue exposure = sceneDelegate->GetLightParamValue(id, HdLightTokens->exposure);
    if(exposure.IsHolding<float>())
        light *= std::pow(2.f, exposure.Get<float>());

    // If we don't have a geometry yet create our geometry.
    if(!geometry)
    {

        if(hd_light_type == HdPrimTypeTokens->rectLight)
        {
            geometry = std::make_shared<Koshi::GeometryQuad>();
            Koshi::GeometryQuad * quad = (Koshi::GeometryQuad *)geometry.get();

            GfMatrix4f transform = GfMatrix4f(sceneDelegate->GetTransform(id));
            geometry->setTransform(Koshi::Transform::fromColumnFirstData(transform.data()));

            quad->temp_light = light;

            static_cast<HdKoshiRenderParam*>(renderParam)->getScene()->addGeometry(GetId().GetString(), geometry.get());
        }

        else if(hd_light_type == HdPrimTypeTokens->domeLight)
        {
            geometry = std::make_shared<Koshi::GeometryEnvironment>();
            Koshi::GeometryEnvironment * environment = (Koshi::GeometryEnvironment *)geometry.get();

            GfMatrix4f transform = GfMatrix4f(sceneDelegate->GetTransform(id));
            geometry->setTransform(Koshi::Transform::fromColumnFirstData(transform.data()));

            environment->temp_light = light;

            VtValue textureFile = sceneDelegate->GetLightParamValue(id, HdLightTokens->textureFile);        
            if(textureFile.IsHolding<SdfAssetPath>())
            {
                std::string filename = textureFile.UncheckedGet<SdfAssetPath>().GetResolvedPath();
                if (filename.empty())
                    filename = textureFile.UncheckedGet<SdfAssetPath>().GetAssetPath();
                environment->createTexture(filename);
            }

            static_cast<HdKoshiRenderParam*>(renderParam)->getScene()->addGeometry(GetId().GetString(), geometry.get());
        }

    }
}

PXR_NAMESPACE_CLOSE_SCOPE
