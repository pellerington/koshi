//
// Copyright 2020 Pixar
//
// Licensed under the Apache License, Version 2.0 (the "Apache License")
// with the following modification; you may not use this file except in
// compliance with the Apache License and the following modification to it:
// Section 6. Trademarks. is deleted and replaced with:
//
// 6. Trademarks. This License does not grant permission to use the trade
//    names, trademarks, service marks, or product names of the Licensor
//    and its affiliates, except as required to comply with Section 4(c) of
//    the License and to reproduce the content of the NOTICE file.
//
// You may obtain a copy of the Apache License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the Apache License with the above modification is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the Apache License for the specific
// language governing permissions and limitations under the Apache License.
//
#include "renderDelegate.h"
#include "mesh.h"
#include "renderBuffer.h"
#include "renderPass.h"
#include "light.h"
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/extComputation.h>

#include <iostream>

PXR_NAMESPACE_OPEN_SCOPE

const TfTokenVector HdKoshiRenderDelegate::SUPPORTED_RPRIM_TYPES =
{
    HdPrimTypeTokens->mesh,
};

const TfTokenVector HdKoshiRenderDelegate::SUPPORTED_SPRIM_TYPES =
{
    HdPrimTypeTokens->camera,
    // HdPrimTypeTokens->material,
    HdPrimTypeTokens->domeLight,
};

const TfTokenVector HdKoshiRenderDelegate::SUPPORTED_BPRIM_TYPES =
{
    HdPrimTypeTokens->renderBuffer,
};

HdKoshiRenderDelegate::HdKoshiRenderDelegate()
: HdRenderDelegate()
{
    _Initialize();
}

HdKoshiRenderDelegate::HdKoshiRenderDelegate(const HdRenderSettingsMap& settingsMap)
: HdRenderDelegate(settingsMap)
{
    _Initialize();
}

void HdKoshiRenderDelegate::_Initialize()
{
    std::cout << "Creating Tiny RenderDelegate" << std::endl;
    _resourceRegistry = std::make_shared<HdResourceRegistry>();
    param = std::make_shared<HdKoshiRenderParam>(&scene, &render_thread);
}

HdKoshiRenderDelegate::~HdKoshiRenderDelegate()
{
    _resourceRegistry.reset();
    std::cout << "Destroying Tiny RenderDelegate" << std::endl;
}

TfTokenVector const& HdKoshiRenderDelegate::GetSupportedRprimTypes() const
{
    return SUPPORTED_RPRIM_TYPES;
}

TfTokenVector const& HdKoshiRenderDelegate::GetSupportedSprimTypes() const
{
    return SUPPORTED_SPRIM_TYPES;
}

TfTokenVector const& HdKoshiRenderDelegate::GetSupportedBprimTypes() const
{
    return SUPPORTED_BPRIM_TYPES;
}

HdResourceRegistrySharedPtr HdKoshiRenderDelegate::GetResourceRegistry() const
{
    return _resourceRegistry;
}

void  HdKoshiRenderDelegate::CommitResources(HdChangeTracker * tracker)
{
    // std::cout << "=> CommitResources RenderDelegate" << std::endl;
}

HdRenderPassSharedPtr HdKoshiRenderDelegate::CreateRenderPass(HdRenderIndex * index, const HdRprimCollection& collection)
{
    std::cout << "Create RenderPass with Collection=" << collection.GetName() << std::endl; 
    return HdRenderPassSharedPtr(new HdKoshiRenderPass(&scene, &render_thread, index, collection));  
}

HdRprim * HdKoshiRenderDelegate::CreateRprim(const TfToken& typeId, const SdfPath& rprimId)
{
    std::cout << "Create Rprim type=" << typeId.GetText() << " id=" << rprimId << std::endl;

    if (typeId == HdPrimTypeTokens->mesh) {
        return new HdKoshiMesh(rprimId);
    } else {
        TF_CODING_ERROR("Unknown Rprim type=%s id=%s", typeId.GetText(), rprimId.GetText());
    }
    return nullptr;
}

void HdKoshiRenderDelegate::DestroyRprim(HdRprim * rPrim)
{
    std::cout << "Destroy Tiny Rprim id=" << rPrim->GetId() << std::endl;
    delete rPrim;
}

HdSprim * HdKoshiRenderDelegate::CreateSprim(const TfToken& typeId, const SdfPath& sprimId)
{
    std::cout << "Create Sprim type=" << typeId.GetText() << " id=" << sprimId << std::endl;

    // TODO: these should be switch statements.
    if (typeId == HdPrimTypeTokens->camera) {
        return new HdCamera(sprimId);
    } else if(typeId == HdPrimTypeTokens->domeLight) {
        return new HdKoshiLight(sprimId, HdPrimTypeTokens->domeLight);
    }
    return nullptr;
}

HdSprim * HdKoshiRenderDelegate::CreateFallbackSprim(const TfToken& typeId)
{
    std::cout << "Create fallback Sprim type=" << typeId.GetText() << std::endl;

    if (typeId == HdPrimTypeTokens->camera) {
        return new HdCamera(SdfPath::EmptyPath());
    } else if(typeId == HdPrimTypeTokens->domeLight) {
        return new HdKoshiLight(SdfPath::EmptyPath(), HdPrimTypeTokens->domeLight);
    }
    return nullptr;
}

void HdKoshiRenderDelegate::DestroySprim(HdSprim *sPrim)
{
    std::cout << "Destroy Tiny Sprim id=" << sPrim->GetId() << std::endl;
    delete sPrim;
}

HdBprim * HdKoshiRenderDelegate::CreateBprim(const TfToken& typeId, const SdfPath& bprimId)
{
    std::cout << "Create Bprim type=" << typeId.GetText() << " id=" << bprimId << std::endl;

    if (typeId == HdPrimTypeTokens->renderBuffer) {
        return new HdKoshiRenderBuffer(bprimId);
    }
    return nullptr;
}

HdBprim * HdKoshiRenderDelegate::CreateFallbackBprim(const TfToken& typeId)
{
    std::cout << "Create fallback Bprim type=" << typeId.GetText() << std::endl;

    if (typeId == HdPrimTypeTokens->renderBuffer) {
        return new HdKoshiRenderBuffer(SdfPath::EmptyPath());
    }
    return nullptr;
}

void HdKoshiRenderDelegate::DestroyBprim(HdBprim * bPrim)
{
    std::cout << "Destroy Tiny Bprim id=" << bPrim->GetId() << std::endl;
    delete bPrim;
}

HdInstancer * HdKoshiRenderDelegate::CreateInstancer(HdSceneDelegate * delegate, const SdfPath& id)
{
    TF_CODING_ERROR("Creating Instancer not supported id=%s", id.GetText());
    return nullptr;
}

void HdKoshiRenderDelegate::DestroyInstancer(HdInstancer * instancer)
{
    TF_CODING_ERROR("Destroy instancer not supported");
}

HdAovDescriptor HdKoshiRenderDelegate::GetDefaultAovDescriptor(const TfToken& name) const
{
    // TODO: We should make these not all be vec4s...
    if (name == HdAovTokens->color) {
        return HdAovDescriptor(HdFormatFloat32Vec4, true, VtValue(GfVec4f(0.0f)));
    } else if (name == HdAovTokens->normal/* || name == HdAovTokens->Neye*/) {
        return HdAovDescriptor(HdFormatFloat32Vec4, true, VtValue(GfVec4f(0.0f)));
    } else if (name == HdAovTokens->depth) {
        return HdAovDescriptor(HdFormatFloat32Vec4, true, VtValue(GfVec4f(0.0f)));
    }
    // } else if (name == HdAovTokens->cameraDepth) {
    //     return HdAovDescriptor(HdFormatFloat32, false, VtValue(0.0f));
    // } else if (name == HdAovTokens->primId || name == HdAovTokens->instanceId || name == HdAovTokens->elementId) {
    //     return HdAovDescriptor(HdFormatInt32, false, VtValue(-1));
    // } else {
    //     HdParsedAovToken aovId(name);
    //     if (aovId.isPrimvar) {
    //         return HdAovDescriptor(HdFormatFloat32Vec3, false, VtValue(GfVec3f(0.0f)));
    //     }
    // }
    return HdAovDescriptor();
}

HdRenderParam * HdKoshiRenderDelegate::GetRenderParam() const
{
    return param.get();
}

PXR_NAMESPACE_CLOSE_SCOPE
