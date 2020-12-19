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
#include "mesh.h"

#include <iostream>
#include "pxr/imaging/hd/mesh.h"
#include "pxr/base/gf/matrix4f.h"
#include "pxr/imaging/hd/meshUtil.h"
#include "renderParam.h"

PXR_NAMESPACE_OPEN_SCOPE

HdKoshiMesh::HdKoshiMesh(/*Koshi::Scene * scene,*/ SdfPath const& id, SdfPath const& instancerId)
: HdMesh(id, instancerId), geometry(nullptr)
{
}

HdDirtyBits HdKoshiMesh::GetInitialDirtyBitsMask() const
{
    return HdChangeTracker::Clean | HdChangeTracker::DirtyTransform;
}

HdDirtyBits HdKoshiMesh::_PropagateDirtyBits(HdDirtyBits bits) const
{
    return bits;
}

void HdKoshiMesh::_InitRepr(TfToken const &reprToken, HdDirtyBits *dirtyBits)
{

}

void HdKoshiMesh::Sync(HdSceneDelegate * sceneDelegate, HdRenderParam * renderParam, HdDirtyBits * dirtyBits, const TfToken& reprToken)
{
    std::cout << "* (multithreaded) Sync Tiny Mesh id=" << GetId() << std::endl;

    const SdfPath& id = GetId();
    Koshi::Scene * scene = ((HdKoshiRenderParam*)renderParam)->getScene();

    if(!geometry)
    {
        geometry = std::make_shared<Koshi::GeometryMesh>();

        GfMatrix4f transform = GfMatrix4f(sceneDelegate->GetTransform(id));
        geometry->setTransform(Koshi::Transform::fromData(transform.data(), false));

        VtValue value = sceneDelegate->Get(id, HdTokens->points);
        points = value.Get<VtVec3fArray>();
        HdDisplayStyle const displayStyle = sceneDelegate->GetDisplayStyle(id);
        topology = HdMeshTopology(GetMeshTopology(sceneDelegate), displayStyle.refineLevel);
        HdMeshUtil meshUtil(&topology, GetId());
        meshUtil.ComputeTriangleIndices(&triangulatedIndices, &trianglePrimitiveParams);

        geometry->setAttribute("vertices", Koshi::Format::FLOAT32, points.size(), 3, points.cdata(), triangulatedIndices.size(), (uint32_t*)triangulatedIndices.cdata());

        scene->addGeometry(GetId().GetAsString(), geometry.get());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
