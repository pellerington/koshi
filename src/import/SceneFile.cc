#include <import/SceneFile.h>

#include <base/ObjectGroup.h>

#include <import/GeometrySceneFile.h>
#include <import/LightSceneFile.h>
#include <import/IntegratorSceneFile.h>
#include <import/MaterialSceneFile.h>
#include <import/TextureSceneFile.h>

SceneFile::SceneFile()
{
    // Add base objects
    Type object("object");
    object.create_object_cb = create_object;
    types.add(object);

    Type object_group("object_group");
    object_group.reserved_attributes.push_back("objects");
    object_group.create_object_cb = create_object_group;
    types.add(object_group);

    // Fill in the types
    GeometrySceneFile::add_types(types);
    LightSceneFile::add_types(types);
    IntegratorSceneFile::add_types(types);
    MaterialSceneFile::add_types(types);
    TextureSceneFile::add_types(types);

    // TODO: Add types from plugins.
}

void SceneFile::Import(const std::string& filename, Scene& scene, /*Render*/Settings& settings)
{
    std::ifstream input_file(filename);
    nlohmann::json scene_file;
    input_file >> scene_file;

    // TODO: Make settings customizable.
    // Add our render settings.
    if(!scene_file["settings"].is_null())
    {
        AttributeAccessor accessor("settings", scene_file["settings"], scene_file, scene, this);
        settings.sampling_quality = accessor.get_float("sampling_quality");
        settings.depth = accessor.get_uint("depth", 2);
        settings.max_depth = accessor.get_uint("max_depth", 32);
        settings.max_samples_per_pixel = accessor.get_uint("max_samples_per_pixel", 64);
        settings.min_samples_per_pixel = accessor.get_uint("min_samples_per_pixel", 4);
        scene_file.erase("settings");
    }

    // TODO: Make camera customizable. And use the same interface as an object.
    // ALSO: Make the camera name get set somewhere else and scene.get_camera("name");
    Vec2u resolution(1);
    float focal_length = 1.f;
    Transform3f camera_transform;
    if(!scene_file["camera"].is_null())
    {
        AttributeAccessor accessor("camera", scene_file["camera"], scene_file, scene, this);
        resolution = accessor.get_vec2u("resolution");
        focal_length = accessor.get_float("focal_length", 1.f);

        const float scale = accessor.get_float("scale", 1.f);
        const Vec3f rotation = 2.f * PI * accessor.get_vec3f("rotation") / 360.f;
        const Vec3f translation = accessor.get_vec3f("translation");
        camera_transform = camera_transform * Transform3f::translation(translation);
        camera_transform = camera_transform * Transform3f::x_rotation(rotation[0]);
        camera_transform = camera_transform * Transform3f::y_rotation(rotation[1]);
        camera_transform = camera_transform * Transform3f::z_rotation(rotation[2]);
        camera_transform = camera_transform * Transform3f::scale(Vec3f(scale, scale, 1.f));
        scene_file.erase("camera");
    }
    scene.set_camera(new Camera(camera_transform, resolution, focal_length));

    for (auto it = scene_file.begin(); it != scene_file.end(); ++it)
    {
        if(!scene.get_object(it.key()))
            CreateObject(it.key(), *it, scene_file, scene);
    }

}

Object * SceneFile::CreateObject(const std::string& name, nlohmann::json& json_object, nlohmann::json& scene_file, Scene& scene, Object * parent)
{
    // Ignore non objects.
    if(!json_object.is_object()) return nullptr;

    // Find the object type.
    if(!json_object["type"].is_string()) return nullptr;
    Type * type = types.get(json_object["type"]);
    if(!type) return nullptr;

    // Remove the type so we don't add it as an attribute.
    json_object.erase("type");

    // Try and create an object.
    AttributeAccessor accessor(name, json_object, scene_file, scene, this);
    Object * object = type->create_object_cb(accessor, parent);
    if(!object) return nullptr;

    // Add our object to the scene and the mapping
    scene.add_object(name, object);

    // Remove the reserved attributes.
    for(auto attr = type->reserved_attributes.begin(); attr != type->reserved_attributes.end(); ++attr)
        json_object.erase(*attr);

    // Add the custom attributes.
    for(auto attr = json_object.begin(); attr != json_object.end(); ++attr)
    {
        // If it is pointing to an object.
        if(attr->is_string())
        {
            const std::string attr_name = *attr;
            // Try and find the object.
            Object * attr_object = scene.get_object(attr_name);
            if(!attr_object)
                attr_object = CreateObject(attr_name, scene_file[attr_name], scene_file, scene);
            object->set_attribute(attr.key(), attr_object);
        }

        // If it is an independent object.
        else if(attr->is_object())
        {
            Object * attr_object = CreateObject(name + "." + attr.key(), *attr, scene_file, scene, object);
            object->set_attribute(attr.key(), attr_object);
        }

        // TODO: Add plugin callbacks for each type.
    }

    return object;
}

Object * SceneFile::create_object(AttributeAccessor& accessor, Object * parent)
{
    return new Object;
}

Object * SceneFile::create_object_group(AttributeAccessor& accessor, Object * parent)
{
    ObjectGroup * object_group = accessor.get_objects("objects");
    return object_group;
}