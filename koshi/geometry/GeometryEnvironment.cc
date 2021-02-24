#include <koshi/geometry/GeometryEnvironment.h>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/sysutil.h>
#include <OpenImageIO/imagebuf.h>

#include <koshi/OptixHelpers.h>

KOSHI_OPEN_NAMESPACE

void GeometryEnvironment::createTexture(const std::string& filename)
{

    // https://github.com/AcademySoftwareFoundation/OpenShadingLanguage/blob/master/src/testshade/optixgridrender.cpp

    // Open image
    OIIO::ImageBuf image;
    if (!image.init_spec(filename, 0, 0)) {
        std::cout << "Could not load texture: " <<  filename << std::endl;
        return;
    }

    OIIO::ROI roi = OIIO::get_roi_full(image.spec());
    int32_t width = roi.width(), height = roi.height();
    std::vector<float> pixels(width * height * 4);

    std::cout << "Texture Resolution: " << roi.width() << " " << roi.height() << std::endl;
 
    for (int j = 0; j < height; j++)
        for (int i = 0; i < width; i++)
            image.getpixel(i, j, 0, &pixels[((j*width) + i) * 4 + 0]);

    cudaResourceDesc res_desc = {};

    // hard-code textures to 4 channels
    int32_t pitch  = width * 4 * sizeof(float);
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

    cudaArray_t   pixelArray;
    CUDA_CHECK (cudaMallocArray (&pixelArray,
                                    &channel_desc,
                                    width,height));

    CUDA_CHECK (cudaMemcpy2DToArray (pixelArray,
                                        /* offset */0,0,
                                        pixels.data(),
                                        pitch,pitch,height,
                                        cudaMemcpyHostToDevice));

    res_desc.resType          = cudaResourceTypeArray;
    res_desc.res.array.array  = pixelArray;

    cudaTextureDesc tex_desc     = {};
    tex_desc.addressMode[0]      = cudaAddressModeWrap;
    tex_desc.addressMode[1]      = cudaAddressModeWrap;
    tex_desc.filterMode          = cudaFilterModeLinear;
    tex_desc.readMode            = cudaReadModeElementType; //cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords    = 1;
    tex_desc.maxAnisotropy       = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode    = cudaFilterModePoint;
    tex_desc.borderColor[0]      = 1.0f;
    tex_desc.sRGB                = 0;

    // Create texture object
    cuda_tex = 0;
    CUDA_CHECK (cudaCreateTextureObject (&cuda_tex, &res_desc, &tex_desc, nullptr));
}

KOSHI_CLOSE_NAMESPACE