#include <iostream>
#include <string>

#include "itkImageFileReader.h"

int main(int argc, char const *argv[]) {
    
    if (argc < 2) {
        std::cerr << "Required: input file path" << std::endl;
        
        return EXIT_FAILURE;
    }
    
    itk::ImageIOBase::Pointer imageIO =
    itk::ImageIOFactory::CreateImageIO(argv[1], itk::ImageIOFactory::ReadMode);
    
    imageIO->SetFileName(argv[1]);
    imageIO->ReadImageInformation();
    
    size_t numDimensions = imageIO->GetNumberOfDimensions();
    std::string componentTypeStr = imageIO->GetComponentTypeAsString(imageIO->GetComponentType());
    std::string pixelTypeStr = imageIO->GetPixelTypeAsString(imageIO->GetPixelType());
    
    std::cout << "File          : " << argv[1] << std::endl;
    std::cout << "Number dim    : " << numDimensions << std::endl;
    std::cout << "Component type: " << componentTypeStr << std::endl;
    std::cout << "Pixel type    : " << pixelTypeStr << std::endl;
    
    return EXIT_SUCCESS;
}