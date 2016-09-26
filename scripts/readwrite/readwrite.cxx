#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkCastImageFilter.h"
#include <dirent.h>

static const unsigned int Dimension = 3;
typedef double                                    InputPixelType;
typedef unsigned char                             OutputPixelType;
typedef itk::Image< InputPixelType, Dimension >   InputImageType;
typedef itk::Image< OutputPixelType, Dimension >  OutputImageType;
typedef itk::ImageFileReader< InputImageType >    ReaderType;
typedef itk::RescaleIntensityImageFilter< InputImageType, InputImageType > RescaleType;
typedef itk::CastImageFilter< InputImageType, OutputImageType > FilterType;
typedef itk::ImageFileWriter< OutputImageType >   WriterType;

using namespace std;

bool has_suffix(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
    str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void replaceExt(string& s, const string& newExt) {
    
    string::size_type i = s.rfind('.', s.length());
    
    if (i != string::npos) {
        s.replace(i+1, newExt.length(), newExt);
    }
}

bool cast_file(string &input, string &output) {
    
    const char * inputImage = input.c_str();
    const char * outputImage = output.c_str();
    
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( inputImage );
    
    RescaleType::Pointer rescale = RescaleType::New();
    rescale->SetInput( reader->GetOutput() );
    rescale->SetOutputMinimum( 0 );
    rescale->SetOutputMaximum( itk::NumericTraits< OutputPixelType >::max() );
    
    FilterType::Pointer filter = FilterType::New();
    filter->SetInput( rescale->GetOutput() );
    
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName( outputImage );
    writer->SetInput( filter->GetOutput() );
    
    try
    {
        writer->Update();
    }
    catch( itk::ExceptionObject & e )
    {
        std::cerr << "Error: " << e << std::endl;
        return false;
    }
    
    return true;
}

int main( int argc, char* argv[] )
{
    if( argc != 3 )
    {
        std::cerr << "Usage: "<< std::endl;
        std::cerr << argv[0];
        std::cerr << "<InputDirectory> <OutputDirectory>";
        std::cerr << std::endl;
        return EXIT_FAILURE;
    }
    
    string input_directory = argv[1], output_directory = argv[2];
    DIR *dir;
    struct dirent *ent;
    
    if ((dir = opendir (input_directory.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            string filename = ent->d_name;
            if ( has_suffix(filename, ".mhd" )) {
                
                string input_path = input_directory + filename;
                
                replaceExt(filename, "nii");
                
                string output_path = output_directory + filename;
                
                bool succes = cast_file(input_path, output_path);
                if (!succes) return EXIT_FAILURE;
            }
            
        }
        closedir (dir);
    } else {
        /* could not open directory */
        perror ("");
        return EXIT_FAILURE;
    }
    

}
