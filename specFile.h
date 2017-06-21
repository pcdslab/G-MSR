#include <string.h>
#include <string>
#include <vector>
#include "spectrum.h"
class specFile
{
public:
	specFile();
	~specFile();
	std::string fileName;
	std::string headerLine;
	std::vector<spectrum> listVals;
};

