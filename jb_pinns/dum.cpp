#include <iostream>
#include <vector>
#include <string>
int main() {
    // Write C++ code here
/*    std::vector<std::string> bob;
    for (int i=0; i<5; i++){
        
        bob.emplace_back(std::to_string(i));
        std::cout<< std::to_string(i)<< std::endl;
        std::cout<< bob[i]<<std::endl;
    }
    std::cout << "Try programiz.pro";
	int j=0;
	int k =0;
	*/
	
	/**while(j<(int)bob.size()){
		std::cout<< j<<std::endl;
		//std::cout<< k<<std::endl;
		std::cout<<std::endl;
		j++;
		k++;
	}
	*/
	std::vector<std::string> bob;
    for (int j=0; j<5; j++)
    {
		std::cout<<std::to_string(j)<<std::endl;
		bob.emplace_back(std::to_string(j));
//        std::cout<< (int)bob.size()<< std::endl;
        std::cout<<bob[j] << std::endl;
	//	std::cout<<std::endl;
		
//		if(j>=(int)bob.size())
//			break;
    }
	for (int j=0; j<5; j++)
    {
		
        std::cout<<bob[j] << std::endl;
	
    }

    return 0;
}