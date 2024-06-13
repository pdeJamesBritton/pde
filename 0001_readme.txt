James Britton
Carleton University
100759027

BISRI 2024 Summer

Hours worked:
Total:
    18
List of hours:
    9,3,1,1,1,3

Up to and including April 11th
    5 hours reading paper
    1 hour gathering papers from references
    2 hours searching for useful courses/materials
    1 hour setting up personal computer to do more work
    
April 14th*
    .5hours - setting up yubico, and codes for multifactor authentication

April 16th *   
    Meeting with Prof. to discuss terms of project. Started at 1pm
    Look at:
        https://www.geeksforgeeks.org/machine-learning/?ref=shm
    Prof. will send my a simplified bec2.py to practise.
    
    -Requirments for libraries:
        Definition of Neural Networks
        Automatic Differentiation
        Obtimization of Functionals
    Libraries:
        mlpack
        Tensor flow/pytorch for C++
            Look at this
            PND (PINNs): Physics-Informed Neural-Network Molecular Dynamics
            https://github.com/USCCACS/PND
        
    Learn what the benchmarks are
        Compare Python, C++, and possible Julia
    Practise using libraries
    
    Once perficient will construct LOSS function (w/ Prof) for model(s) or interest.
    
    Has Anyone implemented PINNs in C++, if so, use there code or learn from...
        See link above
    
    
    What is a functional in general and in this context?
    Paramertization of a Function/Functional need to understand.
        
    useful links
        https://www.geeksforgeeks.org/machine-learning/?ref=shm
            
        Julia Programming Language, I should learn https://www.stochasticlifestyle.com/engineering-trade-offs-in-automatic-differentiation-from-tensorflow-and-pytorch-to-jax-and-julia/
    
Add this and all useful files to GitHub
    fix my GitHub set up
        Use on Windows and Virtual machine
3hours

April 18th-19th*
    Setting up VM-Ubunto
        having issues
        
    Start anew
        Username    james
        password    XJb52o13aSvJ_2
        hostname    Ubunto-2024-04-19-PM
        Domain Name myguest.virtualbox.org
        
        - get sudoers permission
            su -
            usermod -a -G sudo vboxuser
            DONE
        - get sharing working
            C:\Users\hp_br\OneDrive\James
            James Britton
            useful video
                https://www.youtube.com/watch?v=N4C5CeYfntE
        - now VM is buggy with the display
            working for now, referenced
                https://forums.virtualbox.org/viewtopic.php?t=110882&sid=653894214b3f3d88de74c2a3342c009b
                
                I had the same problem, but I fixed. Here is how I did it:

                On Host:
                Device -> Insert Guest Additions CD Image..
                
                On VM:
                Open the disc in the file explorer -> right-click on empty space -> Open in Terminal
                
                In the Terminal:
                sudo ./VBoxLinuxAdditions.run -> sudo rcvoxadd reload
        
                I hope this will work for the rest.

                P.S
                If you already have Guest Additions already installed, try sudo rcvboxadd reload in the terminal and see if that fixed it. Otherwise I think you need to re-install it.
        
        - sudo apt full-upgrade , to upgrade all
        
        -installing gcc/g++ 13
            
    gcc –version – Displays the current GCC version.
    
    sudo apt update – Updates the package repository information.
    
    sudo apt install software-properties-common – Installs the software properties common package.
    
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test – Adds the GCC repository.
    
    sudo apt install gcc-13 g++-13 – Installs GCC version 13.
    
    sudo update-alternatives –install /usr/bin/gcc gcc /usr/bin/gcc-13 100 –slave /usr/bin/g++ g++ /usr/bin/g++-13 – Sets GCC 13 as the default version.


    getting cmake set up (do I need this?)
        https://github.com/Kitware/CMake/releases/download/v3.29.2/cmake-3.29.2-linux-x86_64.tar.gz
        
        https://github.com/mlpack/mlpack/archive/refs/tags/4.3.0.tar.gz
        
            Not sure if I got this right, above
    
    Installing mlpack with video
        https://www.youtube.com/watch?v=4ibiANsznaQ
        
        Error messages
            CMake Warning (dev) at /usr/share/cmake-3.22/Modules/FindPackageHandleStandardArgs.cmake:438 (message):
                The package name passed to `find_package_handle_standard_args` (LIBBFD)
                does not match the name of the calling package (Bfd).  This can lead to
                problems in calling code that expects `find_package` result variables
                (e.g., `_FOUND`) to follow a certain pattern.

            Call Stack (most recent call first):
                CMake/FindBfd.cmake:82 (FIND_PACKAGE_HANDLE_STANDARD_ARGS)
                CMakeLists.txt:236 (find_package)
            This warning is for project developers.  Use -Wno-dev to suppress it.

            CMake Warning at CMakeLists.txt:455 (message):
                txt2man not found; 
                man pages will not be generated.
1hour
April 20th *
        continuing install of mlpack    
            running make,  then plan to do sudo make install
            
            ran
                sudo cmake -DBUILD_SHARED_LIBS=OFF -DDOWNLOAD_DEPENDENCIES=ON -DCMAKE_INSTALL_PREFIX="$install_prefix" -S . -B build
                
                sudo cmake --build build --target install
                
                    will it work??  No
                    this is the error message
                        /usr/bin/ld: attempted static link of dynamic object `/usr/lib/x86_64-linux-gnu/libbfd.so'

                        collect2: error: ld returned 1 exit status

                        gmake[2]: *** [src/mlpack/methods/CMakeFiles/mlpack_approx_kfn.dir/build.make:102: bin/mlpack_approx_kfn] Error 1

                        gmake[1]: *** [CMakeFiles/Makefile2:402: src/mlpack/methods/CMakeFiles/mlpack_approx_kfn.dir/all] Error 2

                        gmake: *** [Makefile:146: all] Error 2

1hour
April 24th
    -Meeting with Vicky about setting up machine
    
********
Week of April 29th, Goals
-duel boot on personal machine
-finish installing/install:
    done May 2nd
    Vim/
  ^^^^^^^  issue with Emacs  ^^^^^^^
    
    done May 2nd
    cuda

    python3
        JAX/FLAX
        XLA
        pytorch
        Keras Neural Network Framework
        Tensorflow
        numply
        matplotlib
    
    done May 2nd    
    gcc/g++
    C++
        done May 2nd
        mlpack
            -Armidillo
            -Cerial
            - ??


        pytorch??


    C# .NET 
    
    grep/sed/awk
    bash
    grep

    github 
    
    can I ssh into my maching from another?
    
    Make a decision, Linux only,  Linux with VM for windows or Dual boot???
    
1 hour    
April 29th *
-completed 
    Respect and Safety (Formerly known as Workplace Violence and Harassment Prevention) 1 hour

3 hours
April 30th *
-Completed
    EHS: Workplace Hazardous Materials Information System
    1 hour
    
    EHS: Worker Health and Safety Awareness
    1hour
-Neural Network lecture 3
    1 hour
-Julia  
    -1hour
    
-Need to review:
    NN
    Julia
    JAX/FLAX

May 1st

May 2nd
flax course
logging into Flax cluster....
hostname: diplodocus.c3.ca
password: unseenuni






May 8th
It seems like I keep forgetting to save my notes, but check in commands txt
GPU/machine learning can run on 16bit percision instead of 64 bit, without loosing preformance

mpi is very difficult to work
mpi4py for python3
there is for C++
-is this useful for our project???
-> first use easier was to parrallize


May 21st, last week I went to CHEO twice and and a missing son.
wrist is better, cut down tree, did planting and work in garden.

This week preparing for presentation and assignment


May 21st to June 8th
*Prepared for interview
->did lots of work in garden
- Verified that libraries (i.e. mlpack...) worked
- set up github with yubikey
- for easy of push/pull switched from https to ssh
- in setting up git/github lost cpp file that I did make

Plan for week of June 9th
- work on skeleton of cpp bec_2d
