void $FUNCTION(void)
{
    BEGIN_TEST(__FUNCTION__);

    $DESCRIPTION

    try {
    /************ BEGIN INITIALIZATION SECTION ************/
    $INITIALIZE
    /************* END INITIALIZATION SECTION *************/
    
    
        double warmup_time;
        {
          timer t;
    /************ BEGIN TIMING SECTION ************/
    $TIME
    /************* END TIMING SECTION *************/
          warmup_time = t.elapsed();
        }
    
        // only verbose
        //std::cout << "warmup_time: " << warmup_time << " seconds" << std::endl;
    
        static const size_t NUM_TRIALS = 5;
        static const size_t MAX_ITERATIONS = 1000;
        static const double MAX_TEST_TIME = 0.5;  //TODO allow to be set by user
    
        size_t NUM_ITERATIONS;
        if (warmup_time == 0)
            NUM_ITERATIONS = MAX_ITERATIONS;
        else
            NUM_ITERATIONS = std::min(MAX_ITERATIONS, std::max( (size_t) 1, (size_t) (MAX_TEST_TIME / warmup_time)));
    
        double trial_times[NUM_TRIALS];
    
        for(size_t trial = 0; trial < NUM_TRIALS; trial++)
        {
            timer t;
            for(size_t i = 0; i < NUM_ITERATIONS; i++){
                 
    /************ BEGIN TIMING SECTION ************/
    $TIME
    /************* END TIMING SECTION *************/
    
            }
    
            trial_times[trial] = t.elapsed() / double(NUM_ITERATIONS);
        }
    
        // only verbose
        //for(size_t trial = 0; trial < NUM_TRIALS; trial++){
        //    std::cout << "trial[" << trial << "]  : " << trial_times[trial] << " seconds\n";
        //}
    
        double best_time = *std::min_element(trial_times, trial_times + NUM_TRIALS);
    
    /************ BEGIN FINALIZE SECTION ************/
    $FINALIZE
    /************* END FINALIZE SECTION *************/
    
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
        cudaError_t error = cudaGetLastError();
        if(error){
            RECORD_TEST_FAILURE(cudaGetErrorString(error));
        } else {
            RECORD_TEST_SUCCESS();
        }
#else
        RECORD_TEST_SUCCESS();
#endif

    }  // end try
    catch (std::bad_alloc) {
        RECORD_TEST_FAILURE("std::bad_alloc");
    }
    catch (unittest::UnitTestException e) {
        RECORD_TEST_FAILURE(e);
    }


    END_TEST();
}
